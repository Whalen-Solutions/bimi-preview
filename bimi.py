"""BIMI SVG converter module.

Converts raster images (JPEG, PNG, WebP, GIF, BMP, TIFF) and existing SVGs
to BIMI-compliant SVG Tiny-PS format.

Uses scikit-image contour tracing as a pure-Python replacement for potrace.
If potrace is available on the system PATH, it will be preferred for higher
quality output.
"""

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from skimage import measure


# ---------------------------------------------------------------------------
# Otsu thresholding
# ---------------------------------------------------------------------------


def otsu_threshold(arr: np.ndarray) -> int:
    """Compute Otsu's optimal binarization threshold."""
    hist, _ = np.histogram(arr.flatten(), bins=256, range=(0, 255))
    hist = hist.astype(float) / hist.sum()
    total_mean = sum(i * hist[i] for i in range(256))
    best_var, best_t = 0.0, 128
    w0, mean0 = 0.0, 0.0
    for t in range(256):
        w0 += hist[t]
        w1 = 1.0 - w0
        if w0 == 0 or w1 == 0:
            continue
        mean0 = (mean0 * (w0 - hist[t]) + t * hist[t]) / w0 if w0 else 0
        mean1 = (total_mean - w0 * mean0) / w1
        var = w0 * w1 * (mean0 - mean1) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t


# ---------------------------------------------------------------------------
# Source assessment & preprocessing (per Section 6a)
# ---------------------------------------------------------------------------


def _upsample_factor(img: Image.Image) -> int:
    min_dim = min(img.width, img.height)
    if min_dim < 400:
        return 8
    elif min_dim < 800:
        return 4
    return 2


def _detect_format(path: str) -> str:
    ext = Path(path).suffix.lower()
    fmt_map = {
        ".jpg": "jpeg",
        ".jpeg": "jpeg",
        ".png": "png",
        ".webp": "webp",
        ".gif": "gif",
        ".bmp": "bmp",
        ".tiff": "tiff",
        ".tif": "tiff",
        ".svg": "svg",
    }
    return fmt_map.get(ext, "unknown")


def _has_transparency(img: Image.Image) -> bool:
    if img.mode in ("RGBA", "LA", "PA"):
        alpha = np.array(img.split()[-1])
        return bool((alpha < 255).any())
    return False


def _dominant_color(img: Image.Image) -> str:
    """Infer the dominant background color from the image edges."""
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    border = max(4, min(h, w) // 20)
    edges = np.concatenate(
        [
            arr[:border].reshape(-1, 3),
            arr[-border:].reshape(-1, 3),
            arr[border:-border, :border].reshape(-1, 3),
            arr[border:-border, -border:].reshape(-1, 3),
        ]
    )
    mean = edges.mean(axis=0).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*mean)


def preprocess_raster(path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Preprocess a raster image for tracing.

    Returns:
        binary:        boolean ndarray, True = foreground (mark) pixels
        gt_mask:       boolean ndarray, ground-truth Otsu mask (before blur)
        upsampled_dim: side length of the square upsampled canvas
    """
    img = Image.open(path)
    fmt = _detect_format(path)
    transparent = _has_transparency(img)
    upsample = _upsample_factor(img)

    # Make square canvas
    max_dim = max(img.width, img.height)
    upsampled_dim = max_dim * upsample

    if transparent:
        # Resize preserving alpha
        img_up = img.resize((upsampled_dim, upsampled_dim), Image.LANCZOS)
        alpha = np.array(img_up.split()[-1])
        gt_mask = alpha > 127
        return gt_mask, gt_mask, upsampled_dim

    img_up = img.resize((upsampled_dim, upsampled_dim), Image.LANCZOS)
    img_gray = img_up.convert("L")

    # Ground-truth mask (before blurring)
    raw_arr = np.array(img_gray)
    gt_t = otsu_threshold(raw_arr)
    light_pct = (raw_arr > gt_t).sum() / raw_arr.size
    if light_pct > 0.5:
        gt_mask = raw_arr <= gt_t  # dark mark on light bg
    else:
        gt_mask = raw_arr > gt_t  # light mark on dark bg

    # Filtering pipeline
    if fmt == "jpeg":
        img_gray = img_gray.filter(ImageFilter.MedianFilter(size=5))
        img_gray = img_gray.filter(ImageFilter.MedianFilter(size=3))
    blur_radius = max(1, upsample // 4)

    img_gray = img_gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    arr = np.array(img_gray)

    threshold = otsu_threshold(arr)
    light_pct2 = (arr > threshold).sum() / arr.size
    if light_pct2 > 0.5:
        binary = arr <= threshold
    else:
        binary = arr > threshold

    return binary, gt_mask, upsampled_dim


# ---------------------------------------------------------------------------
# Contour simplification (Douglas-Peucker)
# ---------------------------------------------------------------------------


def _dp_simplify(contour: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker simplification of an open polyline."""
    if len(contour) < 3:
        return contour

    start, end = contour[0], contour[-1]
    seg = end - start
    seg_len = np.linalg.norm(seg)

    if seg_len < 1e-10:
        # Degenerate segment — use distance from start instead
        dists = np.sqrt(((contour - start) ** 2).sum(axis=1))
    else:
        # Perpendicular distance: |seg x (start - pt)| / |seg|
        diff = start - contour
        dists = np.abs(seg[0] * diff[:, 1] - seg[1] * diff[:, 0]) / seg_len

    max_idx = int(np.argmax(dists))
    max_dist = dists[max_idx]

    if max_dist > tolerance:
        left = _dp_simplify(contour[: max_idx + 1], tolerance)
        right = _dp_simplify(contour[max_idx:], tolerance)
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])


def _simplify_contour(contour: np.ndarray, tolerance: float) -> np.ndarray:
    """Simplify a (possibly closed) contour with Douglas-Peucker.

    Closed contours are split at the point farthest from the centroid
    so that D-P doesn't collapse them to a single segment.
    """
    if len(contour) < 3:
        return contour

    # Detect closed contour (start ≈ end)
    if np.linalg.norm(contour[0] - contour[-1]) < tolerance:
        # Split at the point farthest from centroid
        centroid = contour.mean(axis=0)
        dists = ((contour - centroid) ** 2).sum(axis=1)
        split = int(np.argmax(dists))
        if split == 0 or split == len(contour) - 1:
            split = len(contour) // 2

        first = _dp_simplify(contour[: split + 1], tolerance)
        second = _dp_simplify(contour[split:], tolerance)
        return np.vstack([first[:-1], second])

    return _dp_simplify(contour, tolerance)


# ---------------------------------------------------------------------------
# Cubic Bezier curve fitting (Schneider's algorithm)
# ---------------------------------------------------------------------------


def _chord_length_params(points: np.ndarray) -> np.ndarray:
    """Assign parameter values via chord-length parameterization."""
    diffs = np.diff(points, axis=0)
    dists = np.sqrt((diffs**2).sum(axis=1))
    cumulative = np.concatenate([[0.0], np.cumsum(dists)])
    total = cumulative[-1]
    if total < 1e-10:
        return np.linspace(0.0, 1.0, len(points))
    return cumulative / total


def _estimate_left_tangent(points: np.ndarray) -> np.ndarray:
    """Unit tangent at first point, pointing forward."""
    t = points[min(1, len(points) - 1)] - points[0]
    n = np.linalg.norm(t)
    return t / n if n > 1e-10 else np.array([1.0, 0.0])


def _estimate_right_tangent(points: np.ndarray) -> np.ndarray:
    """Unit tangent at last point, pointing backward."""
    t = points[max(-2, -len(points))] - points[-1]
    n = np.linalg.norm(t)
    return t / n if n > 1e-10 else np.array([-1.0, 0.0])


def _estimate_center_tangent(points: np.ndarray, idx: int) -> np.ndarray:
    """Unit tangent at a split point, pointing forward."""
    v = points[min(idx + 1, len(points) - 1)] - points[max(idx - 1, 0)]
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else np.array([1.0, 0.0])


def _fit_single_cubic(
    points: np.ndarray, params: np.ndarray, tan_left: np.ndarray, tan_right: np.ndarray
) -> np.ndarray:
    """Fit one cubic Bezier to points. Returns (4, 2) control points."""
    p0 = points[0]
    p3 = points[-1]
    seg_len = np.linalg.norm(p3 - p0)

    if len(points) == 2:
        dist = seg_len / 3.0
        return np.array([p0, p0 + dist * tan_left, p3 + dist * tan_right, p3])

    t = params
    b0 = (1 - t) ** 3
    b1 = 3 * (1 - t) ** 2 * t
    b2 = 3 * (1 - t) * t**2
    b3 = t**3

    a1 = b1[:, None] * tan_left[None, :]
    a2 = b2[:, None] * tan_right[None, :]
    residuals = points - b0[:, None] * p0 - b3[:, None] * p3

    C = np.array(
        [
            [(a1 * a1).sum(), (a1 * a2).sum()],
            [(a1 * a2).sum(), (a2 * a2).sum()],
        ]
    )
    X = np.array([(a1 * residuals).sum(), (a2 * residuals).sum()])

    det = C[0, 0] * C[1, 1] - C[0, 1] ** 2
    if abs(det) > 1e-12:
        alpha = (C[1, 1] * X[0] - C[0, 1] * X[1]) / det
        beta = (C[0, 0] * X[1] - C[1, 0] * X[0]) / det
        if alpha >= 1e-6 * seg_len and beta >= 1e-6 * seg_len:
            return np.array([p0, p0 + alpha * tan_left, p3 + beta * tan_right, p3])

    dist = seg_len / 3.0
    return np.array([p0, p0 + dist * tan_left, p3 + dist * tan_right, p3])


def _max_bezier_error(
    points: np.ndarray, params: np.ndarray, cp: np.ndarray
) -> tuple[float, int]:
    """Max squared distance from data points to fitted curve."""
    u = 1 - params
    fitted = (
        (u**3)[:, None] * cp[0]
        + (3 * u**2 * params)[:, None] * cp[1]
        + (3 * u * params**2)[:, None] * cp[2]
        + (params**3)[:, None] * cp[3]
    )
    sq_err = ((points - fitted) ** 2).sum(axis=1)
    idx = int(np.argmax(sq_err))
    return float(sq_err[idx]), idx


def _fit_cubic_beziers(
    points: np.ndarray,
    tolerance: float,
    tan_left: np.ndarray,
    tan_right: np.ndarray,
    depth: int = 0,
) -> list[np.ndarray]:
    """Recursively fit cubic Beziers (Schneider). Returns list of (4,2)."""
    if len(points) <= 2:
        dist = np.linalg.norm(points[-1] - points[0]) / 3.0
        return [
            np.array(
                [
                    points[0],
                    points[0] + dist * tan_left,
                    points[-1] + dist * tan_right,
                    points[-1],
                ]
            )
        ]

    params = _chord_length_params(points)
    cp = _fit_single_cubic(points, params, tan_left, tan_right)
    max_err, split_idx = _max_bezier_error(points, params, cp)

    if max_err <= tolerance**2 or depth > 20:
        return [cp]

    split_idx = max(1, min(split_idx, len(points) - 2))
    tan_c = _estimate_center_tangent(points, split_idx)

    left = _fit_cubic_beziers(
        points[: split_idx + 1], tolerance, tan_left, -tan_c, depth + 1
    )
    right = _fit_cubic_beziers(
        points[split_idx:], tolerance, tan_c, tan_right, depth + 1
    )
    return left + right


def _trace_with_potrace(
    binary: np.ndarray,
) -> tuple[list[str], str] | None:
    """Trace binary mask using potrace for high-quality output.

    Returns ``(path_d_strings, g_transform)`` or *None* if potrace is
    not available or fails.  The transform string converts potrace's
    internal PostScript coordinates to the image pixel space.
    """
    if not shutil.which("potrace"):
        return None

    # Potrace traces black-on-white: foreground → black, background → white
    fg_as_black = (~binary).astype(np.uint8) * 255
    pil_img = Image.fromarray(fg_as_black, mode="L")

    with tempfile.TemporaryDirectory() as tmpdir:
        bmp_path = str(Path(tmpdir) / "mask.bmp")
        svg_path = str(Path(tmpdir) / "traced.svg")
        pil_img.save(bmp_path, format="BMP")

        try:
            subprocess.run(
                ["potrace", bmp_path, "-s", "--flat", "-o", svg_path],
                check=True,
                capture_output=True,
                timeout=30,
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            return None

        svg_text = Path(svg_path).read_text()

    paths = re.findall(r'\bd="([^"]+)"', svg_text)
    if not paths:
        return None

    # Potrace wraps paths in <g transform="translate(0,H) scale(0.1,-0.1)">
    # which converts its internal PostScript coords to pixel space.
    xform_match = re.search(r'<g[^>]+transform="([^"]+)"', svg_text)
    xform = xform_match.group(1) if xform_match else ""

    return paths, xform


def trace_to_svg_paths(
    binary: np.ndarray, upsampled_dim: int, tolerance: float = 2.0
) -> list[str]:
    """Trace binary mask to SVG path strings with cubic Bezier curves."""
    # Pad with background so contours close properly at image edges
    # instead of merging along the boundary.
    padded = np.pad(binary, pad_width=1, mode="constant", constant_values=False)
    contours = measure.find_contours(padded.astype(float), 0.5)

    paths = []
    for contour in contours:
        if len(contour) < 4:
            continue

        # Adjust coordinates back from padded space
        contour = contour - 1.0

        # Phase 1: Douglas-Peucker to reduce thousands of points to key vertices.
        # Use a coarser tolerance so we keep enough shape detail for Bezier fitting.
        dp_tolerance = tolerance * 0.5
        simplified = _simplify_contour(contour, dp_tolerance)
        if len(simplified) < 3:
            continue

        # Phase 2: fit smooth cubic Beziers through the simplified points
        tan_left = _estimate_left_tangent(simplified)
        tan_right = _estimate_right_tangent(simplified)
        beziers = _fit_cubic_beziers(simplified, tolerance, tan_left, tan_right)
        if not beziers:
            continue

        # contour coords are (row, col) = (y, x); swap for SVG
        cp0 = beziers[0]
        parts = [f"M{cp0[0][1]:.1f},{cp0[0][0]:.1f}"]
        for cp in beziers:
            parts.append(
                f"C{cp[1][1]:.1f},{cp[1][0]:.1f} "
                f"{cp[2][1]:.1f},{cp[2][0]:.1f} "
                f"{cp[3][1]:.1f},{cp[3][0]:.1f}"
            )
        parts.append("Z")
        paths.append("".join(parts))

    return paths


# ---------------------------------------------------------------------------
# BIMI SVG assembly
# ---------------------------------------------------------------------------

BIMI_TEMPLATE = """\
<svg version="1.2" baseProfile="tiny-ps"
     xmlns="http://www.w3.org/2000/svg"
     viewBox="0 0 {size} {size}">
  <title>{title}</title>
  <desc>{desc}</desc>
  <rect width="{size}" height="{size}" fill="{bg_color}"/>
  <g transform="translate({tx},{ty}) scale({scale})">
    {paths}
  </g>
</svg>"""


def _compute_path_bounds(path_strs: list[str]) -> tuple[float, float, float, float]:
    """Bounding box of all SVG paths."""
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    for p in path_strs:
        coords = re.findall(r"(-?[\d.]+),(-?[\d.]+)", p)
        for xs, ys in coords:
            x, y = float(xs), float(ys)
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
    return min_x, min_y, max_x, max_y


def raster_to_bimi_svg(path: str, company_name: str) -> str:
    """Convert a raster image to a BIMI-compliant SVG string."""
    binary, gt_mask, upsampled_dim = preprocess_raster(path)

    img = Image.open(path)
    bg_color = _dominant_color(img)

    # Determine mark color from foreground pixels
    arr_rgb = np.array(img.convert("RGB"))
    mask_resized = (
        np.array(
            Image.fromarray(binary.astype(np.uint8) * 255).resize(
                (img.width, img.height), Image.NEAREST
            )
        )
        > 127
    )
    if mask_resized.any():
        fg_pixels = arr_rgb[mask_resized]
        mark_color = "#{:02x}{:02x}{:02x}".format(*fg_pixels.mean(axis=0).astype(int))
    else:
        bg_rgb = tuple(int(bg_color[i : i + 2], 16) for i in (1, 3, 5))
        bg_lum = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
        mark_color = "#000000" if bg_lum > 127 else "#ffffff"

    # Prefer potrace for higher quality tracing
    potrace_xform = ""
    potrace_result = _trace_with_potrace(binary)
    if potrace_result is not None:
        path_strs, potrace_xform = potrace_result
        h, w = binary.shape
        min_x, min_y, max_x, max_y = 0.0, 0.0, float(w), float(h)
    else:
        # Fallback: scikit-image contour tracing + Bezier fitting
        tolerance = max(2.0, upsampled_dim / 400)
        path_strs = trace_to_svg_paths(binary, upsampled_dim, tolerance)

        # If too much path data, retry with coarser tolerance
        while sum(len(p) for p in path_strs) > BIMI_MAX_SIZE and tolerance < 20:
            tolerance *= 1.5
            path_strs = trace_to_svg_paths(binary, upsampled_dim, tolerance)

        if not path_strs:
            raise ValueError("Could not trace any paths from the image.")

        min_x, min_y, max_x, max_y = _compute_path_bounds(path_strs)

    # Compute bounds and center with 85% scale for circle-crop clearance
    content_w = max_x - min_x
    content_h = max_y - min_y

    canvas_size = 800
    av_scale = 0.85
    usable = canvas_size * av_scale
    scale_factor = usable / max(content_w, content_h)

    mark_cx = (min_x + max_x) / 2
    mark_cy = (min_y + max_y) / 2
    tx = canvas_size / 2 - mark_cx * scale_factor
    ty = canvas_size / 2 - mark_cy * scale_factor

    combined_d = " ".join(path_strs)
    path_elem = f'<path d="{combined_d}" fill="{mark_color}" fill-rule="evenodd"/>'

    # Potrace paths use PostScript coords — wrap in its transform
    if potrace_xform:
        paths_svg = f'<g transform="{potrace_xform}">{path_elem}</g>'
    else:
        paths_svg = path_elem

    return BIMI_TEMPLATE.format(
        size=canvas_size,
        title=company_name,
        desc=f"{company_name} logo mark",
        bg_color=bg_color,
        tx=f"{tx:.2f}",
        ty=f"{ty:.2f}",
        scale=f"{scale_factor:.6f}",
        paths=paths_svg,
    )


# ---------------------------------------------------------------------------
# SVG-to-BIMI conversion (minimal changes per spec)
# ---------------------------------------------------------------------------

BIMI_MAX_SIZE = 32 * 1024  # 32 KB


def _svg_has_background_rect(svg: str, vb_w: float, vb_h: float) -> bool:
    """Check if the SVG already has a full-viewport background <rect>."""
    for m in re.finditer(r'<rect\b[^>]*width="([^"]*)"[^>]*height="([^"]*)"', svg):
        try:
            rw, rh = float(m.group(1)), float(m.group(2))
            if rw >= vb_w * 0.95 and rh >= vb_h * 0.95:
                return True
        except ValueError:
            continue
    for m in re.finditer(r'<rect\b[^>]*height="([^"]*)"[^>]*width="([^"]*)"', svg):
        try:
            rh, rw = float(m.group(1)), float(m.group(2))
            if rw >= vb_w * 0.95 and rh >= vb_h * 0.95:
                return True
        except ValueError:
            continue
    return False


def _infer_bg_color_from_svg(path: str) -> str:
    """Render the SVG at low res and sample edge pixels for background color."""
    try:
        import cairosvg
        from io import BytesIO

        png_data = cairosvg.svg2png(url=path, output_width=64, output_height=64)
        img = Image.open(BytesIO(png_data)).convert("RGB")
        return _dominant_color(img)
    except Exception:
        return "#ffffff"


def svg_to_bimi_svg(path: str, company_name: str) -> str:
    """Convert an existing SVG to BIMI-compliant SVG Tiny-PS.

    Only applies changes that BIMI strictly requires.
    """
    with open(path, "r") as f:
        svg = f.read()

    # Remove XML declaration (we'll add our own if needed)
    svg = re.sub(r"<\?xml[^?]*\?>\s*", "", svg).strip()

    # Ensure version and baseProfile on root <svg>
    if "baseProfile=" not in svg:
        svg = re.sub(r"<svg\b", '<svg baseProfile="tiny-ps"', svg, count=1)
    if 'version="1.2"' not in svg:
        svg = re.sub(r"<svg\b", '<svg version="1.2"', svg, count=1)

    # Remove x= and y= from root <svg>
    svg = re.sub(r'(<svg\b[^>]*?)\s+x="[^"]*"', r"\1", svg)
    svg = re.sub(r'(<svg\b[^>]*?)\s+y="[^"]*"', r"\1", svg)

    # Make viewBox square
    vb_w = vb_h = 0.0
    vb_match = re.search(
        r'viewBox="([+-]?[\d.]+)\s+([+-]?[\d.]+)\s+([\d.]+)\s+([\d.]+)"', svg
    )
    if vb_match:
        vb_x, vb_y = float(vb_match.group(1)), float(vb_match.group(2))
        vb_w, vb_h = float(vb_match.group(3)), float(vb_match.group(4))
        if vb_w != vb_h:
            new_size = max(vb_w, vb_h)
            if vb_w < new_size:
                vb_x -= (new_size - vb_w) / 2
            if vb_h < new_size:
                vb_y -= (new_size - vb_h) / 2
            svg = svg.replace(
                vb_match.group(0),
                f'viewBox="{vb_x:.1f} {vb_y:.1f} {new_size:.1f} {new_size:.1f}"',
            )
            vb_w = vb_h = new_size

    # Add solid background <rect> if missing (BIMI requires opaque background)
    if vb_w > 0 and not _svg_has_background_rect(svg, vb_w, vb_h):
        bg_color = _infer_bg_color_from_svg(path)
        vb2 = re.search(
            r'viewBox="([+-]?[\d.]+)\s+([+-]?[\d.]+)\s+([\d.]+)\s+([\d.]+)"',
            svg,
        )
        if vb2:
            bg_rect = (
                f'<rect x="{vb2.group(1)}" y="{vb2.group(2)}" '
                f'width="{vb2.group(3)}" height="{vb2.group(4)}" '
                f'fill="{bg_color}"/>'
            )
            svg = re.sub(
                r"(<svg\b[^>]*>)",
                rf"\1\n  {bg_rect}",
                svg,
                count=1,
            )

    # Add <title> if missing
    if not re.search(r"<title\b", svg):
        svg = svg.replace("</svg>", f"  <title>{company_name}</title>\n</svg>")

    # Add <desc> if missing
    if not re.search(r"<desc\b", svg):
        svg = svg.replace(
            "</svg>",
            f"  <desc>{company_name} logo mark</desc>\n</svg>",
        )

    # Remove forbidden elements
    forbidden = [
        "script",
        "animate",
        "animateTransform",
        "animateMotion",
        "filter",
        "mask",
        "image",
        "foreignObject",
        "pattern",
        "symbol",
        "marker",
        "set",
    ]
    for tag in forbidden:
        svg = re.sub(rf"<{tag}\b[^>]*>.*?</{tag}>", "", svg, flags=re.DOTALL)
        svg = re.sub(rf"<{tag}\b[^>]*/>", "", svg)

    # Remove external href references
    svg = re.sub(r'href="https?://[^"]*"', "", svg)
    svg = re.sub(r'xlink:href="https?://[^"]*"', "", svg)

    # Add XML declaration
    svg = '<?xml version="1.0" encoding="UTF-8"?>\n' + svg

    return svg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_to_bimi(path: str, company_name: str) -> str:
    """Auto-detect format and convert to BIMI SVG."""
    fmt = _detect_format(path)
    if fmt == "svg":
        svg = svg_to_bimi_svg(path, company_name)
    elif fmt == "unknown":
        raise ValueError(f"Unsupported file format: {Path(path).suffix}")
    else:
        svg = raster_to_bimi_svg(path, company_name)

    size = len(svg.encode("utf-8"))
    if size > BIMI_MAX_SIZE:
        raise ValueError(
            f"BIMI SVG is {size:,} bytes, which exceeds the 32 KB limit "
            f"({BIMI_MAX_SIZE:,} bytes). Try a simpler logo or provide an "
            f"SVG source file."
        )
    return svg
