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
import xml.etree.ElementTree as ET
from copy import deepcopy
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
        ".ico": "ico",
        ".svg": "svg",
    }
    return fmt_map.get(ext, "unknown")


def _has_transparency(img: Image.Image) -> bool:
    if img.mode in ("RGBA", "LA", "PA"):
        alpha = np.array(img.split()[-1])
        return bool((alpha < 255).any())
    if img.mode == "P" and "transparency" in img.info:
        return True
    return False


def _dominant_color(img: Image.Image) -> str:
    """Infer the dominant background color from the image edges.

    For transparent images, only fully-opaque edge pixels are sampled.
    If fewer than 16 opaque edge pixels exist the background is assumed
    to be white (the standard BIMI background).
    """
    if _has_transparency(img):
        rgba = np.array(img.convert("RGBA"))
        alpha = rgba[:, :, 3]
        transparent_frac = (alpha <= 127).sum() / alpha.size
        # If a significant portion of the image is transparent, the
        # transparent area IS the background — use white (standard BIMI).
        if transparent_frac > 0.10:
            return "#ffffff"
        # Mostly-opaque image with minor transparency: sample opaque edges.
        h, w = rgba.shape[:2]
        border = max(4, min(h, w) // 20)
        edge_rgba = np.concatenate(
            [
                rgba[:border].reshape(-1, 4),
                rgba[-border:].reshape(-1, 4),
                rgba[border:-border, :border].reshape(-1, 4),
                rgba[border:-border, -border:].reshape(-1, 4),
            ]
        )
        opaque = edge_rgba[:, 3] > 127
        if opaque.sum() < 16:
            return "#ffffff"
        mean = edge_rgba[opaque][:, :3].mean(axis=0).astype(int)
        return "#{:02x}{:02x}{:02x}".format(*mean)

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


def _is_multicolor(img: Image.Image) -> bool:
    """Detect if an image has multiple visually distinct foreground colors.

    For transparent images, foreground is the opaque region.
    For opaque images, foreground is pixels that differ significantly
    from the dominant edge (background) color.

    Returns True when the foreground pixels span a wide enough color
    range that a single fill color would lose important detail.
    """
    rgb = None
    if _has_transparency(img):
        rgba = img.convert("RGBA")
        arr = np.array(rgba)
        alpha = arr[:, :, 3]
        opaque = alpha > 127
        transparent_frac = 1.0 - opaque.sum() / alpha.size
        if transparent_frac >= 0.05 and opaque.sum() >= 16:
            rgb = arr[:, :, :3][opaque]
    if rgb is None:
        # Opaque image: identify foreground by color distance from edges
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
        bg = edges.mean(axis=0)
        dist = np.sqrt(((arr.astype(float) - bg) ** 2).sum(axis=2))
        fg_mask = dist > 30
        if fg_mask.sum() < 16:
            return False
        rgb = arr[fg_mask]

    # Check color spread: std dev across each channel.
    # Anti-aliasing on single-color logos produces std < 10;
    # multi-color logos with distinct regions typically exceed 20.
    std = rgb.astype(float).std(axis=0)
    return bool(std.max() > 20)


def _make_bg_transparent(img: Image.Image) -> Image.Image:
    """Convert an opaque image to RGBA by making background pixels transparent.

    Background is identified as pixels close to the dominant edge color.
    """
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
    bg = edges.mean(axis=0)
    dist = np.sqrt(((arr.astype(float) - bg) ** 2).sum(axis=2))
    alpha = np.where(dist > 30, 255, 0).astype(np.uint8)
    rgba = np.dstack([arr, alpha])
    return Image.fromarray(rgba, "RGBA")


def _quantize_colors(
    img: Image.Image, n_colors: int = 8
) -> list[tuple[str, np.ndarray]]:
    """Quantize a transparent image into color regions.

    Returns a list of ``(hex_color, binary_mask)`` tuples sorted by area
    (largest first).  Each mask is a boolean ndarray at the upsampled
    resolution, True where that color region exists.
    """
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]
    opaque = alpha > 127

    if opaque.sum() < 4:
        return []

    # Set transparent pixels to a uniform color so they consume only
    # one quantization bin, leaving more bins for real foreground colors.
    rgb_arr = arr[:, :, :3].copy()
    rgb_arr[~opaque] = 0
    rgb_img = Image.fromarray(rgb_arr, "RGB")
    quantized = rgb_img.quantize(colors=n_colors, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()
    assert palette is not None
    q_arr = np.array(quantized)

    regions: list[tuple[str, np.ndarray]] = []
    for idx in range(n_colors):
        mask = (q_arr == idx) & opaque
        if mask.sum() < 4:
            continue
        # Use actual mean color of pixels in this bin rather than the
        # palette entry, which can diverge from the true centroid.
        actual = arr[:, :, :3][mask]
        r, g, b = actual.mean(axis=0).astype(int)
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        regions.append((hex_color, mask))

    # Merge regions whose palette colors are perceptually close.
    # Anti-aliasing and JPEG artifacts often split a single logo color
    # into several similar quantized bins — merging them produces
    # cleaner paths and fewer noisy slivers.
    merge_threshold = 50
    merged: list[tuple[tuple[int, int, int], np.ndarray]] = []
    for hex_color, mask in regions:
        rgb = (
            int(hex_color[1:3], 16),
            int(hex_color[3:5], 16),
            int(hex_color[5:7], 16),
        )
        found = False
        for i, (mrgb, mmask) in enumerate(merged):
            dist = sum((a - b) ** 2 for a, b in zip(rgb, mrgb)) ** 0.5
            if dist < merge_threshold:
                # Keep the color of the larger region, union the masks
                merged[i] = (mrgb, mmask | mask)
                found = True
                break
        if not found:
            merged.append((rgb, mask))

    regions = [(f"#{r:02x}{g:02x}{b:02x}", m) for (r, g, b), m in merged]

    # Sort largest region first
    regions.sort(key=lambda x: x[1].sum(), reverse=True)
    return regions


def preprocess_raster(path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Preprocess a raster image for tracing.

    Returns:
        binary:        boolean ndarray, True = foreground (mark) pixels
        gt_mask:       boolean ndarray, ground-truth Otsu mask (before blur)
        upsampled_dim: side length of the square upsampled canvas
    """
    img = Image.open(path)
    fmt = _detect_format(path)

    # Multi-frame formats: ICO has multiple sizes, GIF/WebP/TIFF can be
    # animated or multi-page.  Select the largest frame by pixel area.
    n_frames = getattr(img, "n_frames", 1)
    if n_frames > 1:
        best, best_size = img.copy(), img.size[0] * img.size[1]
        for i in range(n_frames):
            img.seek(i)
            px = img.size[0] * img.size[1]
            if px > best_size:
                best, best_size = img.copy(), px
        img = best

    # Convert palette-with-transparency to RGBA so the alpha channel
    # is available for _has_transparency and the tracing pipeline.
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")

    transparent = _has_transparency(img)
    upsample = _upsample_factor(img)

    # Make square canvas
    max_dim = max(img.width, img.height)
    upsampled_dim = max_dim * upsample

    if transparent:
        rgba = np.array(img.convert("RGBA"))
        transparent_frac = (rgba[:, :, 3] <= 127).sum() / rgba[:, :, 3].size
        if transparent_frac >= 0.05:
            # Genuinely transparent — use alpha as the mask
            img_up = img.resize(
                (upsampled_dim, upsampled_dim), Image.Resampling.LANCZOS
            )
            alpha = np.array(img_up.split()[-1])
            gt_mask = alpha > 127
            return gt_mask, gt_mask, upsampled_dim
        # Nearly opaque (< 5% transparent, e.g. just rounded corners) —
        # fall through to Otsu thresholding which preserves internal detail.

    img_up = img.resize((upsampled_dim, upsampled_dim), Image.Resampling.LANCZOS)
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


def _smooth_contour(contour: np.ndarray, sigma: float = 2.5) -> np.ndarray:
    """Gaussian-smooth contour coordinates to remove pixel staircase jitter.

    Applies 1-D Gaussian smoothing independently to the row and column
    coordinates.  The contour is temporarily "unrolled" (duplicated at
    the ends) so the smoothing wraps correctly for closed contours.
    """
    from scipy.ndimage import gaussian_filter1d

    n = len(contour)
    if n < 5:
        return contour

    # Wrap: prepend/append copies so the filter sees continuity
    wrap = min(n // 4, 20)
    extended = np.concatenate([contour[-wrap:], contour, contour[:wrap]])
    smoothed = np.column_stack(
        [
            gaussian_filter1d(extended[:, 0], sigma),
            gaussian_filter1d(extended[:, 1], sigma),
        ]
    )
    return smoothed[wrap : wrap + n]


def trace_to_svg_paths(
    binary: np.ndarray, upsampled_dim: int, tolerance: float = 2.0
) -> list[str]:
    """Trace binary mask to SVG path strings with adaptive line/curve selection.

    Sharp corners get straight ``L`` segments to preserve geometric
    edges; smooth stretches get cubic Bezier ``C`` curves.
    """
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

        # Smooth the raw pixel-staircase contour before simplification.
        # This removes jagged single-pixel jitter while preserving the
        # overall shape and true corners.
        contour = _smooth_contour(contour)

        # Douglas-Peucker to reduce thousands of points to key vertices.
        dp_tolerance = tolerance * 0.5
        simplified = _simplify_contour(contour, dp_tolerance)
        if len(simplified) < 3:
            continue

        paths.append(_build_adaptive_path(simplified, tolerance))

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


def _clean_color_mask(mask: np.ndarray, min_component_pixels: int) -> np.ndarray:
    """Remove noise from a quantized color mask.

    Drops connected components smaller than *min_component_pixels*.
    Does NOT apply morphological opening — that erodes thin geometric
    features like the triangular facets in low-poly artwork.
    """
    from scipy.ndimage import label

    labeled, n_features = label(mask)  # type: ignore[reportGeneralTypeIssues]
    cleaned = mask.copy()
    for i in range(1, n_features + 1):
        component = labeled == i
        if component.sum() < min_component_pixels:
            cleaned[component] = False
    return cleaned


def _corner_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Turning angle (radians) at *p1* between segments p0→p1 and p1→p2."""
    v1 = p1 - p0
    v2 = p2 - p1
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.arccos(cos_a))


def _build_adaptive_path(
    simplified: np.ndarray, tolerance: float, corner_threshold: float = 0.25
) -> str:
    """Build an SVG sub-path choosing lines vs curves per vertex.

    At each vertex, the turning angle is measured.  Sharp corners
    (angle > *corner_threshold* radians, ~14°) get straight ``L``
    segments.  Smooth stretches between corners are fitted with
    cubic Bezier curves.

    This produces crisp geometric edges where they exist AND smooth
    curves where the contour is rounded — within the same path.
    """
    n = len(simplified)
    # Classify each interior vertex as sharp or smooth
    is_corner = [False] * n
    for i in range(1, n - 1):
        angle = _corner_angle(simplified[i - 1], simplified[i], simplified[i + 1])
        is_corner[i] = angle > corner_threshold
    # First/last are always corners (path start/end)
    is_corner[0] = True
    is_corner[-1] = True

    # contour coords are (row, col) = (y, x); swap for SVG
    def fmt(pt: np.ndarray) -> tuple[float, float]:
        return (pt[1], pt[0])

    parts = ["M{:.1f},{:.1f}".format(*fmt(simplified[0]))]

    i = 1
    while i < n:
        if is_corner[i]:
            # Sharp corner — emit a straight line
            parts.append("L{:.1f},{:.1f}".format(*fmt(simplified[i])))
            i += 1
        else:
            # Collect the smooth run up to the next corner
            run_start = i - 1
            while i < n and not is_corner[i]:
                i += 1
            run_end = min(i, n - 1)
            run_pts = simplified[run_start : run_end + 1]
            if len(run_pts) < 2:
                parts.append("L{:.1f},{:.1f}".format(*fmt(simplified[run_end])))
                i = run_end + 1
                continue
            # Fit Bezier curves to this smooth segment
            tan_l = _estimate_left_tangent(run_pts)
            tan_r = _estimate_right_tangent(run_pts)
            beziers = _fit_cubic_beziers(run_pts, tolerance, tan_l, tan_r)
            for cp in beziers:
                parts.append(
                    "C{:.1f},{:.1f} {:.1f},{:.1f} {:.1f},{:.1f}".format(
                        cp[1][1],
                        cp[1][0],
                        cp[2][1],
                        cp[2][0],
                        cp[3][1],
                        cp[3][0],
                    )
                )
            # Advance past the corner that ended the run
            if i < n:
                parts.append("L{:.1f},{:.1f}".format(*fmt(simplified[i])))
                i += 1

    parts.append("Z")
    return "".join(parts)


def _trace_color_region(
    mask: np.ndarray,
    upsampled_dim: int,
    tolerance: float,
    min_contour_area: float,
) -> list[str]:
    """Trace a cleaned color mask to SVG paths, filtering small contours.

    Uses adaptive per-vertex rendering: sharp corners get straight
    line segments, smooth stretches get Bezier curves.  This handles
    logos that mix geometric edges and curves in a single shape.
    """
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=False)
    contours = measure.find_contours(padded.astype(float), 0.5)

    paths = []
    for contour in contours:
        if len(contour) < 4:
            continue

        contour = contour - 1.0

        # Skip tiny contours (noise from quantization boundaries)
        bbox_w = contour[:, 1].max() - contour[:, 1].min()
        bbox_h = contour[:, 0].max() - contour[:, 0].min()
        if bbox_w * bbox_h < min_contour_area:
            continue

        # Light smoothing to reduce pixel-staircase jitter.  Use a
        # lower sigma than the single-color path so geometric / low-poly
        # details (angular feathers, faceted shapes) stay sharp.
        contour = _smooth_contour(contour, sigma=1.0)

        simplified = _simplify_contour(contour, tolerance)
        if len(simplified) < 3:
            continue

        paths.append(_build_adaptive_path(simplified, tolerance))

    return paths


def _silhouette_element(
    mask: np.ndarray,
    upsampled_dim: int,
    fill: str,
    tolerance: float,
) -> tuple[str | None, tuple[float, float, float, float]]:
    """Build the base-layer SVG element for the opaque silhouette.

    If the silhouette is approximately circular, returns a ``<circle>``
    element for a perfectly smooth edge.  Otherwise falls back to
    Bezier-traced ``<path>``.

    Returns ``(svg_element_string, (min_x, min_y, max_x, max_y))``
    or ``(None, ...)`` on failure.
    """
    # Find bounding box of opaque pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None, (0, 0, 0, 0)

    rmin, rmax = int(np.argmax(rows)), int(mask.shape[0] - 1 - np.argmax(rows[::-1]))
    cmin, cmax = int(np.argmax(cols)), int(mask.shape[1] - 1 - np.argmax(cols[::-1]))

    bbox_h = rmax - rmin + 1
    bbox_w = cmax - cmin + 1
    cx = (cmin + cmax) / 2.0
    cy = (rmin + rmax) / 2.0
    r = (bbox_w + bbox_h) / 4.0  # average radius

    # Check circularity: bbox is roughly square AND the filled area
    # is close to π·r² (within 10%)
    aspect = min(bbox_w, bbox_h) / max(bbox_w, bbox_h) if max(bbox_w, bbox_h) > 0 else 0
    expected_area = np.pi * r * r
    actual_area = float(mask.sum())

    if aspect > 0.9 and 0.80 < actual_area / expected_area < 1.10:
        # Close enough to a circle — emit a perfect <circle>
        # SVG coords: x=col, y=row
        elem = f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}"/>'
        bounds = (cx - r, cy - r, cx + r, cy + r)
        return elem, bounds

    # Not circular — fall back to Bezier-traced path
    path_strs = trace_to_svg_paths(mask, upsampled_dim, tolerance)
    if not path_strs:
        return None, (0, 0, 0, 0)
    combined_d = " ".join(path_strs)
    elem = f'<path d="{combined_d}" fill="{fill}" fill-rule="evenodd"/>'
    bounds = _compute_path_bounds(path_strs)
    return elem, bounds


def _multicolor_raster_to_bimi_svg(path: str, company_name: str) -> str | None:
    """Convert a multi-color raster to BIMI SVG.

    Quantizes the image into distinct color regions, then traces each
    as a *cumulative* mask — the largest region (background) includes
    the full opaque area, the next-largest adds its area on top, and so
    on.  This layered approach eliminates white-gap artifacts between
    adjacent color regions.

    Handles both transparent and opaque images.  Opaque images have
    their background made transparent before quantization.
    """
    img = Image.open(path)

    # Detect the actual background color from the original image BEFORE
    # any transparency manipulation, so it can be used in the output SVG.
    detected_bg = _dominant_color(img)

    # Opaque multicolor images need synthetic transparency so the
    # quantization and silhouette logic can identify the foreground.
    # Track whether the image lacks meaningful transparency — such
    # images produce hard alpha edges with anti-aliasing artifacts
    # that need erosion.  Genuinely transparent images have clean
    # alpha separation and should NOT be eroded (it destroys thin
    # geometric details like angular feathers or talons).
    if _has_transparency(img):
        rgba_arr = np.array(img.convert("RGBA"))
        tfrac = (rgba_arr[:, :, 3] <= 127).sum() / rgba_arr[:, :, 3].size
        needs_erosion = tfrac < 0.05  # nearly opaque, noisy boundaries
    else:
        needs_erosion = True
    if not _has_transparency(img):
        img = _make_bg_transparent(img)

    # Try progressively simpler settings until output fits 32 KB.
    for dim, n_colors in [(600, 10), (512, 8), (400, 6), (300, 4)]:
        upsample = _upsample_factor(img)
        upsampled_dim = max(min(max(img.width, img.height) * upsample, dim), 200)
        img_up = img.resize((upsampled_dim, upsampled_dim), Image.Resampling.LANCZOS)
        regions = _quantize_colors(img_up, n_colors=n_colors)
        if not regions:
            continue

        # Filter thresholds — tuned to remove anti-aliasing noise and
        # stray quantization fragments while keeping real facets.
        min_comp = max(8, int(upsampled_dim * upsampled_dim * 0.0005))
        min_contour = upsampled_dim * upsampled_dim * 0.001
        tolerance = max(2.0, upsampled_dim / 400)

        opaque = np.array(img_up.convert("RGBA"))[:, :, 3] > 127

        first_color, _ = regions[0]

        # If the largest region's color is close to the detected
        # background, treat it as background — use the detected bg
        # color and skip tracing an invisible silhouette.
        fc_rgb = tuple(int(first_color[i : i + 2], 16) for i in (1, 3, 5))
        bg_rgb = tuple(int(detected_bg[i : i + 2], 16) for i in (1, 3, 5))
        bg_dist = sum((a - b) ** 2 for a, b in zip(fc_rgb, bg_rgb)) ** 0.5
        if bg_dist < 30:
            first_color = detected_bg

        cleaned_opaque = _clean_color_mask(opaque, min_comp)

        # Pre-scan detail layers: detect circular regions so we can
        # subtract them from the silhouette (avoids a colored halo
        # peeking out from behind the perfect <circle>).
        detail_results: list[tuple[str, str | np.ndarray, tuple | None]] = []
        circle_masks: list[np.ndarray] = []
        total_opaque = opaque.sum()
        for hex_color, mask in regions[1:]:
            cleaned = _clean_color_mask(mask & opaque, min_comp)
            region_area = cleaned.sum()
            if region_area < min_comp:
                continue
            # Skip tiny regions (< 3% of foreground) — they are almost
            # always anti-aliasing slivers, not meaningful color facets.
            if region_area < total_opaque * 0.03:
                continue

            # Erode to strip anti-aliasing slivers — but only for
            # images that were originally opaque.  The opaque→transparent
            # conversion creates hard alpha edges that bleed intermediate
            # colors.  Genuinely transparent images have clean alpha
            # separation and erosion would destroy real thin details
            # (geometric feathers, angular talons, faceted shapes).
            if needs_erosion:
                from scipy.ndimage import binary_erosion

                eroded = binary_erosion(cleaned, iterations=2)
                eroded = _clean_color_mask(eroded, min_comp)
                if eroded.sum() >= region_area * 0.5:
                    cleaned = eroded
                elif eroded.sum() < min_comp:
                    continue

            # Check if this region is circular — emit <circle> for
            # perfect smoothness (dots, bullets, round icons).
            elem, bounds = _silhouette_element(
                cleaned, upsampled_dim, hex_color, tolerance
            )
            if elem is not None and elem.startswith("<circle"):
                detail_results.append((hex_color, elem, bounds))
                circle_masks.append(cleaned)
                continue

            detail_results.append((hex_color, cleaned, None))

        # Subtract circular detail regions from the silhouette so the
        # base layer doesn't extend behind perfect <circle> elements.
        # Dilate by 2px to also remove anti-aliasing fringe pixels.
        from scipy.ndimage import binary_dilation

        sil_mask = cleaned_opaque.copy()
        for cmask in circle_masks:
            dilated: np.ndarray = np.asarray(
                binary_dilation(cmask, iterations=1), dtype=bool
            )
            sil_mask = sil_mask & ~dilated

        silhouette_elem, sil_bounds = _silhouette_element(
            sil_mask, upsampled_dim, first_color, tolerance
        )
        if silhouette_elem is None:
            continue

        all_path_elems: list[str] = []
        all_min_x = all_min_y = float("inf")
        all_max_x = all_max_y = float("-inf")

        # Skip the silhouette when its color matches the background —
        # it's invisible and wastes bytes (e.g. white rounded rect on
        # white bg).  Still use its bounds for centering.
        if bg_dist >= 30:
            all_path_elems.append(silhouette_elem)
        rmin_x, rmin_y, rmax_x, rmax_y = sil_bounds
        all_min_x = min(all_min_x, rmin_x)
        all_min_y = min(all_min_y, rmin_y)
        all_max_x = max(all_max_x, rmax_x)
        all_max_y = max(all_max_y, rmax_y)

        # Emit detail layers
        for hex_color, data, bounds in detail_results:
            if isinstance(data, str):
                # Already an SVG element (circle)
                all_path_elems.append(data)
                assert bounds is not None
                rmin_x, rmin_y, rmax_x, rmax_y = bounds
                all_min_x = min(all_min_x, rmin_x)
                all_min_y = min(all_min_y, rmin_y)
                all_max_x = max(all_max_x, rmax_x)
                all_max_y = max(all_max_y, rmax_y)
                continue

            # data is a cleaned mask — trace it
            path_strs = _trace_color_region(data, upsampled_dim, tolerance, min_contour)
            t = tolerance
            while (
                sum(len(p) for p in path_strs) > BIMI_MAX_SIZE // len(regions)
                and t < 30
            ):
                t *= 1.5
                path_strs = _trace_color_region(data, upsampled_dim, t, min_contour)

            if not path_strs:
                continue

            combined_d = " ".join(path_strs)
            all_path_elems.append(
                f'<path d="{combined_d}" fill="{hex_color}" fill-rule="evenodd"/>'
            )

            rmin_x, rmin_y, rmax_x, rmax_y = _compute_path_bounds(path_strs)
            all_min_x = min(all_min_x, rmin_x)
            all_min_y = min(all_min_y, rmin_y)
            all_max_x = max(all_max_x, rmax_x)
            all_max_y = max(all_max_y, rmax_y)

        # Need at least one visible element.  When the silhouette color
        # differs from the background it counts as a visible element on
        # its own (e.g. a colored circle on a white background).
        if len(all_path_elems) < 1:
            continue

        content_w = all_max_x - all_min_x
        content_h = all_max_y - all_min_y

        canvas_size = 800
        usable = canvas_size * 0.85
        scale_factor = usable / max(content_w, content_h)

        mark_cx = (all_min_x + all_max_x) / 2
        mark_cy = (all_min_y + all_max_y) / 2
        tx = canvas_size / 2 - mark_cx * scale_factor
        ty = canvas_size / 2 - mark_cy * scale_factor

        paths_svg = "\n    ".join(all_path_elems)
        svg = BIMI_TEMPLATE.format(
            size=canvas_size,
            title=company_name,
            desc=f"{company_name} logo mark",
            bg_color=detected_bg,
            tx=f"{tx:.2f}",
            ty=f"{ty:.2f}",
            scale=f"{scale_factor:.6f}",
            paths=paths_svg,
        )

        if len(svg.encode("utf-8")) <= BIMI_MAX_SIZE:
            return svg

    # All iterations exhausted — return None so caller can fall back
    # to single-color tracing (e.g. when "multicolor" was just
    # anti-aliasing noise, not truly distinct color regions).
    return None


def raster_to_bimi_svg(path: str, company_name: str) -> str:
    """Convert a raster image to a BIMI-compliant SVG string."""
    img = Image.open(path)

    # Multi-color images get per-region color tracing.
    # Falls back to single-color below if multicolor produces no output
    # (e.g. when the "spread" was just anti-aliasing noise).
    if _is_multicolor(img):
        result = _multicolor_raster_to_bimi_svg(path, company_name)
        if result is not None:
            return result

    binary, gt_mask, upsampled_dim = preprocess_raster(path)

    bg_color = _dominant_color(img)

    # Determine mark color from foreground pixels
    arr_rgb = np.array(img.convert("RGB"))
    mask_resized = (
        np.array(
            Image.fromarray(binary.astype(np.uint8) * 255).resize(
                (img.width, img.height), Image.Resampling.NEAREST
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
        # Fallback: scikit-image contour tracing + adaptive paths.
        # Tight tolerance produces cleaner edges — affordable because
        # adaptive L segments are ~4x smaller than C Bezier commands.
        tolerance = max(1.0, upsampled_dim / 800)
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
# SVG-to-BIMI compliance conversion (ElementTree-based)
# ---------------------------------------------------------------------------

BIMI_MAX_SIZE = 32 * 1024  # 32 KB

_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"

# Register namespaces so ET serialization uses clean prefixes
ET.register_namespace("", _SVG_NS)
ET.register_namespace("xlink", _XLINK_NS)

# SVG properties that can appear as presentation attributes
_PRESENTATION_PROPS = {
    "alignment-baseline",
    "baseline-shift",
    "clip-rule",
    "color",
    "color-interpolation",
    "cursor",
    "direction",
    "display",
    "dominant-baseline",
    "fill",
    "fill-opacity",
    "fill-rule",
    "font-family",
    "font-size",
    "font-style",
    "font-variant",
    "font-weight",
    "image-rendering",
    "letter-spacing",
    "opacity",
    "overflow",
    "pointer-events",
    "shape-rendering",
    "stop-color",
    "stop-opacity",
    "stroke",
    "stroke-dasharray",
    "stroke-dashoffset",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-miterlimit",
    "stroke-opacity",
    "stroke-width",
    "text-anchor",
    "text-decoration",
    "text-rendering",
    "transform",
    "unicode-bidi",
    "vector-effect",
    "visibility",
    "word-spacing",
    "writing-mode",
}

# Elements forbidden in SVG Tiny-PS / BIMI
_FORBIDDEN_ELEMENTS = {
    "script",
    "style",
    "animate",
    "animateTransform",
    "animateMotion",
    "animateColor",
    "set",
    "filter",
    "feBlend",
    "feColorMatrix",
    "feComponentTransfer",
    "feComposite",
    "feConvolveMatrix",
    "feDiffuseLighting",
    "feDisplacementMap",
    "feFlood",
    "feGaussianBlur",
    "feImage",
    "feMerge",
    "feMergeNode",
    "feMorphology",
    "feOffset",
    "feSpecularLighting",
    "feTile",
    "feTurbulence",
    "mask",
    "clipPath",
    "pattern",
    "symbol",
    "marker",
    "image",
    "foreignObject",
    "switch",
    "a",
}

# Attributes to strip (not valid in SVG Tiny-PS)
_FORBIDDEN_ATTRS = {
    "class",
    "style",
    "clip-path",
    "mask",
    "filter",
    "marker-start",
    "marker-mid",
    "marker-end",
    "enable-background",
}


def _local_name(tag: str) -> str:
    """Strip namespace URI from an ElementTree tag."""
    return tag.split("}", 1)[1] if "}" in tag else tag


def _ns_tag(local: str) -> str:
    """Create a namespaced SVG tag for ElementTree."""
    return f"{{{_SVG_NS}}}{local}"


def _parse_css_rules(css_text: str) -> list[tuple[list[str], dict[str, str]]]:
    """Parse CSS into ``[(selector_list, {prop: val}), ...]``.

    Handles the subset of CSS found in SVGs from design tools:
    type, ``.class``, ``#id`` selectors and comma-separated groups.
    Skips ``@``-rules (``@import``, ``@font-face``, etc.).
    """
    css_text = re.sub(r"/\*.*?\*/", "", css_text, flags=re.DOTALL)
    # Remove @-rules that don't use braces (e.g. @import, @charset)
    css_text = re.sub(r"@(?:import|charset|namespace)\b[^;{}]*;", "", css_text)
    rules: list[tuple[list[str], dict[str, str]]] = []
    for m in re.finditer(r"([^{}]+)\{([^{}]+)\}", css_text):
        selectors = [
            s.strip()
            for s in m.group(1).split(",")
            if s.strip() and not s.strip().startswith("@")
        ]
        if not selectors:
            continue
        props: dict[str, str] = {}
        for decl in m.group(2).split(";"):
            decl = decl.strip()
            if ":" in decl:
                prop, val = decl.split(":", 1)
                prop, val = prop.strip(), val.strip()
                if prop and val:
                    props[prop] = val
        if props:
            rules.append((selectors, props))
    return rules


def _selector_matches(elem: ET.Element, selector: str) -> bool:
    """Test whether *elem* matches a simple CSS selector.

    Supports type (``circle``), class (``.cls-1``), id (``#logo``),
    universal (``*``), and combinations (``circle.cls-1``).
    """
    local = _local_name(elem.tag)
    classes = set(elem.get("class", "").split())
    elem_id = elem.get("id", "")

    remaining = selector
    tag_name = None
    m = re.match(r"^([a-zA-Z][\w-]*|\*)", remaining)
    if m:
        tag_name = m.group(1)
        remaining = remaining[m.end() :]

    required_classes: list[str] = []
    required_id = None
    while remaining:
        if remaining[0] == ".":
            cm = re.match(r"^\.([\w-]+)", remaining)
            if not cm:
                return False
            required_classes.append(cm.group(1))
            remaining = remaining[cm.end() :]
        elif remaining[0] == "#":
            im = re.match(r"^#([\w-]+)", remaining)
            if not im:
                return False
            required_id = im.group(1)
            remaining = remaining[im.end() :]
        else:
            return False

    if tag_name and tag_name != "*" and local != tag_name:
        return False
    if required_id is not None and elem_id != required_id:
        return False
    for rc in required_classes:
        if rc not in classes:
            return False
    return tag_name is not None or bool(required_classes) or required_id is not None


def _remove_element(root: ET.Element, target: ET.Element) -> None:
    """Remove *target* from its parent in *root*'s tree."""
    for parent in root.iter():
        for child in list(parent):
            if child is target:
                parent.remove(child)
                return


def _inline_styles(root: ET.Element) -> None:
    """Inline ``<style>`` CSS and ``style`` attributes as presentation attrs.

    1. Parses ``<style>`` blocks and applies matching rules (lower specificity).
    2. Parses ``style="..."`` and applies as individual attributes (higher
       specificity — overrides CSS rules).
    3. Removes ``<style>`` elements, ``style`` attributes, and ``class``
       attributes.
    """
    # Collect and parse all <style> blocks
    all_rules: list[tuple[list[str], dict[str, str]]] = []
    style_elems = list(root.iter(_ns_tag("style")))
    for se in style_elems:
        if se.text:
            all_rules.extend(_parse_css_rules(se.text))

    # Apply CSS rules — don't override existing presentation attributes
    for elem in root.iter():
        for selectors, props in all_rules:
            if any(_selector_matches(elem, sel) for sel in selectors):
                for prop, val in props.items():
                    if prop in _PRESENTATION_PROPS and elem.get(prop) is None:
                        elem.set(prop, val)

    # Inline style="..." (highest specificity — overrides everything)
    for elem in root.iter():
        style = elem.get("style")
        if style:
            for decl in style.split(";"):
                decl = decl.strip()
                if ":" in decl:
                    prop, val = decl.split(":", 1)
                    prop, val = prop.strip(), val.strip()
                    if prop and val and prop in _PRESENTATION_PROPS:
                        elem.set(prop, val)
            del elem.attrib["style"]

    # Remove <style> elements (now inlined)
    for se in style_elems:
        _remove_element(root, se)

    # Remove class attributes (no longer needed)
    for elem in root.iter():
        if "class" in elem.attrib:
            del elem.attrib["class"]


def _build_id_map(root: ET.Element) -> dict[str, ET.Element]:
    """Build a map of ``id`` -> element for the tree."""
    return {str(elem.get("id")): elem for elem in root.iter() if elem.get("id")}


def _resolve_use_refs(root: ET.Element) -> None:
    """Replace ``<use>`` elements with inlined copies of referenced content."""
    for _ in range(10):  # guard against circular references
        use_elems = list(root.iter(_ns_tag("use")))
        if not use_elems:
            break

        id_map = _build_id_map(root)
        resolved_any = False

        for use_elem in use_elems:
            href = use_elem.get("href") or use_elem.get(f"{{{_XLINK_NS}}}href")
            if not href or not href.startswith("#"):
                _remove_element(root, use_elem)
                resolved_any = True
                continue

            ref_elem = id_map.get(href[1:])
            if ref_elem is None:
                _remove_element(root, use_elem)
                resolved_any = True
                continue

            clone = deepcopy(ref_elem)
            if "id" in clone.attrib:
                del clone.attrib["id"]

            # Build transform from <use> x/y + existing transform
            try:
                ux = float(use_elem.get("x", 0))
                uy = float(use_elem.get("y", 0))
            except ValueError:
                ux, uy = 0.0, 0.0
            translate = f"translate({ux},{uy})" if ux or uy else ""
            use_xform = use_elem.get("transform", "")
            combined = f"{use_xform} {translate}".strip()

            # Copy presentation attributes from <use> to clone
            for attr, val in use_elem.attrib.items():
                if attr in _PRESENTATION_PROPS and attr != "transform":
                    clone.set(attr, val)

            if combined:
                wrapper = ET.Element(_ns_tag("g"))
                wrapper.set("transform", combined)
                wrapper.append(clone)
                replacement = wrapper
            else:
                replacement = clone

            # Swap <use> for its replacement in the tree
            for parent in root.iter():
                children = list(parent)
                for i, child in enumerate(children):
                    if child is use_elem:
                        parent.remove(use_elem)
                        parent.insert(i, replacement)
                        resolved_any = True
                        break

        if not resolved_any:
            break

    _clean_defs(root)


def _collect_url_refs(root: ET.Element) -> set[str]:
    """Collect all ids referenced via ``url(#...)`` in the tree."""
    refs: set[str] = set()
    url_pat = re.compile(r"url\(#([\w-]+)\)")
    for elem in root.iter():
        for val in elem.attrib.values():
            for m in url_pat.finditer(val):
                refs.add(m.group(1))
    return refs


def _clean_defs(root: ET.Element) -> None:
    """Remove unreferenced ``<defs>`` children and empty ``<defs>`` blocks."""
    refs = _collect_url_refs(root)
    for defs in list(root.iter(_ns_tag("defs"))):
        for child in list(defs):
            cid = child.get("id")
            if cid and cid not in refs:
                defs.remove(child)
        if len(defs) == 0 and not (defs.text and defs.text.strip()):
            _remove_element(root, defs)


def _strip_forbidden_elements(root: ET.Element) -> None:
    """Remove elements not allowed in SVG Tiny-PS."""
    for _ in range(5):
        removed = False
        for parent in list(root.iter()):
            for child in list(parent):
                if _local_name(child.tag) in _FORBIDDEN_ELEMENTS:
                    parent.remove(child)
                    removed = True
        if not removed:
            break


def _strip_foreign_namespaces(root: ET.Element) -> None:
    """Remove elements and attributes from non-SVG namespaces."""
    allowed_ns = {_SVG_NS, _XLINK_NS, ""}
    for parent in list(root.iter()):
        for child in list(parent):
            ns = child.tag.split("}")[0][1:] if "}" in child.tag else ""
            if ns and ns not in allowed_ns:
                parent.remove(child)
    for elem in root.iter():
        for attr in list(elem.attrib):
            if "}" in attr:
                attr_ns = attr.split("}")[0][1:]
                if attr_ns not in allowed_ns:
                    del elem.attrib[attr]


def _clean_for_tiny_ps(root: ET.Element) -> None:
    """Remove attributes not valid in SVG Tiny-PS."""
    for elem in root.iter():
        for attr in list(elem.attrib):
            local = _local_name(attr) if "}" in attr else attr
            if local in _FORBIDDEN_ATTRS:
                del elem.attrib[attr]
            elif local.startswith("data-"):
                del elem.attrib[attr]
            elif local.startswith("on"):
                del elem.attrib[attr]
        # Strip external href/xlink:href on any element
        for href_attr in ("href", f"{{{_XLINK_NS}}}href"):
            val = elem.get(href_attr, "")
            if val.startswith("http://") or val.startswith("https://"):
                del elem.attrib[href_attr]


def _has_text_elements(root: ET.Element) -> bool:
    """Check for ``<text>`` elements in the SVG."""
    return root.find(f".//{_ns_tag('text')}") is not None


def _infer_bg_color_from_svg(path: str) -> str:
    """Render the SVG at low res and sample edge pixels for background color."""
    try:
        import cairosvg
        from io import BytesIO

        png_data = cairosvg.svg2png(url=path, output_width=64, output_height=64)
        assert png_data is not None
        img = Image.open(BytesIO(png_data)).convert("RGB")
        return _dominant_color(img)
    except Exception:
        return "#ffffff"


def _ensure_viewbox(root: ET.Element) -> None:
    """Create ``viewBox`` from ``width``/``height`` if missing."""
    if root.get("viewBox"):
        return
    w = root.get("width", "").rstrip("pxtem")
    h = root.get("height", "").rstrip("pxtem")
    try:
        root.set("viewBox", f"0 0 {float(w)} {float(h)}")
    except ValueError:
        pass


def _ensure_square_viewbox(root: ET.Element) -> None:
    """Make ``viewBox`` square by expanding the shorter side."""
    vb = root.get("viewBox")
    if not vb:
        return
    parts = vb.split()
    if len(parts) != 4:
        return
    vb_x, vb_y, vb_w, vb_h = (float(p) for p in parts)
    if abs(vb_w - vb_h) < 0.01:
        return
    new_size = max(vb_w, vb_h)
    if vb_w < new_size:
        vb_x -= (new_size - vb_w) / 2
    if vb_h < new_size:
        vb_y -= (new_size - vb_h) / 2
    root.set(
        "viewBox",
        f"{vb_x:.1f} {vb_y:.1f} {new_size:.1f} {new_size:.1f}",
    )


def _get_viewbox(root: ET.Element) -> tuple[float, float, float, float] | None:
    """Parse ``viewBox`` into ``(x, y, w, h)`` or ``None``."""
    vb = root.get("viewBox")
    if not vb:
        return None
    parts = vb.split()
    if len(parts) != 4:
        return None
    return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))


def _has_background_rect(root: ET.Element) -> bool:
    """Check if SVG has a full-viewport background ``<rect>``."""
    dims = _get_viewbox(root)
    if not dims:
        return False
    _, _, vb_w, vb_h = dims
    for rect in root.iter(_ns_tag("rect")):
        try:
            rw = float(rect.get("width", "0"))
            rh = float(rect.get("height", "0"))
            if rw >= vb_w * 0.95 and rh >= vb_h * 0.95:
                return True
        except ValueError:
            continue
    return False


def _ensure_background_rect(root: ET.Element, path: str) -> None:
    """Add opaque background ``<rect>`` if missing."""
    dims = _get_viewbox(root)
    if not dims or _has_background_rect(root):
        return
    vb_x, vb_y, vb_w, vb_h = dims
    bg_color = _infer_bg_color_from_svg(path)
    rect = ET.Element(_ns_tag("rect"))
    rect.set("x", str(vb_x))
    rect.set("y", str(vb_y))
    rect.set("width", str(vb_w))
    rect.set("height", str(vb_h))
    rect.set("fill", bg_color)
    root.insert(0, rect)


def _ensure_title_desc(root: ET.Element, company_name: str) -> None:
    """Add ``<title>`` and ``<desc>`` if missing."""
    if root.find(f".//{_ns_tag('title')}") is None:
        title = ET.SubElement(root, _ns_tag("title"))
        title.text = company_name
    if root.find(f".//{_ns_tag('desc')}") is None:
        desc = ET.SubElement(root, _ns_tag("desc"))
        desc.text = f"{company_name} logo mark"


def _serialize_svg(root: ET.Element) -> str:
    """Serialize ElementTree root to a BIMI SVG string."""
    root.set("version", "1.2")
    root.set("baseProfile", "tiny-ps")
    for attr in ("x", "y", "width", "height"):
        if attr in root.attrib:
            del root.attrib[attr]
    svg_str = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + svg_str


def _svg_to_bimi_via_raster(path: str, company_name: str) -> str:
    """Render SVG to PNG via CairoSVG and convert through raster pipeline.

    Used as fallback when the SVG contains features that cannot be
    structurally converted (e.g. ``<text>`` elements).
    """
    import cairosvg

    png_data = cairosvg.svg2png(url=path, output_width=1024, output_height=1024)
    assert png_data is not None
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(png_data)
        tmp_png = f.name
    try:
        return raster_to_bimi_svg(tmp_png, company_name)
    finally:
        Path(tmp_png).unlink(missing_ok=True)


def svg_to_bimi_svg(path: str, company_name: str) -> str:
    """Convert an existing SVG to BIMI-compliant SVG Tiny-PS.

    Performs structural conversions to preserve the original appearance:

    - Inlines CSS ``<style>`` blocks and ``style`` attributes as
      presentation attributes
    - Resolves ``<use>`` references into inline content
    - Strips elements and attributes forbidden in SVG Tiny-PS
    - Removes non-SVG namespace elements and attributes
    - Ensures square viewBox, solid background, and required metadata

    Falls back to rasterize-and-retrace for SVGs containing ``<text>``
    elements or malformed XML.
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except ET.ParseError:
        return _svg_to_bimi_via_raster(path, company_name)

    # SVGs with <text> can't be structurally converted without font rendering
    if _has_text_elements(root):
        return _svg_to_bimi_via_raster(path, company_name)

    # Structural conversions (order matters)
    _inline_styles(root)
    _resolve_use_refs(root)
    _strip_forbidden_elements(root)
    _strip_foreign_namespaces(root)
    _clean_for_tiny_ps(root)

    # BIMI metadata requirements
    _ensure_viewbox(root)
    _ensure_square_viewbox(root)
    _ensure_background_rect(root, path)
    _ensure_title_desc(root, company_name)

    return _serialize_svg(root)


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
