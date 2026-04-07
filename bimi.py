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

    # Multi-frame formats: ICO has multiple sizes, GIF/WebP/TIFF can be
    # animated or multi-page.  Select the largest frame by pixel area.
    if hasattr(img, "n_frames") and img.n_frames > 1:
        best, best_size = img.copy(), img.size[0] * img.size[1]
        for i in range(img.n_frames):
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
        # Resize preserving alpha
        img_up = img.resize((upsampled_dim, upsampled_dim), Image.Resampling.LANCZOS)
        alpha = np.array(img_up.split()[-1])
        gt_mask = alpha > 127
        return gt_mask, gt_mask, upsampled_dim

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
# Multi-color quantization helpers
# ---------------------------------------------------------------------------


def _quantize_colors(
    img: Image.Image, max_colors: int = 6
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Quantize image to a small palette.

    Returns:
        label_map: 2D ndarray of color indices (H x W)
        palette:   list of RGB tuples for each index
    """
    rgb = img.convert("RGBA")
    # Composite onto white to remove transparency before quantizing
    bg = Image.new("RGBA", rgb.size, (255, 255, 255, 255))
    bg.paste(rgb, mask=rgb.split()[3])
    rgb = bg.convert("RGB")

    quantized = rgb.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)
    palette_flat = quantized.getpalette()
    label_map = np.array(quantized)

    # Extract only the palette entries actually used
    used_indices = np.unique(label_map)
    palette = []
    index_remap = {}
    for new_idx, old_idx in enumerate(used_indices):
        r = palette_flat[old_idx * 3]
        g = palette_flat[old_idx * 3 + 1]
        b = palette_flat[old_idx * 3 + 2]
        palette.append((r, g, b))
        index_remap[old_idx] = new_idx

    # Remap label_map to contiguous indices
    remapped = np.zeros_like(label_map)
    for old_idx, new_idx in index_remap.items():
        remapped[label_map == old_idx] = new_idx
    label_map = remapped

    return label_map, palette


def _identify_background_index(
    label_map: np.ndarray, palette: list[tuple[int, int, int]]
) -> int:
    """Return the palette index of the background color."""
    h, w = label_map.shape
    border = max(4, min(h, w) // 20)
    edges = np.concatenate(
        [
            label_map[:border].ravel(),
            label_map[-border:].ravel(),
            label_map[border:-border, :border].ravel(),
            label_map[border:-border, -border:].ravel(),
        ]
    )
    counts = np.bincount(edges, minlength=len(palette))
    return int(np.argmax(counts))


def _merge_similar_colors(
    layers: list[tuple[np.ndarray, tuple[int, int, int]]], threshold: float = 30.0
) -> list[tuple[np.ndarray, str]]:
    """Merge layers whose colors are within Euclidean RGB distance.

    Returns list of (mask, hex_color) with similar colors merged.
    """
    merged: list[tuple[np.ndarray, np.ndarray]] = [
        (mask, np.array(color, dtype=float)) for mask, color in layers
    ]

    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                dist = np.linalg.norm(merged[i][1] - merged[j][1])
                if dist < threshold:
                    # Weighted average color by pixel count
                    c_i = merged[i][0].sum()
                    c_j = merged[j][0].sum()
                    total = c_i + c_j
                    if total > 0:
                        avg = (merged[i][1] * c_i + merged[j][1] * c_j) / total
                    else:
                        avg = merged[i][1]
                    combined_mask = merged[i][0] | merged[j][0]
                    merged[i] = (combined_mask, avg)
                    merged.pop(j)
                    changed = True
                    break
            if changed:
                break

    result = []
    for mask, color in merged:
        rgb = color.astype(int)
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
        result.append((mask, hex_color))
    return result


def preprocess_raster_multicolor(
    path: str, max_colors: int = 6
) -> tuple[list[tuple[np.ndarray, str]], int, str]:
    """Preprocess a raster image for multi-color tracing.

    Returns:
        layers: list of (binary_mask, hex_color) for each foreground color
        upsampled_dim: side length of the upsampled canvas
        bg_hex: hex color of the identified background
    """
    img = Image.open(path)

    # Multi-frame: select largest frame
    if hasattr(img, "n_frames") and img.n_frames > 1:
        best, best_size = img.copy(), img.size[0] * img.size[1]
        for i in range(img.n_frames):
            img.seek(i)
            px = img.size[0] * img.size[1]
            if px > best_size:
                best, best_size = img.copy(), px
        img = best

    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")

    transparent = _has_transparency(img)
    # Multi-color quantization separates regions by color, not edge
    # detection, so heavy upsampling just bloats path data.  Target
    # ~800px which gives good detail for BIMI avatars.
    max_dim = max(img.width, img.height)
    target_dim = 1000
    if max_dim <= target_dim:
        if max_dim < 200:
            upsample = 3
        elif max_dim < 400:
            upsample = 2
        else:
            upsample = 1
        upsampled_dim = max_dim * upsample
    else:
        # Downsample large images to target
        upsample = 1
        upsampled_dim = target_dim

    if transparent:
        img_up = img.resize((upsampled_dim, upsampled_dim), Image.Resampling.LANCZOS)
        alpha = np.array(img_up.split()[-1])
        opaque_mask = alpha > 127

        if not opaque_mask.any():
            return [(opaque_mask, "#000000")], upsampled_dim, "#ffffff"

        # Composite onto white, then quantize opaque pixels into
        # multiple colors instead of averaging them into one.
        bg_rgba = Image.new("RGBA", img_up.size, (255, 255, 255, 255))
        bg_rgba.paste(img_up, mask=img_up.split()[3])
        img_rgb = bg_rgba.convert("RGB")

        label_map, palette = _quantize_colors(img_rgb, max_colors)

        # The composited background (white) is the true background
        bg_idx = _identify_background_index(label_map, palette)
        bg_color = palette[bg_idx]
        bg_hex = "#{:02x}{:02x}{:02x}".format(*bg_color)

        total_pixels = label_map.size
        bg_arr = np.array(bg_color, dtype=float)
        raw_layers: list[tuple[np.ndarray, tuple[int, int, int]]] = []
        for idx, color in enumerate(palette):
            if idx == bg_idx:
                continue
            if np.linalg.norm(np.array(color, dtype=float) - bg_arr) < 30:
                continue
            # AND with the alpha mask so transparent pixels are excluded
            color_mask = (label_map == idx) & opaque_mask
            if color_mask.sum() < total_pixels * 0.005:
                continue
            raw_layers.append((color_mask, color))

        if not raw_layers:
            fg = np.array(img_rgb)[opaque_mask].mean(axis=0).astype(int)
            hex_color = "#{:02x}{:02x}{:02x}".format(*fg)
            return [(opaque_mask, hex_color)], upsampled_dim, bg_hex

        layers = _merge_similar_colors(raw_layers)

        fmt = _detect_format(path)
        blur_radius = max(1, upsample // 4)
        smoothed_layers = []
        for layer_mask, hex_color in layers:
            gray = Image.fromarray(layer_mask.astype(np.uint8) * 255, mode="L")
            if fmt == "jpeg":
                gray = gray.filter(ImageFilter.MedianFilter(size=5))
                gray = gray.filter(ImageFilter.MedianFilter(size=3))
            gray = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            smoothed = np.array(gray) > 127
            if smoothed.any():
                smoothed_layers.append((smoothed, hex_color))

        if not smoothed_layers:
            smoothed_layers = layers

        return smoothed_layers, upsampled_dim, bg_hex

    img_up = img.resize((upsampled_dim, upsampled_dim), Image.Resampling.LANCZOS)

    label_map, palette = _quantize_colors(img_up, max_colors)
    bg_idx = _identify_background_index(label_map, palette)
    bg_color = palette[bg_idx]

    # Distinguish edge-connected background from interior regions of the
    # same color.  Only pixels connected to the image border are true
    # background; interior "holes" (e.g. a white circle center inside a
    # colored ring) are kept as foreground so they trace correctly.
    bg_mask = label_map == bg_idx
    bg_labels = measure.label(bg_mask, connectivity=1)
    # Find labels that touch any border row/column
    border_labels: set[int] = set()
    h, w = bg_labels.shape
    border_labels.update(bg_labels[0, :].tolist())
    border_labels.update(bg_labels[-1, :].tolist())
    border_labels.update(bg_labels[:, 0].tolist())
    border_labels.update(bg_labels[:, -1].tolist())
    border_labels.discard(0)  # 0 = non-background pixels

    # True background: only the border-touching components
    true_bg_mask = np.zeros_like(bg_mask)
    for bl in border_labels:
        true_bg_mask |= bg_labels == bl

    # Interior regions of the background color become a foreground layer
    interior_bg_mask = bg_mask & ~true_bg_mask

    total_pixels = label_map.size
    raw_layers: list[tuple[np.ndarray, tuple[int, int, int]]] = []

    if interior_bg_mask.sum() >= total_pixels * 0.005:
        raw_layers.append((interior_bg_mask, bg_color))

    bg_arr = np.array(bg_color, dtype=float)
    for idx, color in enumerate(palette):
        if idx == bg_idx:
            continue
        # Skip colors too similar to background (e.g. JPEG compression
        # artifacts that split one color into multiple buckets)
        if np.linalg.norm(np.array(color, dtype=float) - bg_arr) < 30:
            continue
        mask = label_map == idx
        # Skip tiny regions (< 0.5% of image)
        if mask.sum() < total_pixels * 0.005:
            continue
        raw_layers.append((mask, color))

    bg_hex = "#{:02x}{:02x}{:02x}".format(*bg_color)

    if not raw_layers:
        # Fallback: treat everything non-background as foreground
        mask = label_map != bg_idx
        fg_rgb = np.array(img_up.convert("RGB"))
        if mask.any():
            fg = fg_rgb[mask].mean(axis=0).astype(int)
            hex_color = "#{:02x}{:02x}{:02x}".format(*fg)
        else:
            hex_color = "#000000"
        return [(mask, hex_color)], upsampled_dim, bg_hex

    layers = _merge_similar_colors(raw_layers)

    # Apply blur to each mask for smoother tracing
    fmt = _detect_format(path)
    blur_radius = max(1, upsample // 4)
    smoothed_layers = []
    for mask, hex_color in layers:
        gray = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        if fmt == "jpeg":
            gray = gray.filter(ImageFilter.MedianFilter(size=5))
            gray = gray.filter(ImageFilter.MedianFilter(size=3))
        gray = gray.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        smoothed = np.array(gray) > 127
        if smoothed.any():
            smoothed_layers.append((smoothed, hex_color))

    if not smoothed_layers:
        # If smoothing eliminated everything, return unsmoothed
        smoothed_layers = layers

    return smoothed_layers, upsampled_dim, bg_hex


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
        # Use L for straight segments, C for curves (saves ~70% per
        # straight segment which is common in text and geometric shapes).
        cp0 = beziers[0]
        parts = [f"M{cp0[0][1]:.1f},{cp0[0][0]:.1f}"]
        for cp in beziers:
            start = cp[0]
            end = cp[3]
            seg_len = np.linalg.norm(end - start)
            if seg_len < 1e-6:
                continue
            # Check if control points lie on the start→end line
            seg_dir = (end - start) / seg_len
            diff1 = cp[1] - start
            diff2 = cp[2] - start
            d1 = abs(seg_dir[0] * diff1[1] - seg_dir[1] * diff1[0])
            d2 = abs(seg_dir[0] * diff2[1] - seg_dir[1] * diff2[0])
            if d1 < tolerance and d2 < tolerance:
                parts.append(f"L{cp[3][1]:.1f},{cp[3][0]:.1f}")
            else:
                parts.append(
                    f"C{cp[1][1]:.1f},{cp[1][0]:.1f} "
                    f"{cp[2][1]:.1f},{cp[2][0]:.1f} "
                    f"{cp[3][1]:.1f},{cp[3][0]:.1f}"
                )
        parts.append("Z")
        paths.append("".join(parts))

    return paths


def _path_bbox_area(path_d: str) -> float:
    """Approximate bounding box area of an SVG path string."""
    coords = re.findall(r"(-?[\d.]+),(-?[\d.]+)", path_d)
    if not coords:
        return 0.0
    xs = [float(c[0]) for c in coords]
    ys = [float(c[1]) for c in coords]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def _filter_small_paths(
    path_strs: list[str], min_area_frac: float, total_area: float
) -> list[str]:
    """Remove paths whose bounding box area is below a fraction of total."""
    threshold = total_area * min_area_frac
    return [p for p in path_strs if _path_bbox_area(p) >= threshold]


def _round_path_coords(path_d: str) -> str:
    """Round all coordinates in an SVG path string to integers."""
    return re.sub(r"-?\d+\.\d+", lambda m: str(round(float(m.group(0)))), path_d)


def _trace_multicolor_layers(
    layers: list[tuple[np.ndarray, str]],
    upsampled_dim: int,
    tolerance: float = 2.0,
    min_area_frac: float = 0.001,
    max_paths_per_layer: int = 80,
) -> list[tuple[list[str], str, str]]:
    """Trace each color layer independently.

    Small contours (bounding box < ``min_area_frac`` of canvas) are
    discarded, and each layer is capped at ``max_paths_per_layer`` paths
    (keeping the largest).  Coordinates are rounded to integers to
    reduce path data size.

    Returns list of (path_d_strings, hex_color, potrace_xform) sorted by
    mask area descending (largest layers render first / underneath).
    """
    results: list[tuple[list[str], str, str, int]] = []
    canvas_area = float(upsampled_dim) ** 2

    for mask, hex_color in layers:
        area = int(mask.sum())
        potrace_result = _trace_with_potrace(mask)
        if potrace_result is not None:
            path_strs, xform = potrace_result
        else:
            path_strs = trace_to_svg_paths(mask, upsampled_dim, tolerance)
            xform = ""

        if not path_strs:
            continue

        # Filter out tiny contour fragments (anti-aliasing artifacts)
        path_strs = _filter_small_paths(path_strs, min_area_frac, canvas_area)
        if not path_strs:
            continue

        # Cap number of paths per layer, keeping largest by bbox area
        if len(path_strs) > max_paths_per_layer:
            path_strs.sort(key=_path_bbox_area, reverse=True)
            path_strs = path_strs[:max_paths_per_layer]

        # Round coordinates to integers to reduce path data size
        path_strs = [_round_path_coords(p) for p in path_strs]

        results.append((path_strs, hex_color, xform, area))

    # Sort by area descending so largest regions render underneath
    results.sort(key=lambda r: r[3], reverse=True)
    return [(paths, color, xform) for paths, color, xform, _ in results]


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
    canvas_size = 800
    av_scale = 0.85
    usable = canvas_size * av_scale

    max_colors = 10
    tolerance = None  # computed per-attempt

    layers, upsampled_dim, bg_color = preprocess_raster_multicolor(path, max_colors)
    if tolerance is None:
        tolerance = max(2.0, upsampled_dim / 400)

    traced = _trace_multicolor_layers(layers, upsampled_dim, tolerance)

    if not traced:
        raise ValueError("Could not trace any paths from the image.")

    # Progressive simplification to stay under 32KB.
    # Priority: increase tolerance first (keeps all features with less
    # detail), then filter small contours, then reduce colors.
    min_area_frac = 0.0005
    max_paths = 200
    total_path_bytes = sum(len(" ".join(ps)) for ps, _, _ in traced)
    while total_path_bytes > 31000:
        if tolerance < 8:
            # First: increase trace tolerance
            tolerance *= 1.5
            traced = _trace_multicolor_layers(
                layers, upsampled_dim, tolerance, min_area_frac, max_paths
            )
        elif min_area_frac < 0.01:
            # Second: filter small contours
            min_area_frac *= 2
            max_paths = max(20, max_paths - 20)
            traced = _trace_multicolor_layers(
                layers, upsampled_dim, tolerance, min_area_frac, max_paths
            )
        elif tolerance < 40:
            # Third: increase tolerance further
            tolerance *= 1.5
            traced = _trace_multicolor_layers(
                layers, upsampled_dim, tolerance, min_area_frac, max_paths
            )
        elif max_colors > 3:
            # Last resort: reduce colors but keep at least 3 (2 foreground)
            max_colors -= 1
            layers, upsampled_dim, bg_color = preprocess_raster_multicolor(
                path, max_colors
            )
            tolerance = max(2.0, upsampled_dim / 400)
            min_area_frac = 0.0001
            max_paths = 200
            traced = _trace_multicolor_layers(
                layers, upsampled_dim, tolerance, min_area_frac, max_paths
            )
        else:
            break
        if not traced:
            raise ValueError("Could not trace any paths from the image.")
        total_path_bytes = sum(len(" ".join(ps)) for ps, _, _ in traced)

    # Compute unified bounding box across all layers
    all_path_strs = []
    has_potrace = False
    potrace_h, potrace_w = 0, 0
    for path_strs, _, xform in traced:
        all_path_strs.extend(path_strs)
        if xform:
            has_potrace = True
            # All layers share the same image dimensions
            potrace_h, potrace_w = layers[0][0].shape

    if has_potrace:
        min_x, min_y = 0.0, 0.0
        max_x, max_y = float(potrace_w), float(potrace_h)
    else:
        min_x, min_y, max_x, max_y = _compute_path_bounds(all_path_strs)

    content_w = max_x - min_x
    content_h = max_y - min_y
    scale_factor = usable / max(content_w, content_h)

    mark_cx = (min_x + max_x) / 2
    mark_cy = (min_y + max_y) / 2
    tx = canvas_size / 2 - mark_cx * scale_factor
    ty = canvas_size / 2 - mark_cy * scale_factor

    # Build path elements for each color layer
    path_elems = []
    for path_strs, hex_color, xform in traced:
        combined_d = " ".join(path_strs)
        elem = f'<path d="{combined_d}" fill="{hex_color}" fill-rule="evenodd"/>'
        if xform:
            elem = f'<g transform="{xform}">{elem}</g>'
        path_elems.append(elem)
    paths_svg = "\n    ".join(path_elems)

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
    return {elem.get("id"): elem for elem in root.iter() if elem.get("id")}


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
