"""BIMI SVG converter module.

Converts raster images (JPEG, PNG, WebP, GIF, BMP, TIFF) and existing SVGs
to BIMI-compliant SVG Tiny-PS format.

Uses PIL for color quantization and potrace for high-quality Bézier tracing
of each color layer.
"""

import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Source assessment & preprocessing
# ---------------------------------------------------------------------------


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


def _prepare_raster(path: str) -> Image.Image:
    """Open a raster image, select the best frame, and normalize mode.

    Returns an RGB or RGBA PIL Image ready for tracing.
    """
    img = Image.open(path)

    # Multi-frame formats: ICO has multiple sizes, GIF/WebP/TIFF can be
    # animated or multi-page.  Select the largest frame by pixel area.
    n_frames: int = getattr(img, "n_frames", 1)
    if n_frames > 1:
        best, best_size = img.copy(), img.size[0] * img.size[1]
        for i in range(n_frames):
            img.seek(i)
            px = img.size[0] * img.size[1]
            if px > best_size:
                best, best_size = img.copy(), px
        img = best

    # Convert palette-with-transparency to RGBA so the alpha channel
    # is available for _has_transparency.
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")

    # Rescue washed-out images where the entire dynamic range is
    # compressed into a narrow band (e.g. near-white LinkedIn exports).
    # Stretch the histogram so the faint content becomes traceable.
    rgb = img.convert("RGB")
    extrema = rgb.getextrema()  # ((r_min,r_max), (g_min,g_max), (b_min,b_max))
    extrema_pairs: tuple[tuple[int, int], ...] = extrema  # type: ignore[assignment]
    dyn_range = max(hi - lo for lo, hi in extrema_pairs)
    if dyn_range < 50:
        from PIL import ImageFilter, ImageOps

        img = ImageOps.autocontrast(rgb)
        img = img.filter(ImageFilter.MedianFilter(size=3))

    return img


# ---------------------------------------------------------------------------
# BIMI SVG assembly
# ---------------------------------------------------------------------------


def _color_dist(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Euclidean distance between two RGB triples."""
    return float(np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def _quantize_colors(
    img: Image.Image,
    bg_color: str,
    max_colors: int,
    fg_mask: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray]]:
    """Quantize an RGB image and return per-color binary masks.

    Returns a list of ``(hex_color, mask)`` pairs for each foreground
    color.  Similar colors are merged into a single group whose mask
    covers *all* pixels of every constituent color — including
    anti-aliasing edges — so potrace traces a complete, smooth outline.

    When *fg_mask* is provided (boolean array from the alpha channel),
    background detection uses it instead of color distance — a palette
    entry is background only if <10% of its pixels are opaque.  This
    preserves white content on colored regions (e.g. white text on a
    cyan ring) that color-distance filtering would discard.
    """
    bg_rgb = tuple(int(bg_color[i : i + 2], 16) for i in (1, 3, 5))

    rgb_img = img.convert("RGB")

    if fg_mask is not None:
        # Build the palette from foreground pixels only so the quantizer
        # allocates all color slots to actual content instead of wasting
        # them on white background variations.
        fg_pixels = np.array(rgb_img)[fg_mask]
        n_fg = len(fg_pixels)
        side = int(np.ceil(np.sqrt(n_fg)))
        padded = np.zeros((side * side, 3), dtype=np.uint8)
        padded[:n_fg] = fg_pixels
        palette_img = Image.fromarray(padded.reshape(side, side, 3), "RGB")
        palette_img = palette_img.quantize(
            colors=max_colors + 1, dither=Image.Dither.NONE
        )
        quantized = rgb_img.quantize(palette=palette_img, dither=Image.Dither.NONE)
    else:
        # +1 to account for the background color occupying a slot
        quantized = rgb_img.quantize(colors=max_colors + 1, dither=Image.Dither.NONE)

    palette = quantized.getpalette()
    assert palette is not None
    arr = np.array(quantized)
    n_palette = len(palette) // 3

    # Collect all non-background palette entries
    entries: list[tuple[tuple[int, ...], int, int]] = []  # (rgb, count, idx)
    for idx in range(n_palette):
        rgb = tuple(palette[idx * 3 : idx * 3 + 3])
        pixel_mask = arr == idx
        count = int(pixel_mask.sum())
        if count == 0:
            continue
        if fg_mask is not None:
            # Use alpha channel: background if <10% of pixels are opaque
            fg_count = int(pixel_mask[fg_mask].sum())
            if fg_count / count < 0.1:
                continue
            # Near-white entries that are only partially opaque are
            # anti-aliasing fringe — filter them to free color slots
            # for real content.  Fully opaque white (e.g. text) passes.
            if _color_dist(rgb, bg_rgb) < 80 and fg_count / count < 0.5:
                continue
        else:
            if _color_dist(rgb, bg_rgb) < 80:
                continue
        entries.append((rgb, count, idx))

    if not entries:
        return []

    # Group similar colors so that anti-aliasing intermediates merge
    # into the dominant color's mask.  Sorted by count so the largest
    # color anchors each group.  A threshold of 60 preserves visually
    # distinct colors (e.g. NBC peacock feathers) while still merging
    # near-identical shades from JPEG compression.
    merge_threshold = 60.0
    entries.sort(key=lambda e: -e[1])
    # Each group: (anchor_rgb, total_count, core_indices, all_indices)
    # core_indices drive the representative color; all_indices (which
    # includes absorbed artifact indices) drive the trace mask.
    groups: list[tuple[tuple[int, ...], int, list[int], list[int]]] = []
    for rgb, count, idx in entries:
        merged = False
        for gi, (g_rgb, g_count, g_core, g_all) in enumerate(groups):
            if _color_dist(rgb, g_rgb) < merge_threshold:
                g_core.append(idx)
                g_all.append(idx)
                groups[gi] = (g_rgb, g_count + count, g_core, g_all)
                merged = True
                break
        if not merged:
            groups.append((rgb, count, [idx], [idx]))

    # Absorb artifact groups into their nearest larger neighbor.
    # JPEG compression creates scattered fringe pixels at color
    # boundaries that quantize into separate palette entries.  These
    # produce noisy trace layers that smudge letterforms.  A group is
    # considered an artifact when its pixels are spatially scattered
    # (low density within its bounding box) AND it is small relative
    # to the dominant group.  Absorbed indices extend the trace mask
    # but do NOT affect the representative color.
    max_count = max(g[1] for g in groups)

    def _is_artifact(indices: list[int]) -> bool:
        mask = np.isin(arr, indices)
        count = int(mask.sum())
        if count < 10:
            return True
        ys, xs = np.where(mask)
        bbox_area = (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        density = count / bbox_area
        return density < 0.10 and count < max_count * 0.25

    changed = True
    while changed:
        changed = False
        for si in range(len(groups) - 1, -1, -1):
            s_rgb, s_count, s_core, s_all = groups[si]
            if not _is_artifact(s_core):
                continue
            best_gi, best_dist = -1, float("inf")
            for gi, (g_rgb, g_count, g_core, g_all) in enumerate(groups):
                if gi == si or _is_artifact(g_all):
                    continue
                d = _color_dist(s_rgb, g_rgb)
                if d < best_dist:
                    best_dist, best_gi = d, gi
            if best_gi >= 0:
                g_rgb, g_count, g_core, g_all = groups[best_gi]
                g_all.extend(s_all)
                groups[best_gi] = (g_rgb, g_count + s_count, g_core, g_all)
                groups.pop(si)
                changed = True

    # Build masks — all_indices (core + absorbed artifacts) form the
    # trace mask for smooth edges; core_indices alone determine the
    # representative color so absorbed fringe pixels don't dilute it.
    rgb_arr = np.array(img.convert("RGB"))
    total_pixels = arr.size
    layers: list[tuple[str, np.ndarray]] = []
    for _rgb, count, core_indices, all_indices in groups:
        # Skip noise groups covering < 0.1% of the image
        if count < max(10, total_pixels // 1000):
            continue
        mask = np.isin(arr, all_indices)
        # Sample the purest interior pixels for accurate brand colors.
        # 1. Erode the full trace mask to exclude anti-aliased edges.
        # 2. Filter to pixels near the median (exclude outliers from
        #    absorbed artifacts or JPEG noise).
        # 3. Take the mean of these pure interior pixels.
        from PIL import ImageFilter

        mask_img = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        interior = np.array(mask_img.filter(ImageFilter.MinFilter(3))) > 127
        color_pixels = rgb_arr[interior] if interior.any() else rgb_arr[mask]
        anchor = np.median(color_pixels.astype(float), axis=0)
        dists = np.sqrt(((color_pixels.astype(float) - anchor) ** 2).sum(axis=1))
        close = color_pixels[dists < 30]
        if len(close) >= 10:
            rep_rgb = close.mean(axis=0).astype(int)
        else:
            rep_rgb = np.median(color_pixels, axis=0).astype(int)
        hex_color = "#{:02x}{:02x}{:02x}".format(*rep_rgb)
        layers.append((hex_color, mask))

    return layers


def _trace_mask_potrace(mask: np.ndarray) -> tuple[list[str], str] | None:
    """Trace a binary mask using potrace.

    Returns ``(path_d_strings, g_transform)`` or ``None`` if potrace
    is unavailable or fails.
    """
    if not shutil.which("potrace"):
        return None

    # potrace traces black-on-white: foreground → black
    fg_as_black = (~mask).astype(np.uint8) * 255
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

    xform_match = re.search(r'<g[^>]+transform="([^"]+)"', svg_text)
    xform = xform_match.group(1) if xform_match else ""

    return paths, xform


def _trace_raster(
    img: Image.Image,
    bg_color: str,
    max_colors: int = 8,
    fg_mask: np.ndarray | None = None,
) -> str:
    """Quantize an image and trace each color layer with potrace.

    Returns the inner SVG markup (``<path>`` elements) for all
    foreground colors.  The coordinate space uses potrace's native
    PostScript coordinates wrapped in its own transform.
    """
    layers = _quantize_colors(img, bg_color, max_colors, fg_mask=fg_mask)
    if not layers:
        raise ValueError("Could not extract any foreground colors from the image.")

    if not shutil.which("potrace"):
        raise ValueError(
            "potrace is required for raster-to-BIMI conversion but was "
            "not found. Install it with: apt install potrace"
        )

    parts: list[str] = []
    for hex_color, mask in layers:
        result = _trace_mask_potrace(mask)
        if result is None:
            continue
        path_strs, xform = result
        combined_d = " ".join(path_strs)
        path_elem = f'<path d="{combined_d}" fill="{hex_color}"/>'
        if xform:
            parts.append(f'<g transform="{xform}">{path_elem}</g>')
        else:
            parts.append(path_elem)

    if not parts:
        raise ValueError("Could not trace any paths from the image.")

    return "\n    ".join(parts)


def raster_to_bimi_svg(path: str, company_name: str) -> str:
    """Convert a raster image to a BIMI-compliant multi-color SVG string."""
    img = _prepare_raster(path)

    # Transparent images always get a white background.  Save the alpha
    # channel as a foreground mask so _quantize_colors can distinguish
    # opaque white content (e.g. white text on a colored ring) from the
    # actual transparent background — color-distance filtering would
    # incorrectly discard both.
    fg_mask: np.ndarray | None = None
    if _has_transparency(img):
        bg_color = "#ffffff"
        fg_mask = np.array(img.split()[-1]) > 128
        bg_img = Image.new("RGB", img.size, (255, 255, 255))
        bg_img.paste(img, mask=img.split()[-1])
        img = bg_img
    else:
        bg_color = _dominant_color(img)

    img = img.convert("RGB")

    # Potrace output is in its own PostScript coordinate space; the
    # outer <g> centers the traced content with circle-crop clearance.
    # Bounds come from the image pixel dimensions (potrace maps 1:1).
    img_w, img_h = img.size
    canvas_size = 800
    av_scale = 0.85
    usable = canvas_size * av_scale
    scale_factor = usable / max(img_w, img_h)

    cx = img_w / 2
    cy = img_h / 2
    tx = canvas_size / 2 - cx * scale_factor
    ty = canvas_size / 2 - cy * scale_factor

    # Trace with progressively fewer colors until the SVG fits under 32 KB
    svg = ""
    for max_colors in (128, 64, 32, 16, 8, 4, 2):
        paths_svg = _trace_raster(img, bg_color, max_colors=max_colors, fg_mask=fg_mask)

        svg = (
            f'<svg version="1.2" baseProfile="tiny-ps"\n'
            f'     xmlns="http://www.w3.org/2000/svg"\n'
            f'     viewBox="0 0 {canvas_size} {canvas_size}">\n'
            f"  <title>{company_name}</title>\n"
            f"  <desc>{company_name} logo mark</desc>\n"
            f'  <rect width="{canvas_size}" height="{canvas_size}"'
            f' fill="{bg_color}"/>\n'
            f'  <g transform="translate({tx:.2f},{ty:.2f})'
            f' scale({scale_factor:.6f})">\n'
            f"    {paths_svg}\n"
            f"  </g>\n"
            f"</svg>"
        )

        if len(svg.encode("utf-8")) <= BIMI_MAX_SIZE:
            return svg

    return svg


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
    return {
        id_val: elem for elem in root.iter() if (id_val := elem.get("id")) is not None
    }


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
