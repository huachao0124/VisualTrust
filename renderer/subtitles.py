"""Render the shared subtitle specifications."""

import hashlib
import json
import math
import unicodedata

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import drawing

SUPERSAMPLE = 3
RenderError = ValueError
CONTEXTUAL_SHAPING_RANGES = (
    (1424, 2303),
    (2304, 3839),
    (4096, 4255),
    (6016, 6143),
    (43392, 43487),
    (43616, 43647),
    (64285, 65023),
    (65136, 65279),
)


def parse_color(value: str, opacity: float = 1.0) -> tuple[int, int, int, int]:
    raw = value.lstrip("#")
    if len(raw) not in (6, 8):
        raise RenderError(f"invalid_color:{value}")
    rgb = tuple((int(raw[index : index + 2], 16) for index in (0, 2, 4)))
    embedded = int(raw[6:8], 16) / 255 if len(raw) == 8 else 1.0
    alpha = max(0, min(255, round(255 * embedded * opacity)))
    return rgb + (alpha,)


def glyph_advance(font: ImageFont.FreeTypeFont, char: str) -> float:
    return float(font.getlength(char))


def requires_contextual_shaping(text: str) -> bool:
    return text_direction(text) == "rtl" or any(
        (
            lower <= ord(char) <= upper
            for char in text
            for lower, upper in CONTEXTUAL_SHAPING_RANGES
        )
    )


def text_direction(text: str) -> str | None:
    """Return the first strong bidi direction for a shaped line."""
    for char in text:
        bidi = unicodedata.bidirectional(char)
        if bidi in {"R", "AL"}:
            return "rtl"
        if bidi == "L":
            return "ltr"
    return None


def _validate_contextual_layout(
    font: ImageFont.FreeTypeFont, text: str, spacing: float
) -> str | None:
    if not requires_contextual_shaping(text):
        return None
    if abs(spacing) > 1e-09:
        raise RenderError("contextual_shaping_requires_zero_letter_spacing")
    if getattr(font, "layout_engine", None) != ImageFont.Layout.RAQM:
        raise RenderError("contextual_shaping_requires_raqm")
    return text_direction(text)


def text_width(font: ImageFont.FreeTypeFont, text: str, spacing: float) -> float:
    if not text:
        return 0.0
    direction = _validate_contextual_layout(font, text, spacing)
    if requires_contextual_shaping(text):
        kwargs = {"direction": direction} if direction else {}
        return float(font.getlength(text, **kwargs))
    return sum((glyph_advance(font, char) for char in text)) + spacing * max(
        0, len(text) - 1
    )


def text_clusters(text: str) -> list[str]:
    """Group combining marks and joiner sequences for character-level wrapping."""
    clusters: list[str] = []
    for char in text:
        codepoint = ord(char)
        joins_previous = bool(clusters) and (
            unicodedata.combining(char) != 0
            or 65024 <= codepoint <= 65039
            or 917760 <= codepoint <= 917999
            or (char == "\u200d")
            or clusters[-1].endswith("\u200d")
        )
        if joins_previous:
            clusters[-1] += char
        else:
            clusters.append(char)
    return clusters


def split_long_token(
    font: ImageFont.FreeTypeFont, token: str, width: float, spacing: float
) -> list[str]:
    """Wrap a no-space token without splitting a basic grapheme cluster."""
    if requires_contextual_shaping(token):
        raise RenderError(f"contextual_token_exceeds_bbox:{token}")
    clusters = text_clusters(token)
    if not clusters:
        return [""]
    parts: list[str] = []
    current = ""
    for cluster in clusters:
        candidate = current + cluster
        if current and text_width(font, candidate, spacing) > width:
            parts.append(current)
            current = cluster
        else:
            current = candidate
        if text_width(font, current, spacing) > width:
            raise RenderError(f"cluster_exceeds_bbox:{cluster}")
    if current:
        parts.append(current)
    return parts


def wrap_paragraph(
    font: ImageFont.FreeTypeFont, paragraph: str, width: float, spacing: float
) -> list[str]:
    words = paragraph.split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if text_width(font, candidate, spacing) <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
            current = ""
        if text_width(font, word, spacing) <= width:
            current = word
            continue
        fragments = split_long_token(font, word, width, spacing)
        lines.extend(fragments[:-1])
        current = fragments[-1]
    if current:
        lines.append(current)
    return lines


def wrap_text(
    font: ImageFont.FreeTypeFont,
    text: str,
    width: float,
    spacing: float,
    max_lines: int,
) -> list[str]:
    lines = []
    for paragraph in text.splitlines() or [text]:
        lines.extend(wrap_paragraph(font, paragraph, width, spacing))
    if len(lines) > max_lines:
        raise RenderError(f"text_exceeds_max_lines:{len(lines)}>{max_lines}:{text}")
    return lines


def draw_spaced_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    spacing: float,
    **kwargs,
) -> None:
    direction = _validate_contextual_layout(font, text, spacing)
    if requires_contextual_shaping(text):
        if direction:
            kwargs["direction"] = direction
        draw.text(xy, text, font=font, **kwargs)
        return
    x, y = xy
    for char in text:
        draw.text((x, y), char, font=font, **kwargs)
        x += glyph_advance(font, char) + spacing


ProductionRenderError = ValueError


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True).encode()


def _overlay_region(
    canvas: list[int], layout: dict, look: dict
) -> tuple[int, int, int, int]:
    box = [float(value) for value in layout["bbox_px"]]
    stroke = float(look["stroke"]["width_px"])
    shadow = look["shadow"]
    blur = float(shadow["blur_px"]) if float(shadow["opacity"]) > 0 else 0.0
    dx = float(shadow["offset_x_px"]) if float(shadow["opacity"]) > 0 else 0.0
    dy = float(shadow["offset_y_px"]) if float(shadow["opacity"]) > 0 else 0.0
    margin = math.ceil(stroke + abs(dx) + abs(dy) + 4 * blur + 8)
    return (
        max(0, math.floor(box[0] - margin)),
        max(0, math.floor(box[1] - margin)),
        min(int(canvas[0]), math.ceil(box[2] + margin)),
        min(int(canvas[1]), math.ceil(box[3] + margin)),
    )


def render_cue_overlay(
    cue: dict,
    look: dict,
    layout: dict,
    font_assets: dict,
    canvas: list[int],
    *,
    fingerprint: bool = True,
) -> dict:
    """Render a shared subtitle cue into a tightly cropped RGBA image."""
    scale = int(SUPERSAMPLE)
    left, top, right, bottom = _overlay_region(canvas, layout, look)
    if left >= right or top >= bottom:
        raise ProductionRenderError(f"overlay_region_empty:{cue['cue_id']}")
    high = Image.new(
        "RGBA", ((right - left) * scale, (bottom - top) * scale), (0, 0, 0, 0)
    )
    box = [
        (float(layout["bbox_px"][0]) - left) * scale,
        (float(layout["bbox_px"][1]) - top) * scale,
        (float(layout["bbox_px"][2]) - left) * scale,
        (float(layout["bbox_px"][3]) - top) * scale,
    ]
    font_id = look["font_asset_id"]
    font_path = drawing.resolve_font(font_id)
    font = ImageFont.truetype(font_path, round(float(layout["font_size_px"]) * scale))
    spacing = float(look["letter_spacing_px"]) * scale
    line_spacing = float(look["line_spacing_px"]) * scale
    stroke_width = round(float(look["stroke"]["width_px"]) * scale)
    lines = wrap_text(
        font, cue["text"], box[2] - box[0], spacing, int(layout["max_lines"])
    )
    ascent, descent = font.getmetrics()
    line_height = ascent + descent
    total_height = line_height * len(lines) + line_spacing * max(0, len(lines) - 1)
    if total_height > box[3] - box[1]:
        raise ProductionRenderError(f"cue_height_overflow:{cue['cue_id']}")
    widths = [text_width(font, line, spacing) for line in lines]
    if any((width > box[2] - box[0] + 0.5 for width in widths)):
        raise ProductionRenderError(f"cue_width_overflow:{cue['cue_id']}")
    text_top = box[1] + (box[3] - box[1] - total_height) / 2
    positions = []
    for index, width in enumerate(widths):
        if layout["text_align"] == "left":
            x = box[0]
        elif layout["text_align"] == "right":
            x = box[2] - width
        else:
            x = box[0] + (box[2] - box[0] - width) / 2
        positions.append((x, text_top + index * (line_height + line_spacing)))
    actual = [
        min((position[0] for position in positions)),
        text_top,
        max((position[0] + width for position, width in zip(positions, widths))),
        text_top + total_height,
    ]
    draw = ImageDraw.Draw(high)
    backing = look["backing"]
    if backing["enabled"]:
        pad_x = float(backing["padding_x_px"]) * scale
        pad_y = float(backing["padding_y_px"]) * scale
        if backing["full_width"]:
            backing_box = [box[0], actual[1] - pad_y, box[2], actual[3] + pad_y]
        else:
            backing_box = [
                actual[0] - pad_x,
                actual[1] - pad_y,
                actual[2] + pad_x,
                actual[3] + pad_y,
            ]
        backing_box = [
            max(box[0], backing_box[0]),
            max(box[1], backing_box[1]),
            min(box[2], backing_box[2]),
            min(box[3], backing_box[3]),
        ]
        draw.rounded_rectangle(
            backing_box,
            radius=float(backing["radius_px"]) * scale,
            fill=parse_color(backing["color"], float(backing["opacity"])),
        )
    shadow = look["shadow"]
    if float(shadow["opacity"]) > 0:
        mask = Image.new("L", high.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        dx = float(shadow["offset_x_px"]) * scale
        dy = float(shadow["offset_y_px"]) * scale
        for line, position in zip(lines, positions):
            draw_spaced_text(
                mask_draw,
                (position[0] + dx, position[1] + dy),
                line,
                font,
                spacing,
                fill=255,
                stroke_width=stroke_width,
                stroke_fill=255,
            )
        blur = float(shadow["blur_px"]) * scale
        if blur > 0:
            mask = mask.filter(ImageFilter.GaussianBlur(blur))
        color = parse_color(shadow["color"], float(shadow["opacity"]))
        layer = Image.new("RGBA", high.size, color)
        layer.putalpha(Image.eval(mask, lambda alpha: round(alpha * color[3] / 255)))
        high.alpha_composite(layer)
    text_draw = ImageDraw.Draw(high)
    fill = parse_color(look["text_color"])
    stroke_fill = parse_color(look["stroke"]["color"])
    for line, position in zip(lines, positions):
        draw_spaced_text(
            text_draw,
            position,
            line,
            font,
            spacing,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill,
        )
    image = high.resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    painted = image.getchannel("A").getbbox()
    if painted is None:
        raise ProductionRenderError(f"cue_overlay_empty:{cue['cue_id']}")
    if (
        left > 0
        and painted[0] == 0
        or (top > 0 and painted[1] == 0)
        or (right < canvas[0] and painted[2] == image.width)
        or (bottom < canvas[1] and painted[3] == image.height)
    ):
        raise ProductionRenderError(f"cue_overlay_region_clipped:{cue['cue_id']}")
    cropped = image.crop(painted)
    absolute = [
        left + painted[0],
        top + painted[1],
        left + painted[2],
        top + painted[3],
    ]
    raw_hash = (
        hashlib.sha256(
            canonical_bytes({"size": list(cropped.size), "mode": cropped.mode})
            + cropped.tobytes()
        ).hexdigest()
        if fingerprint
        else None
    )
    return {
        "image": cropped,
        "position_px": [absolute[0], absolute[1]],
        "painted_bbox_px": absolute,
        "rgba_sha256": raw_hash,
        "lines": lines,
    }
