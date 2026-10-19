"""Execute precompiled drawing instructions without choosing text layout."""

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONTS = {}


def resolve_font(font_id):
    path = FONTS.get(font_id)
    if path is None or not Path(path).is_file():
        raise ValueError(f"Missing font: {font_id}; supply its file in the font map")
    return Path(path)


def gradient(size, start_rgb, end_rgb, x1, x2):
    width, height = size
    t = np.clip(
        (np.arange(width, dtype=np.float32) - float(x1)) / max(1, int(x2) - int(x1)),
        0,
        1,
    )
    start, end = np.array(start_rgb, dtype=np.float32), np.array(
        end_rgb, dtype=np.float32
    )
    colors = (start[None, :] * (1 - t[:, None]) + end[None, :] * t[:, None]).astype(
        np.uint8
    )
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[:, :, :3], pixels[:, :, 3] = colors[None, :, :], 255
    return Image.fromarray(pixels, "RGBA")


def paint(program):
    buffers, fonts = {}, {}
    uses = Counter(i for op in program["ops"] for i in op.get("inputs", []))
    uses[program["output"]] += 1
    for op in program["ops"]:
        kind, target = op["op"], op["out"]
        inputs = [buffers[i] for i in op.get("inputs", [])]
        args, kwargs = op.get("args", []), op.get("kwargs", {})
        if kind == "new":
            buffers[target] = Image.new(
                args[0],
                tuple(args[1]),
                tuple(args[2]) if isinstance(args[2], list) else args[2],
            )
        elif kind == "gradient":
            buffers[target] = gradient(*args)
        elif kind == "composite":
            buffers[target] = Image.alpha_composite(*inputs)
        elif kind == "draw":
            kwargs = dict(kwargs)
            if "font" in kwargs:
                font_id, size = kwargs["font"]
                key = font_id, size
                if key not in fonts:
                    fonts[key] = ImageFont.truetype(str(resolve_font(font_id)), size)
                kwargs["font"] = fonts[key]
            for key in ("fill", "stroke_fill", "outline"):
                if isinstance(kwargs.get(key), list):
                    kwargs[key] = tuple(kwargs[key])
            method = op["method"]
            if method not in {"text", "rectangle", "rounded_rectangle"}:
                raise ValueError("Unsupported drawing operation")
            getattr(ImageDraw.Draw(inputs[0], op.get("mode")), method)(*args, **kwargs)
        elif kind == "blur":
            buffers[target] = inputs[0].filter(ImageFilter.GaussianBlur(args[0]))
        elif kind == "lut":
            buffers[target] = inputs[0].point(args[0])
        elif kind in {"copy", "resize", "crop", "transform", "getchannel"}:
            if kind in {"resize", "crop", "transform"}:
                args = [tuple(args[0]), *args[1:]]
            if isinstance(kwargs.get("fillcolor"), list):
                kwargs = {**kwargs, "fillcolor": tuple(kwargs["fillcolor"])}
            buffers[target] = getattr(inputs[0], kind)(*args, **kwargs)
        elif kind == "paste":
            inputs[0].paste(
                inputs[1], *args, mask=inputs[2] if len(inputs) == 3 else None
            )
        elif kind == "alpha_composite":
            inputs[0].alpha_composite(inputs[1], *args, **kwargs)
        elif kind == "putalpha":
            inputs[0].putalpha(inputs[1] if len(inputs) > 1 else args[0])
        else:
            raise ValueError("Unsupported image operation")
        for ident in op.get("inputs", []):
            uses[ident] -= 1
            if uses[ident] == 0 and ident in buffers:
                buffers.pop(ident).close()
    result = buffers.pop(program["output"])
    for image in buffers.values():
        image.close()
    return result
