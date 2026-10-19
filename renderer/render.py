"""Render exported claim layouts using locally supplied dataset videos."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from PIL import Image

try:
    from . import drawing, subtitles
except ImportError:
    import drawing
    import subtitles

DATASETS = {"BOVText", "VideoMME", "Perception-Test", "VidOR", "MME-VideoOCR"}


def dataset_path(root, relative):
    value = PurePosixPath(relative)
    if (
        value.is_absolute()
        or ".." in value.parts
        or "\\" in relative
        or not value.parts
        or value.parts[0] not in DATASETS
    ):
        raise ValueError("Source paths must be relative to one of the five datasets")
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source symlink escapes the dataset root")
    return path


def cards(spec, condition):
    layout = spec["rendering"]
    if layout["kind"] != "drawing_program":
        raise ValueError("Use the compiled release specifications")
    for episode in layout["episodes"]:
        image = drawing.paint(episode[condition])
        yield episode["start_s"], episode["end_s"], image


def solid_filters(spec, root):
    base = spec["base"]
    if base["mode"] == "source":
        return []
    annotation = json.loads(dataset_path(root, base["annotation"]).read_text())
    filters = []
    # Group identical title rectangles across consecutive annotated frames.
    runs = {}
    complete = []
    for frame, objects in sorted(annotation.items(), key=lambda x: int(x[0])):
        index = int(frame) - 1
        active = set()
        for obj in objects:
            if "target_ids" in base:
                if str(obj["ID"]) not in base["target_ids"]:
                    continue
            elif obj.get("category") != base["category"]:
                continue
            points = obj["points"]
            xs = points[::2]
            ys = points[1::2]
            box = (int(min(xs)), int(min(ys)), math_ceil(max(xs)), math_ceil(max(ys)))
            key = (str(obj["ID"]), box)
            active.add(key)
            if key in runs and runs[key][1] == index - 1:
                runs[key][1] = index
            else:
                if key in runs:
                    complete.append((key, runs.pop(key)))
                runs[key] = [index, index]
        for key in list(runs):
            if key not in active:
                complete.append((key, runs.pop(key)))
    complete.extend(runs.items())
    for (_, box), (start, end) in complete:
        x, y, x2, y2 = box
        filters.append(
            f"drawbox=x={x}:y={y}:w={x2-x}:h={y2-y}:color=0x{base['color']}:t=fill:enable='between(n,{start},{end})'"
        )
    return filters


def math_ceil(x):
    import math

    return math.ceil(x)


def subtitle_cards(spec, subtitle_specs):
    if not spec.get("shared_subtitles"):
        return
    track = subtitle_specs[spec["source_id"]]
    cues = {c["cue_id"]: c for c in track["cues"]}
    canvas = [spec["canvas"]["width"], spec["canvas"]["height"]]
    for layer in track["layers"]:
        look = (
            layer.get("subtitle_look_spec") or track["appearance"]["subtitle_look_spec"]
        )
        layout = (
            layer.get("subtitle_layout")
            or track["appearance"]["subtitle_track_layout"]["tracks"][0]
        )
        painted = subtitles.render_cue_overlay(
            cues[layer["cue_id"]], look, layout, {}, canvas, fingerprint=False
        )
        image = Image.new("RGBA", tuple(canvas))
        image.alpha_composite(painted["image"], tuple(painted["position_px"]))
        yield layer["start_s"], layer["end_s"], image


def timeline(layers, canvas, duration, directory):
    """Sparse PNG timeline: one input stream regardless of subtitle count."""
    directory.mkdir()
    events = defaultdict(list)
    for i, (start, end, image) in enumerate(layers):
        if not 0 <= start < end <= duration + 0.001:
            raise ValueError("Overlay interval outside video")
        name = directory / f"layer_{i}.png"
        image.save(name)
        image.close()
        events[round(start, 6)].append((i, name))
        events[round(min(end, duration), 6)].append((i, None))
    if not events:
        return None
    events[0.0]
    events[round(duration, 6)]
    times = sorted(events)
    active = {}
    entries = ["ffconcat version 1.0"]
    for k, t in enumerate(times):
        for i, path in events[t]:
            if path is None:
                active.pop(i, None)
            else:
                active[i] = path
        composed = Image.new("RGBA", canvas)
        for _, path in sorted(active.items()):
            with Image.open(path) as image:
                composed.alpha_composite(image)
        name = f"event_{k}.png"
        composed.save(directory / name)
        composed.close()
        entries.extend([f"file '{name}'", "option framerate 1000000"])
        if k + 1 < len(times):
            entries.append(f"duration {times[k+1]-t:.6f}")
    manifest = directory / "timeline.ffconcat"
    manifest.write_text("\n".join(entries) + "\n")
    return manifest


def render(spec, source, dataset_root, output, ffmpeg, subtitle_specs=None):
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(source),
            ],
            text=True,
        )
    )
    videos = [s for s in probe["streams"] if s["codec_type"] == "video"]
    if len(videos) != 1 or (videos[0]["width"], videos[0]["height"]) != (
        spec["canvas"]["width"],
        spec["canvas"]["height"],
    ):
        raise ValueError("Source video dimensions differ from the rendering spec")
    if abs(float(probe["format"]["duration"]) - spec["canvas"]["duration_s"]) > 0.25:
        raise ValueError("Source video duration differs from the rendering spec")
    output.mkdir(parents=True, exist_ok=True)
    for name in ["no_claim", "accurate_claim", "misleading_claim"]:
        if (output / (name + ".mp4")).exists():
            raise FileExistsError("Output already exists")
    fills = solid_filters(spec, dataset_root)
    with tempfile.TemporaryDirectory() as temp:
        for condition, name in [
            ("A", "no_claim"),
            ("C", "accurate_claim"),
            ("D", "misleading_claim"),
        ]:

            def layers():
                yield from subtitle_cards(spec, subtitle_specs or {})
                if condition != "A":
                    yield from cards(spec, condition)

            overlay = timeline(
                layers(),
                (spec["canvas"]["width"], spec["canvas"]["height"]),
                spec["canvas"]["duration_s"],
                Path(temp) / condition,
            )
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-n",
                "-i",
                str(source),
            ]
            if overlay:
                command += ["-safe", "0", "-f", "concat", "-i", str(overlay)]
            graph = []
            label = "0:v"
            if fills:
                graph.append("[0:v]" + ",".join(fills) + "[base]")
                label = "base"
            if overlay:
                graph.append(f"[{label}][1:v]overlay=shortest=1[layer]")
                label = "layer"
            if graph:
                script = Path(temp) / "filter.txt"
                script.write_text(";".join(graph))
                command += [
                    "-filter_complex_threads",
                    "1",
                    "-filter_complex_script",
                    str(script),
                    "-map",
                    f"[{label}]",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-preset",
                    "medium",
                    "-fps_mode",
                    "passthrough",
                ]
            else:
                command += ["-map", "0:v:0", "-c:v", "copy"]
            command += [
                "-map",
                "0:a?",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(output / (name + ".mp4")),
            ]
            subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--id", help="Render one case; omit to render every case in --specs")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fonts", type=Path, default=Path(__file__).with_name("fonts.json")
    )
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    mapping = json.loads(args.fonts.read_text())
    drawing.FONTS = {
        key: args.fonts.parent / relative
        for key, relative in mapping.items()
        if relative
    }
    with args.specs.open() as handle:
        specs = [json.loads(line) for line in handle if line.strip()]
    if args.id is not None:
        specs = [spec for spec in specs if spec["id"] == args.id]
    if not specs:
        raise ValueError("Unknown case ID" if args.id is not None else "No cases in specs file")
    ids = [spec["id"] for spec in specs]
    if any(not isinstance(value, str) or value in {"", ".", ".."} for value in ids):
        raise ValueError("Invalid case ID")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case IDs")
    outputs = [args.output if args.id is not None else args.output / quote(value, safe="") for value in ids]
    if args.preview:
        for spec, output in zip(specs, outputs):
            output.mkdir(parents=True, exist_ok=True)
            for condition in ["C", "D"]:
                for i, (_, _, image) in enumerate(cards(spec, condition)):
                    image.save(output / f"{condition}_{i}.png")
                    image.close()
        return
    if args.dataset_root is None:
        parser.error("--dataset-root is required for video rendering")
    data_root = args.specs.resolve().parent
    sources = {s["id"]: s for s in json.loads((data_root / "sources.json").read_text())}
    paths = []
    for spec, output in zip(specs, outputs):
        source = sources[spec["source_id"]]
        path = dataset_path(args.dataset_root, source["path"])
        if not path.is_file():
            raise FileNotFoundError(source["path"])
        if any((output / (condition + ".mp4")).exists()
               for condition in ["no_claim", "accurate_claim", "misleading_claim"]):
            raise FileExistsError(f"Output already exists: {output}")
        paths.append(path)
    if (args.output / "media.jsonl").exists():
        raise FileExistsError("Output manifest already exists")
    shared = {
        s["source_id"]: s
        for s in json.loads((data_root / "subtitles.json").read_text())
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "media.jsonl").open("x") as manifest:
        for index, (spec, path, output) in enumerate(zip(specs, paths, outputs), 1):
            render(spec, path, args.dataset_root, output, args.ffmpeg, shared)
            records = [
                {"id": spec["id"], "condition": c,
                 "path": (output / (c + ".mp4")).relative_to(args.output).as_posix()}
                for c in ["no_claim", "accurate_claim", "misleading_claim"]
            ]
            manifest.write("".join(json.dumps(row) + "\n" for row in records))
            manifest.flush()
            print(f"[{index}/{len(specs)}] {spec['id']}", flush=True)


if __name__ == "__main__":
    main()
