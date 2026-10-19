"""Local visual/audio exclusion adapter; no cloud account or upload service."""

import copy
import json
import subprocess
from pathlib import Path

from dear.protocol import validate_exclusions


def probe(path):
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            encoding="utf-8",
        )
    )


class VideoBlinder:
    def __init__(self, media):
        self.media = media

    def prepare(self, material, exclude):
        source = copy.deepcopy(material["receipt"])
        flags = {
            "visual_withheld": False,
            "audio_withheld": False,
            "prior_intervention": bool(material.get("transformed")),
        }
        if not exclude:
            return source, flags
        source_path = Path(source["video"])
        before_identity = (source_path.stat().st_size, source_path.stat().st_mtime_ns)
        info = probe(source["video"])
        streams = [s for s in info["streams"] if s["codec_type"] == "video"]
        if len(streams) != 1 or any(
            s["codec_type"] not in ("video", "audio") for s in info["streams"]
        ):
            raise ValueError("Expected one video stream with optional audio")
        stream = streams[0]
        rotations = [stream.get("tags", {}).get("rotate", 0)] + [
            s.get("rotation", 0) for s in stream.get("side_data_list", [])
        ]
        if any(float(r) % 360 for r in rotations):
            raise ValueError("Unrotated video required")
        tracks, audio = validate_exclusions(
            exclude, material["duration_s"], stream["width"], stream["height"]
        )
        directory = self.media.directory / f"blind_{self.media.sequence:04d}"
        self.media.sequence += 1
        directory.mkdir(parents=True, exist_ok=False)
        target = directory / "video.mp4"
        command = [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-n",
            "-i",
            source["video"],
            "-map",
            "0:v:0",
        ]
        if tracks:
            filters = []
            for track in tracks:
                x, y, right, bottom = track["box"]
                filters.append(
                    f"drawbox=x={x}:y={y}:w={right-x}:h={bottom-y}:color=black:t=fill:enable='between(t,{track['start_s']:.6f},{track['end_s']:.6f})'"
                )
            graph = directory / "filters.txt"
            graph.write_text(",".join(filters))
            command += [
                "-filter_script:v",
                str(graph),
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
            command += ["-c:v", "copy"]
        if audio:
            command += ["-an"]
        else:
            command += ["-map", "0:a?", "-c:a", "copy"]
        command += ["-map_metadata", "-1", "-movflags", "+faststart", str(target)]
        subprocess.run(command, check=True, capture_output=True)
        after = probe(target)
        tolerance = max(
            0.1, 2 / float(eval_fraction(stream.get("avg_frame_rate", "25/1")))
        )
        if (
            abs(float(after["format"]["duration"]) - float(info["format"]["duration"]))
            > tolerance
        ):
            raise ValueError("Blinding changed video duration")
        if audio and any(s["codec_type"] == "audio" for s in after["streams"]):
            raise ValueError("Audio removal failed")
        flags.update(visual_withheld=bool(tracks), audio_withheld=audio)
        target_identity = (target.stat().st_size, target.stat().st_mtime_ns)
        uri = self.media.client.publish(target)
        if (
            source_path.stat().st_size,
            source_path.stat().st_mtime_ns,
        ) != before_identity or (
            target.stat().st_size,
            target.stat().st_mtime_ns,
        ) != target_identity:
            raise ValueError("Media changed during blinding or delivery")
        return {"video": str(target), "video_url": uri}, flags


def eval_fraction(value):
    from fractions import Fraction

    rate = Fraction(value)
    return rate if rate > 0 else Fraction(25)
