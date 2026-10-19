"""Command-line entry point for DEAR."""

import argparse
import copy
import json
from pathlib import Path

from eval.client import load_client, neutral_media
from eval.run import read_specs

from .agent import DEARAgent
from .detector import investigate
from .media import probe
from .ocr import Settings, TrackedCatalogProvider
from .protocol import answer_prompt
from .reader import BlindAnswer


class Workspace:
    def __init__(self, qa, path, uri, duration):
        self.job = {**copy.deepcopy(qa), "video_url": uri}
        self.materials = {}
        self.original = self._register(
            "video",
            receipt={"video": str(path), "video_url": uri},
            parents=[],
            duration_s=duration,
            transformed=False,
        )

    def _register(self, kind, **values):
        ident = f"m{len(self.materials)}"
        self.materials[ident] = {"id": ident, "kind": kind, **values}
        return ident

    def material(self, ident):
        return copy.deepcopy(self.materials[ident])

    def messages(self, ident, role):
        material = self.materials[ident]
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": role},
                    {
                        "type": "text",
                        "text": answer_prompt(
                            self.job["question"], self.job["options"]
                        ),
                    },
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "material_id": material["id"],
                                "duration_s": material["duration_s"],
                            }
                        ),
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": material["receipt"]["video_url"]},
                    },
                ],
            }
        ]


class Media:
    def __init__(self, directory, client):
        self.directory = Path(directory)
        self.sequence = 0
        self.client = client


def run_one(case, video, directory, client, read_ocr):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    with neutral_media(video) as published:
        return run_published(case, video, client.publish(published), directory, client, read_ocr)


def run_published(case, video, uri, directory, client, read_ocr):
    duration = float(probe(video)["format"]["duration"])
    qa = {k: case[k] for k in ("question", "options")}
    workspace = Workspace(qa, video, uri, duration)

    # Logs omit video URLs/data, file locations and authentication metadata.
    def redact(value):
        if isinstance(value, dict):
            return {
                k: (
                    "[media]"
                    if k in ("video", "video_url", "url", "path")
                    else redact(v)
                )
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [redact(v) for v in value]
        return value

    with (directory / "events.jsonl").open("x", buffering=1, encoding="utf-8") as log:

        def record(event):
            log.write(json.dumps(redact(event), ensure_ascii=False) + "\n")

        terminal = BlindAnswer(
            workspace,
            ask=client,
            media=Media(directory / "media", client),
            record=record,
        )
        provider = TrackedCatalogProvider(read_ocr, settings=Settings(ocr_fps=2))
        try:
            report = investigate(workspace.job, client, material_id=workspace.original)
            record({"event": "detector_report", "report": report})
            result = DEARAgent(terminal, provider, report).run()
        except Exception as error:
            result = {
                "status": "technical_error",
                "response": None,
                "error_type": type(error).__name__,
                "error": str(error)[:500],
            }
        if provider.catalog is not None:
            (directory / "ocr_catalog.json").write_text(
                json.dumps(provider.catalog.value, ensure_ascii=False), encoding="utf-8"
            )
        (directory / "result.json").write_text(
            json.dumps(redact(result), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return result


def paddle_reader(device):
    from paddleocr import PaddleOCR

    engine = PaddleOCR(
        use_angle_cls=False, lang="ch", show_log=False, use_gpu=device == "gpu"
    )

    def read(rgb):
        found = engine.ocr(rgb[:, :, ::-1], cls=False)
        result = []
        for box, (text, confidence) in (found or [[]])[0] or []:
            if not text.strip():
                continue
            xs, ys = zip(*box)
            result.append(
                {
                    "text": text,
                    "confidence": float(confidence),
                    "box": [min(xs), min(ys), max(xs), max(ys)],
                }
            )
        return {"regions": result}

    return read


def main():
    p = argparse.ArgumentParser()
    for name in ("specs", "id", "video", "config", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--ocr-device", choices=["cpu", "gpu"], default="cpu")
    args = p.parse_args()
    client = load_client(args.config)
    try:
        run_one(
            read_specs(args.specs)[args.id],
            args.video,
            args.output,
            client,
            paddle_reader(args.ocr_device),
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
