"""Evaluate rendered videos with resume-by-input identity, not by correctness."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

from .client import configuration_identity, create_client, neutral_media, read_config

CONDITIONS = ("no_claim", "accurate_claim", "misleading_claim")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def read_specs(directory):
    result = {}
    location = Path(directory)
    paths = [location] if location.is_file() else sorted(location.glob("*.jsonl"))
    if not paths:
        raise ValueError("No specification files found")
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["id"] in result:
                raise ValueError("Duplicate case ID")
            result[row["id"]] = row
    return result


def resolve_media(root, relative):
    value = PurePosixPath(relative)
    if value.is_absolute() or ".." in value.parts or "\\" in relative:
        raise ValueError("Media manifest paths must be relative")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Media path leaves its root")
    return path


def prompt(case):
    return (
        case["question"]
        + "\n\n"
        + "\n".join(f"{k}. {v}" for k, v in case["options"].items())
    )


def run(args):
    cases = read_specs(args.specs)
    media = [
        json.loads(line)
        for line in Path(args.media).read_text(encoding="utf-8").splitlines()
    ]
    seen = set()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["status"] == "completed" or not args.retry_errors:
                seen.add(row["input_id"])
    config = read_config(args.config)
    client = create_client(config)
    statuses = Counter()
    try:
        with output.open("a", buffering=1, encoding="utf-8") as log:
            for item in media:
                case = cases[item["id"]]
                condition = item["condition"]
                if condition not in CONDITIONS:
                    raise ValueError("Unknown condition")
                path = resolve_media(args.media_root, item["path"])
                h = hashlib.sha256()
                with path.open("rb") as f:
                    for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                        h.update(chunk)
                public = {k: case[k] for k in ("question", "options")}
                identity = digest(
                    {
                        "qa": public,
                        "media_sha256": h.hexdigest(),
                        "config": configuration_identity(config),
                    }
                )
                if identity in seen:
                    continue
                row = {
                    "id": case["id"],
                    "condition": condition,
                    "model": config["model"],
                    "input_id": identity,
                    "qa_sha256": digest(public),
                    "media_sha256": h.hexdigest(),
                    "generation": config.get("generation", {}),
                }
                try:
                    with neutral_media(path) as video:
                        message, meta = client(
                            [
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt(case)},
                                        {
                                            "type": "video_url",
                                            "video_url": {"url": client.publish(video)},
                                        },
                                    ],
                                }
                            ],
                            tools=[],
                            kind="benchmark",
                        )
                    text = message.get("content")
                    status = (
                        "completed"
                        if meta.get("finish_reason") == "stop"
                        and isinstance(text, str)
                        and text.strip()
                        and not message.get("tool_calls")
                        else "incomplete_response"
                    )
                    row.update(
                        status=status, response=text, raw_message=message, metadata=meta
                    )
                except Exception as error:
                    row.update(
                        status="technical_error",
                        error_type=type(error).__name__,
                        error=str(error)[:500],
                    )
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                seen.add(identity)
                statuses[row["status"]] += 1
    finally:
        client.close()
    print(dict(statuses), file=sys.stderr)
    return statuses


def main():
    p = argparse.ArgumentParser()
    for name in ("specs", "media", "media-root", "config", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--retry-errors", action="store_true")
    if run(p.parse_args())["technical_error"]:
        raise SystemExit("Some inputs failed; rerun with --retry-errors")


if __name__ == "__main__":
    main()
