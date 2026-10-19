"""Blind option mapping, followed by offline gold/foil joins and paired metrics."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .client import load_client
from .mapping_prompt import judge_prompt
from .run import CONDITIONS, digest, read_specs


def map_response(client, case, response):
    text = judge_prompt(case["question"], case["options"], response)
    message, meta = client(
        [{"role": "user", "content": text}], tools=[], kind="answer_mapping"
    )
    if meta.get("finish_reason") != "stop":
        raise ValueError("Incomplete mapper response")
    body = message["content"].strip()
    if body.startswith("```"):
        body = "\n".join(body.splitlines()[1:-1])
    value = json.loads(body)
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("selected_option"), str)
        or value["selected_option"] not in {*case["options"], "ambiguous"}
    ):
        raise ValueError("Mapper selected an option not present in this question")
    return {
        "selected_option": value["selected_option"],
        "raw_message": message,
        "metadata": meta,
    }


def group_rows(rows):
    models = defaultdict(dict)
    for row in rows:
        slot = models[row["model"]].setdefault(row["id"], {})
        slot[row["condition"]] = row
    return models


def complete(items):
    return all(c in items and items[c]["status"] == "mapped" for c in CONDITIONS)


def summaries(rows, cases):
    out = {}
    for model, by_id in group_rows(rows).items():
        triples = [(cases[i], v) for i, v in by_id.items() if complete(v)]

        def calculate(pairs):
            n = len(pairs)
            correct = (
                lambda case, rows, c: rows[c]["selected_option"]
                == case["correct_answer"]
            )
            accuracy = {
                c: sum(correct(k, v, c) for k, v in pairs) / n if n else None
                for c in CONDITIONS
            }
            initial = [(k, v) for k, v in pairs if correct(k, v, "no_claim")]
            trap = lambda c: (
                sum(v[c]["selected_option"] == k["trap_answer"] for k, v in pairs) / n
                if n
                else None
            )
            flips = {
                c: (
                    sum(v[c]["selected_option"] == k["trap_answer"] for k, v in initial)
                    / len(initial)
                    if initial
                    else None
                )
                for c in ("accurate_claim", "misleading_claim")
            }
            return {
                "complete_triples": n,
                "initially_correct": len(initial),
                "accuracy": accuracy,
                "trap_flip_rate": flips,
                "trap_answer_increase": (
                    trap("misleading_claim") - trap("no_claim") if n else None
                ),
            }

        outcomes = [r for v in by_id.values() for r in v.values()]
        statuses = Counter(r["status"] for r in outcomes)
        out[model] = {
            "overall": calculate(triples),
            "classes": {
                c: calculate([(k, v) for k, v in triples if k["class"] == c])
                for c in sorted({k["class"] for k, v in triples})
            },
            "mapping_errors": statuses["mapping_error"],
            "unavailable": statuses["unavailable"],
            "ambiguous": sum(r.get("selected_option") == "ambiguous" for r in outcomes),
        }
    return out


def bootstrap(rows, cases, replicates=5000, seed=1729):
    """Shared source-cluster draws, with equal model weights after pooling cases."""
    import numpy as np

    grouped = group_rows(rows)
    source_ids = sorted(
        {cases[i].get("source_id", i) for model in grouped.values() for i in model}
    )
    names = sorted(grouped)
    lookup = {s: i for i, s in enumerate(source_ids)}
    counts = np.zeros((len(names), len(source_ids), 6), dtype=float)
    for m, name in enumerate(names):
        for ident, items in grouped[name].items():
            if not complete(items):
                continue
            case = cases[ident]
            a, c, d = [items[k]["selected_option"] for k in CONDITIONS]
            initially = a == case["correct_answer"]
            source = lookup[case.get("source_id", ident)]
            counts[m, source] += [
                1,
                initially,
                c == case["correct_answer"],
                d == case["correct_answer"],
                initially and c == case["trap_answer"],
                initially and d == case["trap_answer"],
            ]

    def rates(totals):
        with np.errstate(invalid="ignore", divide="ignore"):
            values = np.column_stack(
                [
                    totals[:, 1] / totals[:, 0],
                    totals[:, 2] / totals[:, 0],
                    totals[:, 3] / totals[:, 0],
                    totals[:, 4] / totals[:, 1],
                    totals[:, 5] / totals[:, 1],
                ]
            )
        return np.nanmean(values, axis=0)

    labels = [
        "no_claim_accuracy",
        "accurate_claim_accuracy",
        "misleading_claim_accuracy",
        "accurate_claim_flip_rate",
        "misleading_claim_flip_rate",
    ]
    if not source_ids or not np.any(counts[:, :, 0]):
        return {}
    point = rates(counts.sum(axis=1))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        weights = np.bincount(
            rng.integers(0, len(source_ids), len(source_ids)), minlength=len(source_ids)
        )
        samples.append(rates(np.einsum("msk,s->mk", counts, weights)))
    samples = np.array(samples)
    return {
        "unit": "source_video",
        "model_weighting": "equal",
        "replicates": replicates,
        "seed": seed,
        "metrics": {
            name: {
                "mean": float(point[i]) if np.isfinite(point[i]) else None,
                "ci95": [
                    float(x) if np.isfinite(x) else None
                    for x in np.nanquantile(samples[:, i], [0.025, 0.975])
                ],
            }
            for i, name in enumerate(labels)
        },
    }


def main():
    p = argparse.ArgumentParser()
    for name in ("specs", "responses", "config", "output"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    cases = read_specs(args.specs)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "mapping_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    client = load_client(args.config)
    selected = {}
    try:
        for line in Path(args.responses).read_text().splitlines():
            row = json.loads(line)
            case = cases[row["id"]]
            if row["qa_sha256"] != digest(
                {k: case[k] for k in ("question", "options")}
            ):
                raise ValueError("Response QA does not match the release")
            key = (row["model"], row["id"], row["condition"])
            if key in selected and selected[key]["input_id"] != row["input_id"]:
                raise ValueError(
                    "Multiple input/configuration versions for one model/case/condition; score separate runs"
                )
            if key in selected and selected[key]["status"] == "mapped":
                continue
            result = {k: row[k] for k in ("id", "model", "condition", "input_id")}
            if row["status"] != "completed":
                result.update(status="unavailable")
                selected[key] = result
                continue
            binding = digest(
                {
                    "protocol": judge_prompt.__doc__ or "blind_mapping_v1",
                    "prompt": judge_prompt(
                        case["question"], case["options"], row["response"]
                    ),
                }
            )
            try:
                if binding not in cache:
                    cache[binding] = map_response(client, case, row["response"])
                    temp = out / "mapping_cache.tmp"
                    temp.write_text(json.dumps(cache, ensure_ascii=False))
                    temp.replace(cache_path)
                result.update(
                    status="mapped",
                    selected_option=cache[binding]["selected_option"],
                    mapping_id=binding,
                )
            except Exception as error:
                result.update(status="mapping_error", error_type=type(error).__name__)
            selected[key] = result
    finally:
        client.close()
    rows = list(selected.values())
    (out / "scores.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    (out / "metrics.json").write_text(json.dumps(summaries(rows, cases), indent=2))
    (out / "aggregate.json").write_text(json.dumps(bootstrap(rows, cases), indent=2))


if __name__ == "__main__":
    main()
