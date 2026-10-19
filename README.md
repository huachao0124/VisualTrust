# VisualTrust

Code and construction specifications for **VisualTrust: Video Reasoning under
On-Screen Claim Framing**.

[Paper](https://arxiv.org/abs/2610.21159) | [PDF](https://arxiv.org/pdf/2610.21159)

## Contents

- `specs/`: the 5,063 released questions, options, correct/trap answers, paired claims, classes, and layouts.
- `specs/sources.json`, `specs/subtitles.json`: dataset-relative source bindings and shared subtitle specifications.
- `renderer/`: solid-background preparation and timed text rendering.
- `eval/`: inference, blind response-to-option mapping, and paired summaries.
- `dear/`: conflict investigation, evidence selection, OCR tracking, and independent reading.

No videos, font binaries, model weights, account configurations, experiment logs,
or cluster-management code are included. Obtain the five source datasets under
their respective terms. Place their contents beneath the following directories:
`BOVText/`, `VideoMME/`, `Perception-Test/`, `VidOR/`, and `MME-VideoOCR/`.
`VideoMME/data/` refers to the original Video-MME video directory.

## Setup

Use Python 3.11+, FFmpeg with libx264, and ffprobe on `PATH`.

```sh
python -m pip install -r requirements.txt
```

Fonts are user-supplied. In a local copy of `renderer/fonts.json`, replace the
required null entries with font filenames relative to that JSON file. Font IDs
are stable references, not licenses to redistribute the files. No font is
silently substituted. A different font can change glyph shape and fit.

## Render

Render every case in one dataset, including all three claim conditions:

```sh
python -m renderer.render --specs specs/Perception-Test.jsonl --dataset-root datasets --fonts local_fonts.json --output rendered/Perception-Test
```

Replace `Perception-Test` with `BOVText`, `VideoMME`, `VidOR`, or `MME-VideoOCR`
to render another dataset. Each case has its own output directory, with a combined
`media.jsonl` at the dataset output root. Case IDs are URL-encoded in directory names.
Use a new output directory; existing videos are not overwritten.

To render just one case, add `--id`:

```sh
python -m renderer.render --specs specs/BOVText.jsonl \
  --id BOVText:Cls10_Program_video11 --dataset-root datasets \
  --fonts local_fonts.json --output rendered/example
```

Outputs are `no_claim.mp4`, `accurate_claim.mp4`, `misleading_claim.mp4`, and a
relative-path `media.jsonl` manifest. `--preview` produces transparent claim
layers without source video. Source files are never modified.

The public base uses solid fills for the selected original-title regions,
shared across conditions. Claim specs contain timed drawing instructions with
text, font identifiers, positions, colors, shadows, and transforms already
resolved. The renderer executes those instructions without selecting line breaks
or fitting text. No extraction service or old renderer is required.
Solid fills and local font choices mean these renders are not asserted to be
byte-identical to evaluated media.

## Evaluate And Score

Create model and judge configs from `eval/model_config.example.json`. Configure the
endpoint and model, and keep the actual key in the named environment variable.
Set the model's sampling and thinking settings explicitly in `generation`.
The runner does not impose an output-token cap.

```sh
python -m eval.run --specs specs --media rendered/example/media.jsonl \
  --media-root rendered/example --config model.json --output responses.jsonl
python -m eval.score --specs specs --responses responses.jsonl \
  --config judge.json --output scores
```

The default client uses Chat Completions with video bytes. For native video or
File APIs, set `adapter` to a locally installed `module:factory`. The returned
client implements `publish(path)`, `close()`, and a callable accepting messages,
tools, and request kind. It returns an assistant message plus `finish_reason`,
model and usage metadata. No internal provider gateway is needed.

Completed inference inputs are reused by QA, video-content and configuration
identity. `--retry-errors` retries unresolved inputs without replaying successes.
Scoring deduplicates identical mapping prompts and sends only question, options,
and final response to the mapper, never the correct/trap answers or condition.
Mapped outputs must be a listed option or `ambiguous`. Ambiguous is incorrect;
technical/incomplete responses remain unavailable. Summaries use complete
three-condition cases. Trap-flip denominators contain no-claim-correct cases.
Overall aggregates weight models equally using unrounded values. Confidence
intervals resample source-video clusters with 5,000 shared bootstrap draws.

## DEAR

Install PaddleOCR 2.x and the corresponding PaddlePaddle CPU or GPU runtime for
your platform. The default command uses CPU OCR; `--ocr-device gpu` opts in to
GPU. OCR runs on a fixed 2 FPS grid. Optical flow tracks regions between those
observations; tracking failures and scene cuts do not trigger extra OCR calls.

```sh
python -m dear.run --specs specs --id BOVText:Cls10_Program_video11 \
  --video rendered/example/misleading_claim.mp4 --config model.json \
  --output dear_output
```

The detector sees video and question, without options or an earlier response.
The selector sees the report, video and original QA. Text is selected by
queried OCR track ID; other visual regions/intervals and the whole audio track
may also be withheld. Both branches invoke one fresh reader: unchanged media
when no conflict is reported, otherwise the selected derived media. The reader
receives neither the report, OCR text, selection rationale nor previous replies.
There is no post-reader arbitration. The portable media adapter uses FFmpeg
encoding while retaining duration; server upload and deployment code is excluded.

For programmatic use, `dear.run.run_one` accepts an injected model client and
OCR callback. This supports a different OCR installation without changing the
controller. Each call returns status separately from response correctness.

The DEAR modules are `run` (entry point), `agent` (control flow), `detector`
(consistency check), `reader` (isolated final response), `selection` (OCR queries
and exclusions), `ocr` (fixed-grid detection and tracking), `media` (masking),
and `protocol` (shared prompts and tool validation). `eval/client.py` provides
the configurable API interface; no service endpoint or credential is bundled.

## Citation

```bibtex
@misc{zhu2026visualtrustvideoreasoningonscreen,
      title={VisualTrust: Video Reasoning under On-Screen Claim Framing},
      author={Huachao Zhu and Junjielong Xu and Sicheng Li and Liang Yang and Mingtao Chen and Zhichao Hu and Li Jiang and Pinjia He},
      year={2026},
      eprint={2610.21159},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.21159},
}
```
