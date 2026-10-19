# VisualTrust

Benchmark specifications and code for **VisualTrust: Video Reasoning under
On-Screen Claim Framing**.

[Paper](https://arxiv.org/abs/2610.21159) | [PDF](https://arxiv.org/pdf/2610.21159)

VisualTrust tests whether video MLLMs let an on-screen claim override what the
video shows. Each of its 5,063 cases pairs a multiple-choice question with three
versions of one video: no claim, an accurate claim, and a misleading claim. The
option the misleading claim supports is the trap answer. This repository renders
the three versions from the source datasets, evaluates models on them, and runs
DEAR, the paper's training-free mitigation.

| Path | Contents |
| --- | --- |
| `specs/` | Cases, one JSON Lines file per source dataset, with source videos in `sources.json` and shared subtitle tracks in `subtitles.json` |
| `renderer/` | Rendering of the three claim conditions |
| `eval/` | Inference, blind response-to-option mapping, and summary metrics |
| `dear/` | `detector` (consistency check), `agent` (control flow), `selection` and `ocr` (OCR queries and tracking), `media` (masking), `reader` (independent reading), `protocol` (shared prompts and validation), `run` (entry point) |

Videos and font files are not included.

## Setup

Install Python 3.11+, FFmpeg 5.1+ with libx264, and ffprobe, then:

```sh
python -m pip install -r requirements.txt
```

Obtain BOVText, Video-MME, Perception Test, VidOR, and MME-VideoOCR under their
respective terms, and arrange them under one dataset root:

```text
datasets/
  BOVText/{Train,Test}/{Video,Annotation}/
  VideoMME/data/
  Perception-Test/{train,valid}/videos/
  VidOR/videos/video/
  MME-VideoOCR/Video/
```

Fonts are user-supplied. Copy `renderer/fonts.json` and set each entry to a font
file path relative to the copy.

## Usage

### Render

Render every case in one dataset, or one case with `--id`:

```sh
python -m renderer.render --specs specs/Perception-Test.jsonl --dataset-root datasets \
  --fonts local_fonts.json --output rendered/Perception-Test
python -m renderer.render --specs specs/BOVText.jsonl --id BOVText:Cls10_Program_video11-1 \
  --dataset-root datasets --fonts local_fonts.json --output rendered/example
```

Each case yields `no_claim.mp4`, `accurate_claim.mp4`, and `misleading_claim.mp4`.
A dataset run puts each case in a directory named by its URL-encoded case ID, and
`--id` writes the three videos into `--output` itself. `media.jsonl` at the output
root lists every video by relative path. `--preview` writes the claim cards as
transparent PNGs and needs no source video.

Rendering checks every required font before it starts. Each source video must
match its spec's frame size and lie within 0.25 s of its duration. Existing outputs
are never overwritten, and source files are never modified.

Renders can differ from the evaluated videos in two ways. For the 246 BOVText
cases with a creator title, the released base covers the title with a solid color,
whereas the paper's base repaired it with the video inpainting model UnderEraser.
A substitute font changes glyph shapes, and because subtitle cues are wrapped with
the configured font, it can also make a cue overflow and stop rendering.

### Evaluate and Score

Create one config for the evaluated model and one for the judge from
`eval/model_config.example.json`. Each config names the endpoint and model. The
API key goes in the environment variable named by `api_key_env`. Sampling and
thinking settings go in `generation`, and no output-token cap is added.

```sh
python -m eval.run --specs specs --media rendered/example/media.jsonl \
  --media-root rendered/example --config model.json --output responses.jsonl
python -m eval.score --specs specs --responses responses.jsonl \
  --config judge.json --output scores
```

A rerun skips inputs already answered with the same question, video, and model
configuration. Changing only `timeout`, `retries`, or `api_key_env` keeps earlier
responses. `--retry-errors` also retries inputs that failed or ended incomplete,
and the runner exits with an error status if any input fails.

Scoring writes per-response mappings to `scores.jsonl`, per-model and per-class
metrics to `metrics.json`, and the cross-model summary to `aggregate.json`:

- The judge sees only the question, options, and final response, never the correct answer, the trap answer, or the claim condition.
- Each response maps to one listed option or `ambiguous`, and ambiguous responses count as incorrect.
- Metrics use complete three-condition cases.
- The trap-flip rate is computed over cases answered correctly with no claim.
- `aggregate.json` weights the listed models equally and gives 95% intervals from 5,000 bootstrap resamples of source videos.

### Run DEAR

Install PaddleOCR 2.x and the PaddlePaddle runtime for your platform.

```sh
python -m dear.run --specs specs --id BOVText:Cls10_Program_video11-1 \
  --video rendered/example/misleading_claim.mp4 --config model.json \
  --output dear_output
```

An independent subagent checks the video against the question, without the options,
and reports conflicting accounts. If it reports none, the reader answers from the
original video. Otherwise, a selector withholds the content it judges misleading.
It locates on-screen text through OCR tracks read at 2 frames per second. The
reader then answers from the resulting video in a fresh context. It never receives
the conflict report, OCR text, selection rationale, or earlier responses.

`dear_output/result.json` holds the final response and its execution status, and
`events.jsonl` logs each step without video data or file paths. OCR runs on the CPU
unless `--ocr-device gpu` is given. For programmatic use, `dear.run.run_one` accepts
an injected model client and OCR callback.

## Custom Model Clients

The default client sends video bytes through Chat Completions. For native video
or file-upload APIs, set `adapter` in the config to a locally installed
`module:factory`. The factory receives the config and returns a client with:

- `publish(path)`, which returns the URL placed in the request. The file is named `video.mp4` and exists until the model call returns.
- `close()`, which releases resources.
- A call taking `messages`, `tools`, `kind`, and possibly further keyword arguments, which returns an assistant message and metadata with `finish_reason`, `model`, and `usage`.

## Case Format

| Field | Content |
| --- | --- |
| `id` | Case ID, `<source_id>-<n>` |
| `source_id` | Source video, listed in `sources.json` |
| `question`, `options` | The question and its lettered options |
| `correct_answer` | The option the video supports |
| `trap_answer` | The option the misleading claim supports |
| `claims` | The `accurate` and `misleading` claim texts |
| `class` | Claim-fact class, defined in Appendix B of the paper |
| `canvas` | Frame size, frame rate, and duration of the source video |
| `base` | `source` keeps the video unchanged; `solid_fill` covers the annotated creator title |
| `rendering` | Timed claim-card drawing instructions for `accurate_claim` and `misleading_claim` |
| `shared_subtitles` | Present when the case uses a subtitle track from `subtitles.json` |

## License

The code is released under the MIT License. The specifications in `specs/` are
released under CC BY-NC 4.0 (`specs/LICENSE`). Content taken from the source
datasets remains subject to their terms.

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
