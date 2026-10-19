# VisualTrust

Code and benchmark specifications for **VisualTrust: Video Reasoning under
On-Screen Claim Framing**.

[Paper](https://arxiv.org/abs/2610.21159) | [PDF](https://arxiv.org/pdf/2610.21159)

## Contents

- `specs/`: the 5,063 benchmark cases, one JSON Lines file per source dataset.
- `specs/sources.json`: the source video of each case, as a path relative to its dataset.
- `specs/subtitles.json`: subtitle tracks shared by all three claim conditions of a case.
- `renderer/`: renders the no-claim, accurate-claim, and misleading-claim videos.
- `eval/`: inference, blind response-to-option mapping, and summary metrics.
- `dear/`: evidence-consistency checking, evidence selection, OCR tracking, and the independent reader.

Videos and font files are not included.

## Data

Obtain BOVText, Video-MME, Perception Test, VidOR, and MME-VideoOCR under their
respective terms. Place them under one dataset root in this layout, which the
paths in `specs/sources.json` follow:

```text
datasets/
  BOVText/{Train,Test}/Video/  and  BOVText/{Train,Test}/Annotation/
  VideoMME/data/
  Perception-Test/{train,valid}/videos/
  VidOR/videos/video/
  MME-VideoOCR/Video/
```

Each line of `specs/<dataset>.jsonl` is one case:

| Field | Content |
| --- | --- |
| `id` | Case ID, `<source_id>-<n>` |
| `source_id` | Source video, listed in `sources.json` |
| `question`, `options` | The question and its lettered options |
| `correct_answer` | The option supported by the video |
| `trap_answer` | The option the misleading claim supports |
| `claims` | The `accurate` and `misleading` claim texts |
| `class` | Claim-fact class code, defined in Appendix B of the paper |
| `canvas` | Frame size, frame rate, and duration of the source video |
| `base` | `source` keeps the video unchanged; `solid_fill` covers the annotated creator title |
| `rendering` | Timed claim-card drawing instructions for `accurate_claim` and `misleading_claim` |
| `shared_subtitles` | Present when the case uses a subtitle track from `subtitles.json` |

## Setup

Use Python 3.11+, FFmpeg 5.1+ with libx264, and ffprobe on `PATH`.

```sh
python -m pip install -r requirements.txt
```

Fonts are user-supplied. Copy `renderer/fonts.json` and set each entry to a font
file path relative to the copy. Rendering checks every required font before it
starts and stops if one is missing. A `.ttc` collection uses its first face.
A different font file can change glyph shapes and line fit.

## Render

Render every case in one dataset, including all three claim conditions:

```sh
python -m renderer.render --specs specs/Perception-Test.jsonl --dataset-root datasets \
  --fonts local_fonts.json --output rendered/Perception-Test
```

Replace `Perception-Test` with `BOVText`, `VideoMME`, `VidOR`, or `MME-VideoOCR`
to render another dataset. To render one case, add `--id`:

```sh
python -m renderer.render --specs specs/BOVText.jsonl \
  --id BOVText:Cls10_Program_video11-1 --dataset-root datasets \
  --fonts local_fonts.json --output rendered/example
```

Each case produces `no_claim.mp4`, `accurate_claim.mp4`, and `misleading_claim.mp4`.
A dataset-wide run puts each case in a directory named by its URL-encoded case ID.
`media.jsonl` at the output root lists every video by relative path.
The renderer refuses to overwrite existing outputs and never modifies source files.
`--preview` writes the claim cards as transparent PNGs and needs no source video.

A source video must match its spec's frame size and lie within 0.25 s of its
duration. Claim cards follow drawing instructions in which text, fonts, positions,
colors, shadows, and transforms are already resolved, so claim text is never
rewrapped. Shared subtitle cues are wrapped with the configured font, and a cue
that does not fit its region stops rendering.

For the 246 BOVText cases with a creator title, the paper's claim-blinded base
repairs the title region with the video inpainting model UnderEraser. The released
base fills that region with a solid color instead. Renders of these cases, and
renders made with substitute fonts, therefore differ from the evaluated videos.

## Evaluate and Score

Create model and judge configs from `eval/model_config.example.json`. Set the
endpoint and model, and keep the API key in the environment variable named by
`api_key_env`. Set the model's sampling and thinking settings explicitly in
`generation`. The runner imposes no output-token cap.

```sh
python -m eval.run --specs specs --media rendered/example/media.jsonl \
  --media-root rendered/example --config model.json --output responses.jsonl
python -m eval.score --specs specs --responses responses.jsonl \
  --config judge.json --output scores
```

The default client sends video bytes through Chat Completions. For native video
or file-upload APIs, set `adapter` to a locally installed `module:factory`. The
factory receives the config and returns a client with three parts:

- `publish(path)` returns the URL placed in the request. The file is named `video.mp4` and exists until the model call returns.
- `close()` releases resources.
- Calling the client with `messages`, `tools`, `kind`, and possibly further keyword arguments returns an assistant message and metadata with `finish_reason`, `model`, and `usage`.

Inference and scoring follow these rules:

- A rerun skips inputs already answered for the same question, video content, and model configuration. `timeout`, `retries`, and `api_key_env` do not count as configuration.
- `--retry-errors` retries inputs whose last attempt failed or ended incomplete. The runner exits with an error status if any input fails.
- The mapper receives only the question, options, and final response. It never receives the correct answer, the trap answer, or the claim condition.
- Identical mapping prompts are sent once per judge configuration and cached in `scores/mapping_cache.jsonl`.
- Each response maps to a listed option or `ambiguous`. Ambiguous responses count as incorrect.
- Summaries use complete three-condition cases, so a case with a failed or incomplete response is excluded.
- The trap-flip rate's denominator is the cases answered correctly with no claim.
- `aggregate.json` weights its listed models equally, using unrounded per-model values. Its 95% intervals resample source videos with 5,000 bootstrap draws shared across models.

## DEAR

Install PaddleOCR 2.x and the PaddlePaddle CPU or GPU runtime for your platform.
OCR runs on the CPU unless `--ocr-device gpu` is given.

```sh
python -m dear.run --specs specs --id BOVText:Cls10_Program_video11-1 \
  --video rendered/example/misleading_claim.mp4 --config model.json \
  --output dear_output
```

DEAR proceeds as follows:

- An independent subagent checks the video against the question, without the options or an earlier response, and reports conflicting accounts.
- If no conflict is reported, the reader receives the original video.
- Otherwise, the selector weighs the reported accounts against the video, question, and options. It locates on-screen text by querying OCR tracks.
- The selector can withhold text tracks, other visual regions or intervals, and the whole audio track. Withheld content is masked or removed with the video's duration unchanged.
- The reader answers in a fresh context from the resulting video and the original question and options. It receives none of the conflict report, OCR text, selection rationale, or earlier responses, and its response is the final answer.

The selector may query OCR several times before it selects. A query that repeats
without new results ends the run. PaddleOCR runs on a fixed grid of 2 frames per
second, and optical flow tracks text regions between OCR frames.

`dear_output/result.json` holds the final response and its execution status.
`events.jsonl` logs each step without video data or file paths.
For programmatic use, `dear.run.run_one` accepts an injected model client and OCR
callback, so a different OCR installation needs no change to the controller.

The DEAR modules are `run` (entry point), `agent` (control flow), `detector`
(consistency check), `reader` (independent reading), `selection` (OCR queries
and exclusions), `ocr` (fixed-grid detection and tracking), `media` (masking),
and `protocol` (shared prompts and tool validation).

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
