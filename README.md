# speech-diarization-lab

[![ci](https://github.com/gradientsj/speech-diarization-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/gradientsj/speech-diarization-lab/actions/workflows/ci.yml)

Speaker-attributed transcription: who said what, with timestamps, from a
single audio file. Whisper (via CTranslate2) produces the words, a
diarization pipeline built from open parts produces the speakers, and a
tested alignment joins them. Everything that decides the output is scored:
WER and DER are implemented from scratch against hand-computed values, and
the diarizer is evaluated against the pretrained pyannote reference under
identical metrics on a reproducible benchmark.

**[Listen to the output](https://gradientsj.github.io/speech-diarization-lab/)**:
three benchmark mixtures with the speaker-attributed transcript synced to the
audio: per-speaker timeline, click-any-word-to-seek, live word highlighting.
The transcripts shown are real pipeline output, hallucinations and all (the
trailing *"Thank you for watching."* on mix_000 is Whisper inventing words
over silence, visible in one listen and invisible in a pooled WER).

## The problem

A transcript without speakers is close to unusable for meetings, interviews,
and calls; "who said what" is the actual product. Off-the-shelf pieces exist
for each stage (ASR, voice activity detection, speaker embeddings,
diarization pipelines), but the joins between them are where quality is won
or lost, and the joins are exactly what turnkey wrappers hide. This repo
builds the full path with every join visible and tested.

## The pipeline

```mermaid
flowchart LR
    AUDIO[/"audio file"/] --> ASR["faster-whisper (CTranslate2)<br/>words + timestamps"]
    AUDIO --> VAD["silero VAD<br/>speech regions"]
    VAD --> EMB["ECAPA-TDNN embeddings<br/>1.5s windows, 0.75s stride"]
    EMB --> CLU["agglomerative clustering<br/>cosine, average linkage"]
    CLU --> TURNS["speaker turns"]
    ASR --> ALIGN["alignment<br/>max-overlap word assignment"]
    TURNS --> ALIGN
    ALIGN --> OUT[/"speaker-attributed JSON / SRT / RTTM"/]
```

Two diarization backends sit behind one interface, the same shape as my
other lab repos: a thing built from parts compared against a pretrained
reference.

- **`clustered`** (built here): silero VAD, windowed ECAPA-TDNN speaker
  embeddings, agglomerative clustering over cosine distance, turn building.
  Every stage boundary is a pure function with unit tests; the model calls
  are thin, isolated wrappers.
- **`pyannote`** (reference): the pretrained
  `pyannote/speaker-diarization-3.1` pipeline. Gated on Hugging Face, so it
  needs `HF_TOKEN` with the model terms accepted.

## Quickstart

```bash
uv sync --extra models          # core install is light; model backends are an extra

# full pipeline: transcribe + diarize + align
uv run diarlab attribute meeting.wav --srt meeting.srt

# stages individually
uv run diarlab transcribe meeting.wav --model small --compute-type int8
uv run diarlab diarize meeting.wav --num-speakers 2

# the gated reference backend (after accepting the pyannote model terms)
uv sync --extra reference
uv run diarlab diarize meeting.wav --backend pyannote

# the HTTP serving layer: upload a file, poll the job, fetch JSON or SRT
uv sync --extra models --extra serve
uv run uvicorn diarlab.server:app --port 8000
curl -X POST localhost:8000/jobs -F file=@meeting.wav   # -> {"id": ...}
curl localhost:8000/jobs/<id>                           # -> status + segments
curl localhost:8000/jobs/<id>/srt                       # -> subtitles
```

The server runs jobs on one worker thread: the models load once and the GPU
is a serial resource. The measured real-time factors say that is enough for
interactive use (small/float16 transcribes at 0.022 RTF on an A10, roughly
45x real time); scaling past one box is a load balancer in front of more
instances, not threads. Model, device, and compute type come from
`DIARLAB_MODEL` / `DIARLAB_DEVICE` / `DIARLAB_COMPUTE`.

## Running the benchmark

The whole loop is three commands; the corpus is public and the mixtures are
seeded, so any machine reproduces the same benchmark bit for bit.

```bash
uv sync --extra models
uv run python -m diarlab.bench fetch      # LibriSpeech dev-clean, 337 MB
uv run python -m diarlab.bench build      # 12 seeded mixtures + manifest
uv run python -m diarlab.bench run --model small --compute-type int8 --device cpu
```

Each `run` writes `reports/benchmark_<backend>_<model>_<compute>_<device>.json`
with pooled WER, DER (split into miss / false alarm / confusion), per-mixture
rows, and ASR real-time factors.

A fourth subcommand calibrates the one real hyperparameter, the clustering
distance threshold. Embeddings are computed once per mixture and clustering
is rerun across the grid, so the whole sweep costs about one benchmark run:

```bash
uv run python -m diarlab.bench sweep --device cuda
```

It picks the threshold on half the mixtures, scores it on the other half,
and writes the full grid to `reports/threshold_sweep.json`.

On a CUDA box (tested on Lambda Stack / Ubuntu 22.04):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/gradientsj/speech-diarization-lab && cd speech-diarization-lab
uv sync --extra models
uv run python -m diarlab.bench fetch && uv run python -m diarlab.bench build
uv run python -m diarlab.bench run --model large-v3 --compute-type float16 --device cuda
```

If CTranslate2 cannot find cuDNN/cuBLAS (a `libcudnn` load error), point it
at the pip-installed NVIDIA libraries:

```bash
export LD_LIBRARY_PATH=$(uv run python -c "import os, nvidia.cublas.lib, nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ':' + os.path.dirname(nvidia.cudnn.lib.__file__))")
```

The gated reference backend, once `HF_TOKEN` is set and the pyannote model
terms are accepted:

```bash
uv sync --extra models --extra reference
uv run python -m diarlab.bench run --backend pyannote --device cuda
```

## How it is measured

- **WER** (word error rate) for transcription and **DER** (diarization error
  rate, md-eval semantics: missed speech + false alarm + speaker confusion
  over reference speech time, with a no-score collar and Hungarian speaker
  mapping) are implemented from scratch in plain Python and tested against
  hand-computed values, including the overlap and collar cases. The scoring
  math is the product here, so it should be auditable rather than imported.
- Where a metric is undefined (empty reference), it returns NaN, and NaN
  must be treated as a failure by anything gating on it.
- The benchmark is **synthetic conversations built from LibriSpeech**:
  single-speaker utterances interleaved with seeded gaps, so turn boundaries
  and transcripts are exact by construction and nothing requires credentials
  to download. The trade-off is stated plainly: no overlapped speech, no
  channel mismatch, read-speech acoustics. Scores on it are an upper bound
  on conversational performance and are for comparing systems under
  identical conditions, not for quoting as real-world accuracy.

## Results

Pooled over the 12 mixtures (10.9 minutes of audio, 2-4 speakers each),
DER collar 0.25 s; the clustered backend's distance threshold is stated per
row. GPU rows ran on an NVIDIA A10 (Lambda, Ubuntu 22.04), CPU rows on a
consumer Windows desktop. Per-mixture rows and exact configs are in
`reports/`.

### Transcription and serving

| model | compute | hardware | WER | mean RTF |
|---|---|---|---:|---:|
| tiny | float16 | A10 | 5.50% | 0.015 |
| base | float16 | A10 | 3.87% | 0.016 |
| small | float16 | A10 | **2.84%** | 0.022 |
| large-v3 | float16 | A10 | 4.23% | 0.087 |
| large-v3 | int8_float16 | A10 | 4.59% | 0.098 |
| large-v3 | int8 | A10 | 3.87% | 0.083 |
| tiny | int8 | CPU | 5.68% | 0.117 |
| base | int8 | CPU | 3.57% | 0.185 |
| small | int8 | CPU | 6.17% | 0.559 |

What the table says:

- **small beats large-v3 on this benchmark** (2.84% vs 3.87-4.59% at
  float16), at a quarter of the RTF. On clean read speech the largest model
  buys nothing, which is exactly why serving decisions should be made from
  a measured table rather than from model reputation.
- **int8 quantization is mostly accuracy-neutral, with one revealing
  exception.** On tiny, base, and large-v3 it matches float16 (it even
  edges it out on large-v3). On small, 11 of 12 mixtures match float16
  within noise, but on one mixture the int8 model silently drops whole
  utterances (59 of 157 reference words deleted, that mixture's WER 40.1%
  vs 3.8% at float16), inflating the pooled WER from 2.84% to 6.17%.
  Aggregate numbers hide exactly this kind of tail failure; the
  per-mixture rows in `reports/` are what surfaced it.
- RTF below 0.1 everywhere on the A10 means the full large-v3 pipeline
  clears real time with >10x headroom, and CPU int8 keeps even small at
  ~2x real time with tiny at 8.5x.

### Diarization

| backend | speaker count | threshold | DER | miss | false alarm | confusion |
|---|---|---|---:|---:|---:|---:|
| clustered (from parts) | estimated | 0.75 (calibrated) | **4.36%** | 4.36% | 0.00% | 0.00% |
| clustered (from parts) | estimated | 0.60 (old default) | 5.64% | 4.36% | 0.00% | 1.28% |
| clustered (from parts) | oracle | 0.60 | 4.36% | 4.36% | 0.00% | 0.00% |
| pyannote-3.1 (reference) | estimated | n/a | 9.80% | 7.57% | 0.00% | 2.24% |

- The DER is **identical across every ASR configuration and across
  Windows-CPU vs Linux-A10**, per mixture to three decimals: the diarizer
  is fully decoupled from the ASR and deterministic across platforms.
- Given the true speaker count, confusion drops to exactly zero: clustering
  itself attributes no time to the wrong speaker on these mixtures. The
  whole 1.28% confusion component at the old 0.60 threshold came from
  overcounting speakers (3->4, 4->5 on a few mixtures), and the 4.36% floor
  is missed speech at VAD boundaries. Each error component points at
  exactly one stage to improve, and the threshold calibration below
  recovered the confusion component without being given the count.
- **The from-parts pipeline beats the pretrained reference here, and the
  caveat matters as much as the number.** pyannote-3.1 scores 9.80% on the
  same mixtures under the same DER implementation, with the gap mostly in
  missed speech (7.57% vs 4.36%): its segmentation trims utterance
  boundaries more aggressively than silero VAD on clean read speech with
  hard turn changes. That is the regime this benchmark constructs and the
  regime the simple pipeline is built for; pyannote is tuned for
  conversational audio with overlap and boundary ambiguity, none of which
  exists here. It estimated the speaker count correctly on 11 of 12
  mixtures (one 2->3 overcount). The honest claim is narrow: on
  non-overlapped read-speech mixtures, VAD + ECAPA + agglomerative
  clustering is sufficient and the heavier pipeline buys nothing; the
  overlapped-speech mixtures planned below are where the ranking should
  flip.

### Threshold calibration

The clustering distance threshold is the from-parts pipeline's one real
hyperparameter. `bench sweep` calibrates it honestly: embeddings are
computed once per mixture, clustering is rerun across a 0.30-0.90 grid,
the threshold is chosen on six mixtures and scored on the six it never
saw. Full grid in `reports/threshold_sweep.json`.

| threshold | calibration DER | held-out DER | speaker count correct |
|---:|---:|---:|---:|
| 0.50 | 28.83% | 32.62% | 0/12 |
| 0.55 | 17.36% | 15.78% | 3/12 |
| 0.60 | 5.75% | 5.52% | 7/12 |
| 0.70 | 5.34% | 4.42% | 9/12 |
| **0.75** | **4.94%** | **3.73%** | **12/12** |
| 0.80 | 4.94% | 5.62% | 11/12 |
| 0.85 | 4.94% | 11.50% | 10/12 |
| 0.90 | 7.05% | 33.74% | 5/12 |

What the sweep says:

- **Calibration recovers the oracle bound without the oracle.** At 0.75 the
  speaker count is right on all 12 mixtures, confusion is exactly zero, and
  pooled DER equals the 4.36% VAD miss floor, the same number the
  oracle-count condition reaches. The threshold is now the shipped default,
  and the benchmark report regenerated at it confirms the sweep's numbers.
- **The failure mode is asymmetric.** From 0.60 to 0.80 the curve is nearly
  flat; above 0.80 it collapses (11.5% at 0.85, 33.7% at 0.90) as distinct
  speakers merge into one cluster. Undercutting the threshold splits one
  speaker into two, which the Hungarian mapping partially forgives;
  overcutting merges two speakers into one, which it cannot. Err low.
- The usual caveat: 0.75 is calibrated on clean read speech. The value to
  trust is the shape of the curve and the calibration procedure, not the
  constant; on a new acoustic regime, rerun the sweep.

## Repository layout

```
src/diarlab/
  metrics.py     # WER + DER from scratch, tested against hand-computed values
  align.py       # word -> speaker assignment rules (max overlap, gap fallback)
  windows.py     # VAD post-processing, embedding windows, turn building (pure)
  cluster.py     # agglomerative clustering over cosine distance
  mixtures.py    # synthetic conversations with exact ground truth
  asr.py         # faster-whisper wrapper (lazy import)
  vad.py         # silero VAD wrapper (lazy import)
  embeddings.py  # ECAPA-TDNN wrapper (lazy import)
  diarize.py     # the two backends behind one interface
  formats.py     # JSON / SRT / RTTM writers
  audio.py       # mono float32 loading + polyphase resampling
  cli.py         # transcribe / diarize / attribute
  server.py      # FastAPI upload -> job -> JSON/SRT (pipeline injectable)
  bench.py       # fetch / build / run / sweep
tests/           # CPU-only, no model downloads (the server is tested with
                 # an injected stub pipeline)
```

## What I'd do next

1. **Looser VAD padding for the 4.36% miss floor**, now the only error
   component left in the from-parts diarizer after calibration zeroed the
   confusion.
2. **Overlapped-speech mixtures**: the current benchmark has none, which
   flatters every system; partial-overlap construction is the obvious next
   stressor and the place the pyannote comparison should get interesting.
3. **Real conversational data**: AMI headset mix as a second benchmark with
   published baselines to sanity-check against.
4. **Streaming output on the server**: a WebSocket endpoint that emits
   speaker-attributed segments as the decode progresses, instead of one
   JSON at the end. The job API is the right substrate for it.

## License

MIT
