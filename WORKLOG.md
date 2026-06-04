# Work log

Running notes on what was done and what each step found. Newest first.
Numbers here are snapshots; the README tables are the source of truth.

## 2026-06-04: live mode

- **Built the live capture path**: a `/live` page grabs tab, screen, or
  microphone audio in the browser and streams int16 PCM over a
  WebSocket; the server resamples, processes rolling 5-second chunks,
  and pushes speaker-attributed segments back about one chunk behind
  real time. Inference shares the single worker with uploaded jobs, so
  everything serializes on one GPU.
- **Online speaker tracking** keeps identities stable across chunks:
  cluster each chunk locally, match cluster centroids against running
  global centroids, mint a new id only when nothing is close. First
  attempt reused the dendrogram threshold (0.75) for centroid matching
  and merged both speakers of mix_000 into one; centroids are denoised
  means, so cross-speaker centroid distances run far below window-level
  linkage distances. Measured the actual separation on the benchmark
  mixtures (same-speaker 0.20-0.42, different-speaker 0.58+) and set
  the centroid threshold to 0.50, the midpoint with margin. After the
  fix, streaming mix_000 attributes every utterance to the right
  speaker with stable ids end to end.
- Found and fixed a phantom-speaker bug along the way: a chunk with
  detectable speech but no decodable words minted an id that never
  recurred. The tracker now only sees chunks that produced words.
  Boundary slivers can still orphan an id (documented, cosmetic).
- The known limits are in the module docstring and on the page itself:
  chunk boundaries split words, and concurrent speech still logs one
  speaker at a time, exactly as the overlap benchmark predicts.

## 2026-06-04: domain recalibration and the demo reference toggle

- **Recalibrated on AMI dev meetings** (`fetch-ami --dev`, then
  `sweep --set ami-dev`): three dev-partition meetings, one per site,
  never the test meetings. The threshold sweep (extended grid, since
  spontaneous-speech embeddings spread wider) chose 0.80; the pad sweep
  confirmed 0.25 was already right. One constant changed.
- **That one change closed 40% of the gap to pyannote** on the test
  meetings: 18.71% to 16.55% DER, confusion 5.34% to 3.25%. The
  calibration procedure transfers across domains even though the
  constants do not, which was the claim worth testing.
- **What it did not fix**: speaker counting. Even at 0.80 the meetings
  cluster to 13 to 32 speakers for 4. Fixed-distance dendrogram cutting
  is the wrong stopping rule for spontaneous speech; a duration-aware
  merge or eigengap criterion is the new top roadmap item.
- The int8 rerun also reproduced the float16 repetition loop finding
  from the other side: same audio, 26.6% pooled WER vs 40.7%.
- **Demo: reference vs hypothesis toggle.** The timeline now outlines
  the ground-truth turns under the predicted bands (reference speakers
  matched to predicted ones by overlap), so boundary errors and missed
  speech are visible per pixel. Reference turns for the three demo
  mixtures ship as static JSON next to the predictions.

## 2026-06-04: AMI headset mix, the reality check

- **Added AMI as a third benchmark set** (`bench fetch-ami`, then
  `run --set ami`): three 4-speaker meetings from the test partition,
  references parsed from the NXT segment annotations (which genuinely
  overlap), words from the word-level XML. Same manifest schema as the
  synthetic sets, so the scoring path is identical.
- **The ranking flipped, as hypothesized.** pyannote-3.1 13.25% DER vs
  18.71% for the from-parts pipeline. After two synthetic wins for the
  simple pipeline, real meetings are where the pretrained model earns
  its keep. This is the result that makes the earlier wins credible.
- **Calibration did not transfer, loudly.** The threshold calibrated on
  read speech estimated 24-50 speakers for 4-person meetings. The caveat
  shipped next to the calibrated default was correct in practice, and
  per-domain recalibration moved to the top of the roadmap.
- **Second ASR tail event of the project**: small/float16 repetition
  loop on one meeting (92.5% WER; same audio at int8: 23.6%). Pooled
  means hid it; per-file rows surfaced it. Same lesson as the int8
  word-drop on the synthetic set.
- Anchored against the pyannote model card (18.8% on full AMI test, no
  collar, word-based references) with protocol differences stated
  rather than implied comparability.

## 2026-06-04: pad calibration, overlapped speech, streaming

- **Calibrated the VAD edge padding** with the same held-out protocol as
  the threshold (`bench sweep --param pad`). The 4.36% "miss floor" turned
  out to be VAD edge trimming: padding regions by 250 ms takes pooled DER
  from 4.36% to 0.38%, speaker count right on 12/12. The calibrated pad
  equals the DER collar, which is partly an artifact of how DER scores
  boundaries; noted in the README rather than glossed over. Past 0.30 the
  padding bridges speaker gaps and confusion returns. Reran the threshold
  sweep at the new pad: 0.75 still wins, so the shipped pair is jointly
  verified.
- **Built the overlapped-speech benchmark.** Mixture construction now
  supports seeded partial overlap at speaker changes (waveforms sum,
  reference turns genuinely overlap, capped at half the shorter
  utterance). Two regressions caught during development: an extra rng
  draw and integer/float rounding drift would each have silently changed
  the seeded plain mixtures; the final construction is byte-identical to
  the original (verified by checksum), so all published reports stay
  valid.
- **Scored both backends on the overlap set** (6.5% of speech time
  overlapped, no recalibration): clustered 4.50% DER vs pyannote-3.1 at
  11.73%. The clustered miss (4.13%) tracks the overlapped time, which is
  structural: the pipeline emits one speaker at a time, so it must miss
  the second concurrent speaker. The expected ranking flip toward
  pyannote did not happen; left open as a hypothesis for longer overlaps
  and real corpora.
- **Streaming output on the server.** faster-whisper decodes lazily, so
  the server now diarizes first (the fast stage) and attributes each ASR
  segment as it decodes. A WebSocket at `/jobs/{id}/stream` emits
  segments live and replays the stream for late connections. Verified
  with stub-pipeline tests and against real models on the A10.
- Ops note: the GitHub remote was force-pushed from a stale clone twice
  today, the second time wiping pushed work. Restored from the local
  superset with `--force-with-lease` after verifying the remote had
  nothing unique. Lesson: `git fetch && git reset --hard origin/main` on
  secondary machines before touching anything.

## 2026-06-04: threshold calibration and the serving layer

- **Calibrated the clustering distance threshold** on held-out mixtures
  (`bench sweep`): embeddings computed once per mixture, clustering rerun
  across a 0.30-0.90 grid, chosen on six mixtures, scored on the six it
  never saw. The calibrated 0.75 got the speaker count right on all 12
  mixtures and recovered the oracle-count DER bound without the oracle.
  The failure mode is asymmetric: under-merging is partially forgiven by
  the Hungarian mapping, over-merging is unrecoverable, so the curve
  collapses above 0.80. Shipped 0.75 as the default.
- **Added the HTTP serving layer**: FastAPI behind a `serve` extra, jobs
  on one worker thread (the RTF table is the capacity argument: ~45x real
  time per A10 with small/float16). The pipeline function is injectable,
  so the tests cover the whole HTTP surface with a stub and CI never
  downloads a model.
- Swept the remaining typographic artifacts out of the prose.

## 2026-06-03: pyannote reference and the demo page

- **Closed the pyannote comparison** the repo was built to make: 9.80%
  DER for the pretrained reference vs 5.64% for the from-parts pipeline,
  same mixtures, same from-scratch scorer. Getting pyannote 3.x running
  on a modern stack took three pins/fixes, each now documented in
  pyproject: torchaudio < 2.9 (AudioMetaData removed), huggingface_hub
  < 1.0 (use_auth_token removed), and a torch.load weights_only
  allowlist for the checkpoint's pickled globals.
- **Built the demo page**: a single static HTML file over the pipeline's
  own JSON output. Per-speaker timeline, color-coded transcript,
  word-level highlight synced to the audio, click anything to seek.
  Deployed to GitHub Pages via Actions from `demo/`.
- The demo paid for itself immediately: mix_000's transcript ends with
  "Thank you for watching.", a Whisper hallucination over trailing
  silence, absent from the reference, audible in one listen, invisible
  in pooled WER.

## 2026-06-03: benchmark numbers (A10 and CPU)

- WER by model size and compute type, DER decomposition, and real-time
  factors across an NVIDIA A10 and a consumer CPU. Headlines: small beats
  large-v3 on clean read speech at a quarter of the RTF; int8 is
  accuracy-neutral except one mixture where the small model silently
  drops 59 words (the per-mixture rows surfaced it, the pooled number
  hid it); DER identical across every ASR config and across platforms,
  confirming the diarizer is decoupled and deterministic.
- Oracle-speaker-count condition isolated the error budget: all
  confusion came from overcounting speakers, the rest is VAD miss.

## 2026-06-03: scaffold

- Pipeline (faster-whisper + silero VAD + ECAPA + agglomerative
  clustering + max-overlap alignment), WER and DER implemented from
  scratch against hand-computed values, seeded LibriSpeech mixtures with
  exact ground truth, 63 CPU-only tests, CI.
