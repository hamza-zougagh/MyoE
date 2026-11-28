# Benchmark record

## Completed studies

| Study | Encoder source pool | Target group | Budget 1 E macro-F1 | Budget 3 E macro-F1 | Interpretation |
|---|---|---|---:|---:|---|
| Session 1 Study 2 | 18 Session 1 people, three seeds | 25 source-disjoint targets | 0.6447 | 0.7476 | Exploratory |
| Session 2 A/C/E | Same 18 Session 1 people, three seeds | Same 25 targets on another day | 0.6846 | 0.7786 | Prospective within-cohort cross-session |

Budget-one means average three calibration draws by three seeds within person. Budget-three means average three seeds within person. Twenty-five person means are then weighted equally. The values do not isolate the single seed-2101 release artifact.

## E-V2 development benchmark

The architecture selection used a separate E-12 source checkpoint trained on 12 source people. It did not evaluate the bundled E-18 checkpoint. Six previously exposed source-validation participants were each evaluated with calibration trial 1 and trial 2, separately, on query trials 3–7. Participant values average the two calibration cases.

| Deidentified participant | E-12 macro-F1 | V2-M | V2-A |
|---|---:|---:|---:|
| P1 | 0.644466 | 0.634358 | 0.650413 |
| P2 | 0.633719 | 0.607931 | 0.631016 |
| P3 | 0.568982 | 0.645169 | 0.623274 |
| P4 | 0.779865 | 0.723580 | 0.761731 |
| P5 | 0.604074 | 0.599665 | 0.597420 |
| P6 | 0.563881 | 0.521829 | 0.558965 |
| Equal-person mean | 0.632498 | 0.622089 | 0.637137 |

The fixed advancement rule required a mean gain of at least 0.01 and wins for at least four of six people. Neither candidate advanced. This was exploratory architecture selection with one seed and already exposed participants.

## Historical CPU measurements

All three encoders used the same support window and one CPU thread, with 100 warmups and 1,000 repetitions. The model measurement included the affine head. The end-to-end measurement included original per-window preprocessing and omitted acquisition and I/O. Median and p95 E-12 times were 0.240/0.249 ms for the model and 0.481/0.507 ms with preprocessing. Its serialized encoder-plus-head payload was 43,781 bytes. The historical 22.3–26.2 s E trajectory timings included checkpoint/head instrumentation and are not directly comparable to later candidate-only training times. Measurements reflect the Ryzen 5 development laptop and its software environment.

Historical timings do not independently establish performance of the standalone package or exact E-18 release artifact. Reproduction should record CPU, operating system, dependency versions, thread counts, input shapes, warmups and repetitions.

Standalone package measurements and software verification are recorded in [release-verification.md](release-verification.md).

A synthetic calibration timing using the packaged E-18 encoder on the Ryzen 5 development laptop, with one CPU thread, NumPy 2.5.3 and PyTorch 2.14, took 19.03 seconds for 2,000 adaptation updates and the final calibration-head fit. This synthetic run does not measure recognition performance. The original pipeline produced bit-identical results for the tested synthetic fixture.

On the same Windows system with Python 3.14.4, PyTorch 2.14 CPU, one CPU thread, 100 warmups and 1,000 iterations, public batch-one model-only inference measured 0.452/0.625 ms median/p95. Exact first-window preprocessing plus public probability inference measured 0.849/1.099 ms. The full five-second, 39-window public recording API measured 16.234/20.739 ms. No acquisition or disk I/O was included. The tested personalized NPZ payload was 42,800 bytes and contained 9,008 parameters. These are synthetic software measurements, not recognition outcomes.

## Optional Fast personalization (`mode="fast"`)

Exploratory GRABMyo Session 1 evaluation (`truncated_1000`, 1,000 updates on the 2,000-step warmup-plus-cosine schedule vs. `standard` 2,000 updates across 25 source-disjoint Session 1 targets):

- **1-shot calibration (recommended for Fast mode)**: all-seeds mean macro-F1 delta `-0.0019` (`0/25` `< -0.01`, worst `-0.0069`); seed-2101 all-draws mean delta `-0.0023` (`1/25` `< -0.01`); 100% stable; ~1.72×–1.93× faster.
- **3-shot calibration (optional/caveated)**: all-seeds mean macro-F1 delta `-0.0059` (`7/25` `< -0.01`, worst `-0.0142`; seed-2101 `4/25` `< -0.01`, worst `-0.0186`). `Standard` (`T=2000`) remains the default for multi-repetition calibration.

Synthetic CPU software timings from `python benchmarks/latency.py` (Windows 11, Ryzen 5 CPU, Python 3.14.4, PyTorch 2.15.0+cpu, one CPU thread, 100 warmups and 1,000 timed calls; fabricated inputs, not recognition outcomes):

| Mode | Updates | Personalization time | Single-window preproc + inference (median / p95) | Archive size |
|---|---:|---:|---:|---:|
| `standard` | 2,000 | 21.83 s | 0.759 / 1.053 ms | 43,084 bytes |
| `fast` | 1,000 | 9.53 s (`2.29×` faster) | 0.755 / 1.012 ms | 43,093 bytes |
