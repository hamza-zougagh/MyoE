# Release verification

This record covers MyoE v0.1.0 software behavior, tensor provenance and synthetic measurements. It does not establish independent-participant generalization.

## Encoder provenance

The bundled encoder is the 18-source GRABMyo Session 1 source encoder trained for 20,000 updates with seed 2101. Its original trusted Torch checkpoint had SHA-256 `b14dcd6b09fb7bbf322ce36290c69522bebc73eb2b8233ea5576fd61a3ccc1ad`. The original checkpoint was read locally with `weights_only=True` during conversion. The sorted, typed, shaped encoder state digest was `46efd84b1f06befb2e2c9ffa66dc37977cbb6d6a8e57e62d951ae99c88204178` before and after conversion. The shipped numeric NPZ SHA-256 is `1cc9a0afec84abebd0e0b9526c5d521ccbf382e98a5997722a8b6c81404477df`. It contains only the eight encoder arrays; it contains no head, recording, exposure log, or target data. The project owner approved its redistribution under CC BY 4.0.

## Exact synthetic equivalence

On the pinned research environment, a synthetic five-second, 16-channel recording produced bit-identical 39 x 16 x 20 preprocessed windows. A separate synthetic one-repetition-per-class episode used 624 windows, RNG seed 2101, participant stream coordinate 17, and draw 1. The original research pipeline and package each ran all 2,000 updates and solved the final calibration-only head. Encoder state hashes matched exactly, and all final weights and biases matched bit-for-bit. Source initialization was the verified encoder. No real recordings were accessed. The independent comparison is retained in the local research evidence. Timings were 24.74 s for the original instrumented path and 19.03 s for the package path, synthetic, on the machine below. These figures do not estimate accuracy or clinical performance.

## Installation and tests

Windows 11 on the development Ryzen 5 CPU, Python 3.14.4. The pinned research environment with Torch 2.14.0+cpu, NumPy 2.5.3, and SciPy 1.18.1 passed the synthetic original-pipeline equivalence above. The 8-case core test suite passed in a separate fresh Python virtual environment with independently installed CPU Torch 2.15.0+cpu, NumPy 2.5.5, SciPy 1.18.1, pytest 9.2.0, and Streamlit 1.67.0. The wheel was installed from disk, imported from the fresh environment's site-packages while the working directory was outside the repository, and tested with pytest importlib mode. This fresh-environment result tests compatibility, not cross-version bitwise scientific identity.

The previously verified build of `myoe-0.1.0-py3-none-any.whl` had SHA-256 `4a5b399a80ca42d1a876532ab8bf4d0294636fa175fddef81e5582204c9d547a`. `myoe/core.py` SHA-256 `f10e6d8633ed228ef0279c531ac8bc97cfd2e179e6b5e8920913c0f47a36f0b7`. The test suite includes invalid archive shapes, oversized archives, malformed NPY headers, object-array rejection, invalid calibration, finite inputs, reproducible personalization, and export/import.

The GitHub Actions matrix passed on Windows and Ubuntu with Python 3.11 and 3.14 for source commit `7fabd627f4655e73b37f3e052fbe124b93461c86`. WSL was not available on the development machine. Numerical reproducibility across OS, hardware and library versions is not guaranteed.

## Synthetic CPU measurements

Run `python benchmarks/latency.py` from a checkout after installation. Windows 11, Ryzen 5 CPU, Python 3.14.4, Torch 2.14.0+cpu, one CPU thread, 100 warmups and 1,000 timed calls, synthetic physical-mV trial. The latency fixture used `fit_head` on the frozen source encoder, with the same encoder architecture and classifier array shapes as a fully personalized model. Model-only batch-one public probability inference measured median 0.45245 ms and p95 0.625205 ms. A single exact first-window RMS transformation plus the same public prediction API measured median 0.8493 ms and p95 1.098535 ms. The complete five-second, 39-window `predict_recording` path measured median 16.2339 ms and p95 20.73861 ms. These timings exclude file I/O, signal acquisition, and waiting for a recording; actual performance varies by machine and load. The public implementation also computes probability outputs and validates inputs. The original historical 0.48 ms per-window result used an internal benchmark path and is not directly interchangeable with these measurements.

The source encoder has 7,968 parameters. A personalized model has 9,008 parameters, including the 16-by-64 head and 16 biases. The tested uncompressed numeric NPZ export of the frozen-encoder `fit_head` fixture was 42,800 bytes; a fully personalized model has the same array shapes. Historic ~43.8 kB refers to a Torch serialization in another benchmark; file size depends on the format.
