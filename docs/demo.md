# Recorded-signal demonstration

Start the application with `streamlit run demo/app.py`. Choose the procedurally generated synthetic example to explore the interface without a dataset download. Synthetic signals and their example predictions demonstrate the workflow and do not establish biological validity or recognition performance.

The app also accepts user-supplied `.npz` files with NumPy arrays. Signals must use physical millivolts, 2048 Hz sampling, channels F1–F16 in order and exactly five seconds per recording.

- A calibration file contains `signals` shaped `(16, 10240, 16)` and integer `labels` shaped `(16,)`, covering gesture IDs 1–16 once each.
- A query file contains `signal` shaped `(10240, 16)`. It may contain integer `labels` as a scalar for the recording or an array of 39 window labels.
- A personalized model is exported and imported as a validated NPZ file.

The source encoder has no classifier. To show a before-and-after comparison, the app fits a calibration-only head on the frozen source encoder, then compares it with the same encoder after full personalization and refitting the head. Both use the provided calibration recording. Replay, waveform, probabilities, latency and optional label-based metrics are calculated from the selected recording. The replay represents stored signals and does not acquire live data.
