# Technical details

MyoE expects 16 forearm channels ordered F1 through F16, sampled at 2048 Hz and expressed in physical millivolts. A five-second recording is cropped to the fixed 0.5–4.5 s interval, then divided into 200 ms windows at 100 ms strides, yielding 39 windows. Twenty contiguous ten-millisecond RMS bins form the temporal axis. Each window's RMS values are divided by their global RMS and transformed using `log1p`. A zero window maps to zero. Preprocessing is per window and does not fit population normalization statistics.

The source encoder uses two Conv1d layers with kernel 3 and padding 1, GroupNorm with 8 groups and epsilon 1e-5, and ReLU. Channel widths are 16→32→64. Temporal mean and zero-safe L2 normalization produce 64-dimensional features. The affine head maps these features to 16 gesture classes.

Target adaptation uses the source initialization, deterministic sampling, Adam and the historical 2,000-update schedule. It discards the temporary training head and independently fits a final head on the real calibration windows. Class IDs are 1 through 16. See the [model card](model-card.md) for provenance.

`preprocess` accepts an array of shape `(10240, 16)` and returns 39 windows of shape `(16, 20)`. `predict_windows` operates on processed windows. The source checkpoint has no head, so `fit_head` or `personalize` must be called before prediction. Personalization expects 39 chronological windows for each of 16 gesture classes from one repetition per gesture. Use package functions to save and load personalized models as validated NPZ arrays. Load files only from trusted sources.

The source recordings used a specific electrode montage from healthy adult volunteers. Signal units, hardware, electrode placement and preprocessing must match or be separately validated before interpreting predictions.
