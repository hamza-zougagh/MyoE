"""Personalize from user-provided NumPy recordings, then predict one trial.

Calibration is an NPZ with `signals` (16, 10240, 16) and `labels` (16,).
The query is an NPZ with `signal` (10240, 16). Signals are physical mV at
2048 Hz with channels F1–F16 in that order.
"""

import argparse
from pathlib import Path

import numpy as np

from myoe import load_pretrained, personalize, predict_recording, preprocess, save_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("query", type=Path)
    parser.add_argument("--mode", choices=("standard", "fast"), default="standard")
    parser.add_argument("--output", type=Path, default=Path("personalized.npz"))
    args = parser.parse_args()

    with np.load(args.calibration, allow_pickle=False) as data:
        signals = data["signals"]
        labels = data["labels"]
        if signals.shape != (16, 10240, 16) or labels.shape != (16,):
            raise ValueError("Calibration requires signals (16,10240,16) and labels (16,)")
        windows = np.concatenate([preprocess(signal) for signal in signals])
        window_labels = np.repeat(labels, 39)
    model = personalize(load_pretrained(), windows, window_labels, mode=args.mode, seed=2101)
    with np.load(args.query, allow_pickle=False) as data:
        predictions = predict_recording(model, data["signal"])
    save_model(model, args.output)
    print("Predicted gesture per window:", predictions["labels"].tolist())
    print("Saved personalized model:", args.output)


if __name__ == "__main__":
    main()
