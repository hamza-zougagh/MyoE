"""Evaluate separately supplied calibration and query recordings.

Each NPZ must contain `signals` with shape (N,10240,16) and integer `labels`
with shape (N,). Supply 16 support recordings, one for each gesture. Each
query file must be disjoint from the support recordings and, ideally, from
the source participants used to train the released encoder.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import myoe


def read(path):
    blob = Path(path).read_bytes()
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {'signals', 'labels'}:
            raise ValueError('Archive must contain signals and labels')
        signals, labels = data['signals'], data['labels']
    if signals.ndim != 3 or signals.shape[1:] != (10240, 16) or labels.shape != (len(signals),):
        raise ValueError('Expected (N,10240,16) and (N,) arrays')
    if not np.issubdtype(labels.dtype, np.integer) or not np.isfinite(signals).all() or not np.isin(labels, range(1,17)).all():
        raise ValueError('Invalid values or labels')
    identities = [hashlib.sha256(np.ascontiguousarray(signal).tobytes()).hexdigest() for signal in signals]
    windows = np.concatenate([myoe.preprocess(signal) for signal in signals])
    return windows, np.repeat(labels, 39), hashlib.sha256(blob).hexdigest(), identities


def metrics(y, p):
    predicted = p.argmax(axis=1) + 1
    scores = []
    for label in range(1,17):
        tp = np.sum((y == label) & (predicted == label))
        fp = np.sum((y != label) & (predicted == label))
        fn = np.sum((y == label) & (predicted != label))
        scores.append(float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.)
    return {'macro_f1':float(np.mean(scores)), 'accuracy':float(np.mean(predicted == y)),
            'cross_entropy':float(-np.log(np.clip(p[np.arange(len(y)), y-1], np.finfo(float).tiny, 1)).mean()),
            'f1_by_gesture':scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('support'); parser.add_argument('query')
    parser.add_argument('--output', default='benchmark-results.json')
    args = parser.parse_args()
    sx, sy, sh, si = read(args.support); qx, qy, qh, qi = read(args.query)
    if sh == qh or set(si) & set(qi):
        raise ValueError('Calibration and query files must differ')
    model = myoe.personalize(myoe.load_pretrained(), sx, sy)
    result = {'scope':'user supplied query data; disjointness is the operator responsibility',
              'support_sha256':sh, 'query_sha256':qh, 'window_metrics':metrics(qy,myoe.predict_windows(model,qx)['probabilities'])}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
