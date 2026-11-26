"""Original E encoder, personalization, and CPU inference."""
from __future__ import annotations

import copy
import hashlib
import io
import zipfile
import math
from fractions import Fraction
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from torch import nn

SOURCE_STATE_HASH = "46efd84b1f06befb2e2c9ffa66dc37977cbb6d6a8e57e62d951ae99c88204178"


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(16, 32, 3, padding=1), nn.GroupNorm(8, 32, eps=1e-5), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.GroupNorm(8, 64, eps=1e-5), nn.ReLU())

    def forward(self, x):
        z = self.layers(x).mean(dim=2)
        norm = torch.linalg.vector_norm(z, dim=1, keepdim=True)
        return z / torch.where(norm > 0, norm, torch.ones_like(norm))


def _new_encoder():
    with torch.random.fork_rng(devices=[]):
        return Encoder()


def _model_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


class MyoEModel:
    """A source encoder with an optional calibration-only classifier head."""
    def __init__(self, encoder, weights=None, bias=None):
        self.encoder = encoder.cpu().eval()
        self.weights = weights
        self.bias = bias

    def save(self, path):
        save_model(self, path)


def _read_payload(path):
    """Read the bounded numeric archive without enabling pickle."""
    if hasattr(path, "read"):
        blob = path.read(2_000_001)
    else:
        with open(path, "rb") as stream:
            blob = stream.read(2_000_001)
    if len(blob) > 2_000_000:
        raise ValueError("Model archive exceeds 2 MB")
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            info = archive.infolist()
            schema = {name: (tuple(value.shape), np.dtype("float32"))
                      for name, value in _new_encoder().state_dict().items()}
            head = {"head.weights": ((16, 64), np.dtype("float64")),
                    "head.bias": ((16,), np.dtype("float64"))}
            names = [item.filename[:-4] for item in info if item.filename.endswith(".npy")]
            if (len(names) != len(info) or len(names) != len(set(names)) or
                    set(names) not in (set(schema), set(schema) | set(head)) or
                    sum(item.file_size for item in info) > 2_000_000):
                raise ValueError("Invalid model archive members")
            schema.update(head)
            for item in info:
                if item.compress_size > 2_000_000 or item.file_size > 2_000_000:
                    raise ValueError("Model archive member exceeds limit")
                with archive.open(item) as member:
                    version = np.lib.format.read_magic(member)
                    if version != (1, 0):
                        raise ValueError("Unsupported numeric-array format")
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(member)
                    expected_shape, expected_dtype = schema[item.filename[:-4]]
                    if (shape != expected_shape or dtype != expected_dtype or fortran or
                            item.file_size - member.tell() != math.prod(shape) * dtype.itemsize):
                        raise ValueError("Invalid numeric-array header")
        with np.load(io.BytesIO(blob), allow_pickle=False) as arrays:
            payload = {name: arrays[name].copy() for name in arrays.files}
    except Exception as exc:
        raise ValueError("Invalid MyoE numeric archive") from exc
    return payload


def _from_payload(payload):
    model = _new_encoder()
    schema = model.state_dict()
    names = set(schema)
    if set(payload) not in (names, names | {"head.weights", "head.bias"}):
        raise ValueError("Invalid MyoE archive members")
    state = {}
    for name, template in schema.items():
        array = payload[name]
        if array.dtype != np.float32 or tuple(array.shape) != tuple(template.shape) or not np.isfinite(array).all():
            raise ValueError("Invalid encoder state array")
        state[name] = torch.from_numpy(array)
    model.load_state_dict(state, strict=True)
    weights, bias = payload.get("head.weights"), payload.get("head.bias")
    if weights is not None:
        if weights.dtype != np.float64 or bias.dtype != np.float64 or weights.shape != (16, 64) or bias.shape != (16,):
            raise ValueError("Invalid classifier arrays")
        if not np.isfinite(weights).all() or not np.isfinite(bias).all():
            raise ValueError("Nonfinite classifier parameters")
    return MyoEModel(model, weights, bias)


def load_pretrained(path=None):
    """Load the 18-source seed-2101 encoder; fit a head before prediction."""
    path = files("myoe").joinpath("data/source-2101.npz") if path is None else path
    model = _from_payload(_read_payload(path))
    if _model_hash(model.encoder) != SOURCE_STATE_HASH or model.weights is not None:
        raise ValueError("Pretrained encoder identity does not match the verified source")
    return model


def save_model(model, path):
    if not isinstance(model, MyoEModel):
        raise TypeError("Expected a MyoEModel")
    arrays = {name: value.detach().cpu().numpy().copy() for name, value in model.encoder.state_dict().items()}
    if model.weights is not None:
        arrays.update({"head.weights": model.weights, "head.bias": model.bias})
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    if hasattr(path, "write"):
        path.write(buffer.getvalue())
    else:
        Path(path).write_bytes(buffer.getvalue())


def load_model(path):
    """Load a schema-validated, non-pickle MyoE numeric archive."""
    return _from_payload(_read_payload(path))


def _sample_index(t):
    value = 2048 * t + Fraction(1, 2)
    return value.numerator // value.denominator


def preprocess(samples, sample_rate=2048):
    """Transform one 5-second, 16-channel physical-mV trial into 39 windows."""
    if sample_rate != 2048:
        raise ValueError("Expected 2048 Hz samples")
    x = np.asarray(samples, dtype=np.float64)
    if x.shape != (10240, 16) or not np.isfinite(x).all():
        raise ValueError("Expected finite (10240, 16) physical-mV samples")
    result = np.empty((39, 16, 20), dtype=np.float32)
    for k in range(39):
        edges = [_sample_index(Fraction(1, 2) + Fraction(k, 10) + Fraction(j, 100)) for j in range(21)]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                energy = np.stack([np.sqrt(np.mean(x[a:b] ** 2, axis=0, dtype=np.float64))
                                   for a, b in zip(edges[:-1], edges[1:])], axis=1)
                q = np.sqrt(np.mean(energy ** 2, dtype=np.float64))
                result[k] = (np.log1p(energy / q) if q > 0 else np.zeros_like(energy)).astype(np.float32)
            except FloatingPointError as exc:
                raise ValueError("Signal magnitude overflows preprocessing") from exc
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite transformed windows")
    return result


def _windows(windows):
    x = np.asarray(windows, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (16, 20) or not len(x) or not np.isfinite(x).all():
        raise ValueError("Expected finite windows with shape (N,16,20)")
    return np.ascontiguousarray(x)


def _calibration(windows, labels):
    x = _windows(windows)
    y = np.asarray(labels)
    if y.shape != (len(x),) or not np.issubdtype(y.dtype, np.integer) or set(y.tolist()) != set(range(1, 17)):
        raise ValueError("Calibration must contain integer gesture labels 1 through 16")
    if len(x) != 624 or any(np.count_nonzero(y == gesture) != 39 for gesture in range(1, 17)):
        raise ValueError("Provide exactly 39 windows from one trial per gesture")
    return x, y.astype(np.int64)


def _features(encoder, x):
    encoder.eval()
    with torch.inference_mode():
        return np.concatenate([encoder(torch.from_numpy(part.copy())).numpy().astype(np.float64)
                               for part in np.array_split(x, max(1, math.ceil(len(x) / 1024)))])


def fit_head(model, windows, labels):
    """Fit only the original calibration classifier on a frozen encoder."""
    x, y = _calibration(windows, labels)
    with threadpool_limits(limits=1):
        features = _features(model.encoder, x)
    counts = np.bincount(y, minlength=17)
    masses = np.asarray([1 / (16 * counts[g]) for g in y], dtype=np.float64)
    def objective(parameters):
        w, b = parameters[:1024].reshape(16, 64), parameters[1024:]
        logits = features @ w.T + b
        lp = logits - logsumexp(logits, axis=1, keepdims=True)
        value = .001 / 2 * (np.sum(w*w) + np.sum(b*b)) - np.dot(masses, lp[np.arange(len(y)), y-1])
        residual = np.exp(lp)
        residual[np.arange(len(y)), y-1] -= 1
        residual *= masses[:, None]
        gradient = np.concatenate(((.001*w + residual.T @ features).ravel(), .001*b + residual.sum(axis=0)))
        return float(value), gradient
    with threadpool_limits(limits=1):
        result = minimize(objective, np.zeros(1040, dtype=np.float64), jac=True, method="L-BFGS-B",
                          options={"maxiter": 1000, "maxfun": 20000, "maxls": 50, "maxcor": 20,
                                   "gtol": 1e-8, "ftol": 1e-15})
        _, gradient = objective(result.x)
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all() or not np.isfinite(gradient).all() or np.max(np.abs(gradient)) > 1e-6:
        raise RuntimeError("Calibration head failed the numerical convergence check")
    return MyoEModel(copy.deepcopy(model.encoder), result.x[:1024].reshape(16, 64).copy(), result.x[1024:].copy())


def _learning_rate(step):
    return .001 * step / 100 if step <= 100 else .00001 + (.001 - .00001) / 2 * (1 + math.cos(math.pi * (step-100) / 1900))


def _stream_seed(seed, participant, draw):
    key = f"P0-FC2-stream-v1|{seed}|target-sampling|{participant:02d}|{draw}"
    return int.from_bytes(hashlib.sha256(key.encode("ascii")).digest()[:8], "big")


def personalize(model, windows, labels, *, seed=2101, participant=1, draw=1, progress=None):
    """Run 2,000 full-encoder updates and refit a head on calibration windows."""
    x, y = _calibration(windows, labels)
    if not isinstance(seed, int) or not isinstance(participant, int) or not isinstance(draw, int) or participant < 1 or draw < 0:
        raise ValueError("seed, participant and draw must be integer stream coordinates")
    if model.weights is not None or _model_hash(model.encoder) != SOURCE_STATE_HASH:
        raise ValueError("Personalization must start from the verified source encoder")
    encoder = copy.deepcopy(model.encoder).train()
    # Isolate initialization from the caller's RNG even though these parameters are reset.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = nn.Linear(64, 16)
    with torch.no_grad():
        head.weight.zero_(); head.bias.zero_()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=.001,
                                 betas=(.9, .999), eps=1e-8, weight_decay=0)
    rng = np.random.default_rng(_stream_seed(seed, participant, draw))
    pools = {gesture: np.flatnonzero(y == gesture) for gesture in range(1, 17)}
    with threadpool_limits(limits=1):
        for step in range(1, 2001):
            selected = []
            for _ in range(128):
                rng.integers(1)  # person, including singleton draw
                gesture = int(rng.integers(1, 17))
                rng.integers(1)  # one calibration trial per gesture
                pool = pools[gesture]
                selected.append(int(pool[rng.integers(len(pool))]))
            for group in optimizer.param_groups:
                group["lr"] = _learning_rate(step)
            optimizer.zero_grad(set_to_none=True)
            logits = head(encoder(torch.from_numpy(x[selected].copy())))
            loss = (torch.nn.functional.cross_entropy(logits, torch.from_numpy(y[selected].copy() - 1))
                    + .0001 / 2 * sum(torch.sum(p*p) for p in encoder.parameters())
                    + .001 / 2 * sum(torch.sum(p*p) for p in head.parameters()))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            parameters = list(encoder.parameters()) + list(head.parameters())
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise RuntimeError("Nonfinite training gradient")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in parameters):
                raise RuntimeError("Nonfinite training update")
            if progress is not None:
                progress(step, 2000)
    return fit_head(MyoEModel(encoder), x, y)


def predict_windows(model, windows):
    if model.weights is None or model.bias is None:
        raise ValueError("Fit a calibration head or personalize before prediction")
    x = _windows(windows)
    logits = _features(model.encoder, x) @ model.weights.T + model.bias
    probabilities = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    return {"labels": logits.argmax(axis=1) + 1, "probabilities": probabilities}


def predict_recording(model, samples, sample_rate=2048):
    return predict_windows(model, preprocess(samples, sample_rate=sample_rate))
