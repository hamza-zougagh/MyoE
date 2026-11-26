"""Fabricated signals and safe, bounded NPZ import for the MyoE demo."""
from io import BytesIO
import zipfile

import numpy as np

FS = 2048
SAMPLES = 5 * FS
CHANNELS = 16
MAX_ARCHIVE_BYTES = 48 * 1024 * 1024


def synthetic_recording(gesture, seed=19):
    """Produce a reproducible, fabricated signal with no donor recordings."""
    if not 1 <= int(gesture) <= 16:
        raise ValueError('Gesture must be an integer from 1 to 16')
    rng = np.random.default_rng(seed + 100 * int(gesture))
    t = np.arange(SAMPLES, dtype=np.float64) / FS
    c = np.arange(CHANNELS, dtype=np.float64)
    envelope = np.sin(np.pi * np.clip((t - .5) / 4, 0, 1)) ** 2
    envelope *= (.6 + .4 * np.cos(2 * np.pi * .7 * t) ** 2)
    shape = .4 + .6 * np.cos((c - (int(gesture) - 1)) * np.pi / 16) ** 2
    carrier = np.sin(2 * np.pi * (65 + 5 * int(gesture)) * t[:, None] + c[None, :] * .4)
    return (0.025 * rng.standard_normal((SAMPLES, CHANNELS)) +
            0.12 * envelope[:, None] * carrier * shape[None, :])


def synthetic_calibration():
    signals = np.stack([synthetic_recording(g, seed=19) for g in range(1, 17)])
    return signals, np.arange(1, 17, dtype=np.int64)


def load_archive(content, mode):
    """Validate the NPZ schema and NPY headers before materializing arrays."""
    if len(content) > MAX_ARCHIVE_BYTES:
        raise ValueError('The uploaded archive exceeds the 48 MiB limit.')
    if mode == 'calibration':
        required = {'signals.npy', 'labels.npy'}
        allowed = required
        shapes = {'signals.npy': {(16, SAMPLES, CHANNELS)}, 'labels.npy': {(16,)}}
    elif mode == 'recording':
        required = {'signal.npy'}
        allowed = {'signal.npy', 'labels.npy'}
        shapes = {'signal.npy': {(SAMPLES, CHANNELS)}, 'labels.npy': {(), (39,)}}
    else:
        raise ValueError('Unknown archive type')
    with zipfile.ZipFile(BytesIO(content)) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if (len(set(names)) != len(names) or not required.issubset(names)
                or not set(names).issubset(allowed)):
            raise ValueError('Unexpected or duplicate archive fields.')
        if sum(info.file_size for info in infos) > MAX_ARCHIVE_BYTES:
            raise ValueError('The expanded archive exceeds the 48 MiB limit.')
        for info in infos:
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unsupported NPY header version.')
                kinds = 'iu' if info.filename == 'labels.npy' else 'fiu'
                if (shape not in shapes[info.filename] or dtype.kind not in kinds
                        or dtype.itemsize not in (1, 2, 4, 8) or dtype.hasobject):
                    raise ValueError('Invalid array shape or numeric dtype.')
                expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                if stream.tell() + expected != info.file_size:
                    raise ValueError('Array length does not match its header.')
    with np.load(BytesIO(content), allow_pickle=False) as data:
        if mode == 'calibration':
            signal = np.asarray(data['signals'], dtype=np.float64)
            labels = np.asarray(data['labels'])
            if signal.shape != (16, SAMPLES, CHANNELS) or labels.shape != (16,):
                raise ValueError('Calibration needs signals (16, 10240, 16) and labels (16,).')
            if not np.issubdtype(labels.dtype, np.integer) or set(labels.tolist()) != set(range(1, 17)):
                raise ValueError('Calibration labels must contain each integer from 1 to 16 once.')
        elif mode == 'recording':
            signal = np.asarray(data['signal'], dtype=np.float64)
            labels = np.asarray(data['labels']) if 'labels' in data else None
            if signal.shape != (SAMPLES, CHANNELS):
                raise ValueError('Recording needs signal shaped (10240, 16).')
            if labels is not None and (labels.shape not in [(), (39,)] or
                    not np.issubdtype(labels.dtype, np.integer) or
                    np.any((labels < 1) | (labels > 16))):
                raise ValueError('Recording labels must be one integer or 39 integers from 1 to 16.')
        else:
            raise ValueError('Unknown archive type')
        if not np.isfinite(signal).all():
            raise ValueError('Signals must be finite.')
        return signal, labels


def labels_for_windows(labels):
    if labels is None:
        return None
    return np.full(39, int(labels), dtype=np.int64) if np.ndim(labels) == 0 else np.asarray(labels, dtype=np.int64)


def confusion_matrix(truth, predicted):
    result = np.zeros((16, 16), dtype=np.int64)
    np.add.at(result, (truth - 1, predicted - 1), 1)
    return result
