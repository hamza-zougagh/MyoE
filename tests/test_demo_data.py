import io
import zipfile

import numpy as np
import pytest

from demo.data import confusion_matrix, labels_for_windows, load_archive, synthetic_recording


def test_synthetic_is_deterministic_and_has_no_raw_data_dependency():
    a = synthetic_recording(3)
    assert a.shape == (10240, 16)
    assert np.array_equal(a, synthetic_recording(3))
    assert not np.array_equal(a, synthetic_recording(4))


def test_recording_archive_checks_numeric_data_and_labels():
    buf = io.BytesIO()
    np.savez_compressed(buf, signal=np.zeros((10240, 16)), labels=np.int64(3))
    signal, labels = load_archive(buf.getvalue(), 'recording')
    assert signal.shape == (10240, 16)
    assert np.array_equal(labels_for_windows(labels), np.full(39, 3))
    bad = io.BytesIO()
    np.savez(bad, signal=np.zeros((10240, 16)), labels=np.int64(17))
    with pytest.raises(ValueError):
        load_archive(bad.getvalue(), 'recording')


def test_calibration_requires_each_class_once():
    buf = io.BytesIO()
    np.savez_compressed(buf, signals=np.zeros((16, 10240, 16)), labels=np.ones(16, dtype=int))
    with pytest.raises(ValueError):
        load_archive(buf.getvalue(), 'calibration')


def test_confusion_counts():
    matrix = confusion_matrix(np.array([1, 1, 2]), np.array([1, 2, 2]))
    assert matrix.shape == (16, 16)
    assert matrix[0, 0] == matrix[0, 1] == matrix[1, 1] == 1
    assert matrix.sum() == 3


def test_archive_expanded_size_cap():
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('signal.npy', b'0' * (48 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match='expanded'):
        load_archive(out.getvalue(), 'recording')


def test_archive_rejects_invalid_header_before_numpy_load(monkeypatch):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {
        'descr': '<f8', 'fortran_order': False, 'shape': (10**9, 16),
    })
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w') as archive:
        archive.writestr('signal.npy', header.getvalue())
    def forbidden_load(*args, **kwargs):
        raise AssertionError('Invalid data reached np.load')
    monkeypatch.setattr(np, 'load', forbidden_load)
    with pytest.raises(ValueError, match='shape or numeric dtype'):
        load_archive(out.getvalue(), 'recording')


def test_archive_rejects_duplicate_and_unexpected_members():
    array = io.BytesIO()
    np.save(array, np.zeros((10240, 16)))
    for names in [('signal.npy', 'signal.npy'), ('signal.npy', 'extra.npy')]:
        out = io.BytesIO()
        with pytest.warns(UserWarning) if names[0] == names[1] else __import__('contextlib').nullcontext():
            with zipfile.ZipFile(out, 'w') as archive:
                for name in names:
                    archive.writestr(name, array.getvalue())
        with pytest.raises(ValueError, match='Unexpected or duplicate'):
            load_archive(out.getvalue(), 'recording')
