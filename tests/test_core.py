import io
import numpy as np
import pytest
import torch
import myoe
from myoe.core import _model_hash


def support():
    rng = np.random.default_rng(42)
    x = np.abs(rng.normal(size=(16 * 39, 16, 20))).astype(np.float32)
    return x, np.repeat(np.arange(1, 17, dtype=np.int64), 39)


def test_preprocess_shape_zero_and_input_validation():
    assert np.array_equal(myoe.preprocess(np.zeros((10240, 16))), np.zeros((39,16,20), np.float32))
    with pytest.raises(ValueError): myoe.preprocess(np.zeros((100,16)))
    with pytest.raises(ValueError): myoe.preprocess(np.zeros((10240,16)), 1000)
    x = np.zeros((10240,16)); x[0,0] = np.nan
    with pytest.raises(ValueError): myoe.preprocess(x)


def test_pretrained_head_roundtrip_and_deterministic_prediction():
    model = myoe.load_pretrained()
    assert _model_hash(model.encoder) == '46efd84b1f06befb2e2c9ffa66dc37977cbb6d6a8e57e62d951ae99c88204178'
    x, y = support()
    with pytest.raises(ValueError): myoe.predict_windows(model, x)
    calibrated = myoe.fit_head(model, x, y)
    first = myoe.predict_windows(calibrated, x)
    assert first['probabilities'].shape == (len(x), 16)
    assert np.allclose(first['probabilities'].sum(axis=1), 1)
    buf = io.BytesIO(); myoe.save_model(calibrated, buf); buf.seek(0)
    restored = myoe.load_model(buf)
    second = myoe.predict_windows(restored, x)
    assert np.array_equal(first['labels'], second['labels'])
    assert np.array_equal(first['probabilities'], second['probabilities'])
    with pytest.raises(ValueError): myoe.fit_head(model, x, np.zeros(len(x),dtype=int))
    with pytest.raises(ValueError): myoe.predict_windows(calibrated, np.zeros((1,2,3)))


def test_untrusted_shapes_fail_closed():
    model = myoe.load_pretrained()
    buf = io.BytesIO()
    arrays = {name: value.detach().numpy() for name, value in model.encoder.state_dict().items()}
    arrays["head.weights"] = np.zeros(1)
    arrays["head.bias"] = np.zeros(16)
    np.savez(buf, **arrays)
    buf.seek(0)
    with pytest.raises(ValueError): myoe.load_model(buf)


def test_personalize_reproducible_and_progress():
    model = myoe.load_pretrained()
    x, y = support()
    calls=[]
    a = myoe.personalize(model, x, y, participant=17, draw=1, progress=lambda i, total: calls.append((i,total)))
    b = myoe.personalize(model, x, y, participant=17, draw=1)
    assert calls[0] == (1, 2000) and calls[-1] == (2000, 2000)
    assert _model_hash(a.encoder) == _model_hash(b.encoder)
    assert np.array_equal(a.weights, b.weights) and np.array_equal(a.bias, b.bias)
    assert model.weights is None
    with pytest.raises(ValueError): myoe.personalize(a, x, y)


def test_malicious_object_array_is_rejected():
    buf = io.BytesIO(); np.savez(buf, payload=np.array([object()],dtype=object)); buf.seek(0)
    with pytest.raises(ValueError): myoe.load_model(buf)


def test_extreme_signal_is_rejected():
    with pytest.raises(ValueError): myoe.preprocess(np.full((10240, 16), 1e308))


def test_missing_and_oversized_archives_are_rejected():
    missing=io.BytesIO(); np.savez(missing, example=np.zeros(1)); missing.seek(0)
    with pytest.raises(ValueError): myoe.load_model(missing)
    with pytest.raises(ValueError): myoe.load_model(io.BytesIO(b'x' * 2_000_001))


def test_malformed_shape_header_rejected_before_array_allocation():
    import zipfile
    model = myoe.load_pretrained()
    buf = io.BytesIO(); myoe.save_model(model, buf); buf.seek(0)
    bad = io.BytesIO()
    with zipfile.ZipFile(buf) as source, zipfile.ZipFile(bad, 'w') as target:
        for info in source.infolist():
            if info.filename == 'layers.0.weight.npy':
                header = io.BytesIO()
                np.lib.format.write_array_header_1_0(header, {'descr':'<f4','fortran_order':False,'shape':(999999999,)})
                target.writestr(info.filename, header.getvalue())
            else:
                target.writestr(info.filename, source.read(info))
    bad.seek(0)
    with pytest.raises(ValueError): myoe.load_model(bad)
