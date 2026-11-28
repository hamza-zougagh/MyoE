import io
import numpy as np
import pytest
import torch
import myoe
import json
import math
import zipfile
from myoe.core import PERSONALIZATION_CONFIGS, _learning_rate, _model_hash


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


def test_personalize_standard_matches_default_and_roundtrip():
    model = myoe.load_pretrained()
    x, y = support()
    default_model = myoe.personalize(model, x, y, participant=17, draw=1)
    explicit_standard = myoe.personalize(model, x, y, mode="standard", participant=17, draw=1)
    assert _model_hash(default_model.encoder) == _model_hash(explicit_standard.encoder)
    assert np.array_equal(default_model.weights, explicit_standard.weights)
    assert np.array_equal(default_model.bias, explicit_standard.bias)
    assert default_model.personalization_mode == "standard"
    assert default_model.metadata == PERSONALIZATION_CONFIGS["standard"]
    buf = io.BytesIO()
    myoe.save_model(default_model, buf)
    buf.seek(0)
    restored = myoe.load_model(buf)
    assert _model_hash(restored.encoder) == _model_hash(default_model.encoder)
    assert np.array_equal(restored.weights, default_model.weights)
    assert np.array_equal(restored.bias, default_model.bias)
    assert restored.personalization_mode == "standard"
    assert restored.metadata == PERSONALIZATION_CONFIGS["standard"]


def test_personalize_fast_mode_schedule_determinism_and_roundtrip():
    assert math.isclose(_learning_rate(1), 0.001 / 100, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(_learning_rate(100), 0.001, rel_tol=0, abs_tol=1e-15)
    expected_1000 = 0.00001 + (0.001 - 0.00001) / 2 * (1 + math.cos(math.pi * 900 / 1900))
    assert math.isclose(_learning_rate(1000), expected_1000, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(_learning_rate(2000), 0.00001, rel_tol=0, abs_tol=1e-15)

    model = myoe.load_pretrained()
    x, y = support()
    calls = []
    fast_a = myoe.personalize(
        model, x, y, mode="fast", participant=17, draw=1,
        progress=lambda step, total: calls.append((step, total)),
    )
    fast_b = myoe.personalize(model, x, y, mode="fast", participant=17, draw=1)
    assert len(calls) == 1000 and calls[0] == (1, 1000) and calls[-1] == (1000, 1000)
    assert _model_hash(fast_a.encoder) == _model_hash(fast_b.encoder)
    assert np.array_equal(fast_a.weights, fast_b.weights)
    assert np.array_equal(fast_a.bias, fast_b.bias)
    assert fast_a.personalization_mode == "fast"
    assert fast_a.metadata == {
        "personalization_mode": "fast",
        "adaptation_steps": 1000,
        "warmup_steps": 100,
        "schedule_total_steps": 2000,
        "schedule_type": "truncated_from_2000_cosine",
        "peak_lr": 0.001,
        "min_lr": 0.00001,
    }
    pred = myoe.predict_windows(fast_a, x[:16])
    assert pred["probabilities"].shape == (16, 16)
    assert np.allclose(pred["probabilities"].sum(axis=1), 1.0)
    assert np.all((pred["labels"] >= 1) & (pred["labels"] <= 16))

    buf = io.BytesIO()
    myoe.save_model(fast_a, buf)
    buf.seek(0)
    restored = myoe.load_model(buf)
    assert restored.personalization_mode == "fast"
    assert restored.metadata == fast_a.metadata
    assert _model_hash(restored.encoder) == _model_hash(fast_a.encoder)
    assert np.array_equal(restored.weights, fast_a.weights)
    assert np.array_equal(restored.bias, fast_a.bias)
    restored_pred = myoe.predict_windows(restored, x[:16])
    assert np.array_equal(pred["labels"], restored_pred["labels"])
    assert np.array_equal(pred["probabilities"], restored_pred["probabilities"])


def test_legacy_v010_archive_loads_cleanly_without_metadata():
    model = myoe.load_pretrained()
    x, y = support()
    calibrated = myoe.fit_head(model, x, y)
    # Construct a legacy v0.1.0 NPZ containing only encoder + head arrays and no metadata.json.
    arrays = {name: value.detach().cpu().numpy().copy() for name, value in calibrated.encoder.state_dict().items()}
    arrays["head.weights"] = calibrated.weights.copy()
    arrays["head.bias"] = calibrated.bias.copy()
    legacy_buf = io.BytesIO()
    np.savez(legacy_buf, **arrays)
    legacy_buf.seek(0)
    with zipfile.ZipFile(legacy_buf) as archive:
        assert "metadata.json" not in archive.namelist()
    legacy_buf.seek(0)
    restored = myoe.load_model(legacy_buf)
    assert _model_hash(restored.encoder) == _model_hash(calibrated.encoder)
    assert np.array_equal(restored.weights, calibrated.weights)
    assert np.array_equal(restored.bias, calibrated.bias)
    assert restored.personalization_mode == "standard"
    assert restored.metadata == PERSONALIZATION_CONFIGS["standard"]
    orig_pred = myoe.predict_windows(calibrated, x[:16])
    restored_pred = myoe.predict_windows(restored, x[:16])
    assert np.array_equal(orig_pred["labels"], restored_pred["labels"])
    assert np.array_equal(orig_pred["probabilities"], restored_pred["probabilities"])


@pytest.mark.parametrize("bad_mode", ["turbo", "", "FAST", None, 123])
def test_invalid_personalization_mode_raises(bad_mode):
    model = myoe.load_pretrained()
    x, y = support()
    with pytest.raises(ValueError, match="Unknown personalization mode"):
        myoe.personalize(model, x, y, mode=bad_mode)


def test_malformed_metadata_in_archive_rejected():
    model = myoe.load_pretrained()
    x, y = support()
    calibrated = myoe.fit_head(model, x, y)
    base = io.BytesIO()
    myoe.save_model(calibrated, base)
    for bad_meta in [
        b"not json",
        json.dumps({"personalization_mode": "turbo"}).encode("utf-8"),
        b" " * 5000,
    ]:
        buf = io.BytesIO(base.getvalue())
        with zipfile.ZipFile(buf, "a") as archive:
            archive.writestr("metadata.json", bad_meta)
        buf.seek(0)
        with pytest.raises(ValueError):
            myoe.load_model(buf)
