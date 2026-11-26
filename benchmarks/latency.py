"""Measure batch-one model and preprocessing latency on fabricated data."""
import io, json, os, platform, time
import numpy as np
from fractions import Fraction
import torch
import myoe

def measure(fn):
    for _ in range(100): fn()
    values = []
    for _ in range(1000):
        start = time.perf_counter_ns(); fn(); values.append((time.perf_counter_ns()-start)/1e6)
    return {'median_ms':float(np.median(values)),'p95_ms':float(np.quantile(values,.95))}

def main():
    torch.set_num_threads(1)
    rng=np.random.default_rng(42)
    signal=rng.normal(size=(10240,16))
    x=np.concatenate([myoe.preprocess(signal) for _ in range(16)])
    y=np.repeat(np.arange(1,17),39)
    model=myoe.fit_head(myoe.load_pretrained(),x,y)
    def model_one(): return myoe.predict_windows(model,x[:1])
    def single_window():
        edges = [int(2048*(Fraction(1,2)+Fraction(j,100))+Fraction(1,2)) for j in range(21)]
        energy = np.stack([np.sqrt(np.mean(signal[a:b]**2,axis=0,dtype=np.float64))
                           for a,b in zip(edges[:-1],edges[1:])],axis=1)
        q = np.sqrt(np.mean(energy**2,dtype=np.float64))
        return (np.log1p(energy/q) if q > 0 else np.zeros_like(energy)).astype(np.float32)
    assert np.array_equal(single_window(), myoe.preprocess(signal)[0])
    def window_and_predict(): return myoe.predict_windows(model,single_window()[None])
    # Public preprocess processes an entire 5 s trial, therefore this is per-trial, not per-window.
    buf=io.BytesIO();myoe.save_model(model,buf)
    result={'scope':'synthetic; full public API, one CPU thread', 'platform':platform.platform(),
            'processor':platform.processor(), 'python':platform.python_version(), 'torch':torch.__version__,
            'model_batch_one':measure(model_one), 'single_window_with_preprocessing':measure(window_and_predict),
            'full_recording':measure(lambda:myoe.predict_recording(model,signal)),
            'personalized_npz_bytes':len(buf.getvalue()),'parameters':sum(v.numel() for v in model.encoder.parameters())+1040}
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
