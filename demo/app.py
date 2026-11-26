"""Explore MyoE with five-second, 16-channel recordings."""
from io import BytesIO
import time

import numpy as np
import pandas as pd
import streamlit as st

from demo.data import FS, confusion_matrix, labels_for_windows, load_archive, synthetic_calibration, synthetic_recording
from myoe import fit_head, load_model, load_pretrained, personalize, predict_recording, preprocess

st.set_page_config(page_title='MyoE | Recorded signal lab', page_icon='〰️', layout='wide')
st.markdown('''<style>
.stApp {background:#f5f9f8;color:#183331}
.block-container {max-width:1320px;padding-top:2rem}
h1,h2,h3 {letter-spacing:-0.035em;color:#123532}
[data-testid="stSidebar"] {background:#eaf4f2;border-right:1px solid #d1e4df}
[data-testid="stMetric"] {background:#fff;border:1px solid #dbeae7;border-radius:12px;padding:14px}
.stButton button, .stDownloadButton button {border-radius:9px}
.small-note {font-size:.88rem;color:#58716d}
</style>''', unsafe_allow_html=True)

st.title('MyoE')
st.caption('Lightweight cross-user sEMG recognition · recorded-signal workspace')
st.info('Recorded replay only. This research demonstration is not a medical device or live prosthetic controller.')

with st.sidebar:
    st.subheader('Recording workspace')
    input_type = st.radio('Data source', ['Synthetic example', 'Upload recordings'])
    st.caption('Signals use 16 forearm channels, sampled at 2,048 Hz, in physical mV. Each recording is exactly five seconds.')
    calibration_upload = query_upload = None
    if input_type == 'Upload recordings':
        calibration_upload = st.file_uploader('Calibration archive (.npz)', type=['npz'], key='cal')
        query_upload = st.file_uploader('Recorded signal (.npz)', type=['npz'], key='query')
        with st.expander('Required archive fields'):
            st.code('Calibration: signals (16, 10240, 16), labels (16,)\nRecording: signal (10240, 16), optional labels scalar or (39,)')
            st.caption('Labels are integer gestures 1–16. Calibration contains one recording per gesture.')
    else:
        gesture = st.selectbox('Synthetic example gesture', range(1, 17), format_func=lambda x: f'Gesture {x:02d}')
        st.warning('Synthetic recordings are fabricated to demonstrate the interface. Their metrics have no clinical or scientific meaning.')
    st.divider()
    model_upload = st.file_uploader('Import personalized model', type=['npz'], key='model')
    if model_upload and st.button('Load model', use_container_width=True):
        try:
            st.session_state.personalized = load_model(BytesIO(model_upload.getvalue()))
            st.session_state.pop('baseline', None)
            st.session_state.pop('personalization_seconds', None)
            st.success('Personalized model loaded.')
        except Exception as exc:
            st.error(f'Unable to load model: {exc}')
    if 'personalized' in st.session_state:
        stream = BytesIO()
        try:
            st.session_state.personalized.save(stream)
            st.download_button('Export personalized model', stream.getvalue(), 'myoe-personalized.npz',
                               mime='application/octet-stream', use_container_width=True)
        except Exception as exc:
            st.error(f'Unable to export model: {exc}')

try:
    source = load_pretrained()
except Exception as exc:
    source = None
    st.error(f'The pretrained checkpoint could not be loaded: {exc}')

@st.cache_data(show_spinner=False)
def example_calibration():
    return synthetic_calibration()


@st.cache_data(show_spinner=False)
def example_recording(gesture):
    return synthetic_recording(gesture, seed=29)


if input_type == 'Synthetic example':
    calibration, calibration_labels = example_calibration()
    recording = example_recording(gesture)
    truth = np.full(39, gesture, dtype=np.int64)
else:
    calibration = calibration_labels = recording = truth = None
    try:
        if calibration_upload:
            calibration, calibration_labels = load_archive(calibration_upload.getvalue(), 'calibration')
        if query_upload:
            recording, labels = load_archive(query_upload.getvalue(), 'recording')
            truth = labels_for_windows(labels)
    except (ValueError, KeyError, OSError) as exc:
        st.error(f'Invalid input archive: {exc}')

cal_col, status_col = st.columns([2, 1])
with cal_col:
    st.subheader('Personalization')
    st.write('One labeled calibration recording per gesture is used to fit a baseline head and adapt the encoder.')
    ready = source is not None and calibration is not None
    if st.button('Personalize model', disabled=not ready, type='primary'):
        try:
            progress = st.progress(0, text='Preparing calibration windows')
            windows = np.concatenate([preprocess(record) for record in calibration], axis=0)
            targets = np.repeat(calibration_labels, 39)
            st.session_state.baseline = fit_head(source, windows, targets)
            def report(step, total):
                if step == total or step % 20 == 0:
                    progress.progress(step / total, text=f'Personalizing · {step:,} / {total:,} updates')
            start = time.perf_counter()
            st.session_state.personalized = personalize(source, windows, targets, seed=2101, progress=report)
            st.session_state.personalization_seconds = time.perf_counter() - start
            progress.progress(1.0, text='Personalization complete')
        except Exception as exc:
            st.error(f'Personalization failed: {exc}')
with status_col:
    st.subheader('Model status')
    state = 'Personalized' if 'personalized' in st.session_state else 'Awaiting calibration'
    st.write(f'**{state}**')
    if 'personalization_seconds' in st.session_state:
        st.caption(f"Adaptation time {st.session_state.personalization_seconds:.1f} s · CPU")
    st.caption('The before view uses a calibration-fitted head on the frozen source encoder.')

st.divider()
st.subheader('Recorded signal')
if recording is None:
    st.info('Select a synthetic example or upload a recording to view its waveform.')
    st.stop()

selection = st.multiselect('Channels', list(range(1, 17)), default=[1, 2, 3, 4], format_func=lambda c: f'F{c}')
if selection:
    indices = np.arange(0, len(recording), 8)
    frame = pd.DataFrame({f'F{c}': recording[indices, c - 1] for c in selection}, index=indices / FS)
    frame.index.name = 'Time (s)'
    st.line_chart(frame, height=230)

model = st.session_state.get('personalized')
if model is None:
    st.info('Personalize or import a model to view gesture predictions.')
    st.stop()

try:
    start = time.perf_counter()
    result = predict_recording(model, recording)
    elapsed = (time.perf_counter() - start) * 1000
    labels = np.asarray(result['labels'])
    probabilities = np.asarray(result['probabilities'])
    before = predict_recording(st.session_state.baseline, recording) if 'baseline' in st.session_state else None
except Exception as exc:
    st.error(f'Prediction failed: {exc}')
    st.stop()

st.caption(f'Elapsed preprocessing and inference for 39 windows: {elapsed:.1f} ms ({elapsed / 39:.2f} ms/window). This live measurement includes Python overhead and is not a batch-one benchmark.')
index = st.slider('Replay position', 0, 38, 0, format='Window %d')
cols = st.columns(3)
cols[0].metric('Gesture', f'{labels[index]:02d}')
cols[1].metric('Model confidence', f'{probabilities[index].max():.1%}')
cols[2].metric('Recording time', f'{.5 + index * .1:.1f}–{.7 + index * .1:.1f} s')
prob_frame = pd.DataFrame({'Gesture': [f'{x:02d}' for x in range(1, 17)], 'Probability': probabilities[index]}).set_index('Gesture')
st.bar_chart(prob_frame, height=220)
st.line_chart(pd.DataFrame({'Personalized': labels}, index=np.round(.6 + np.arange(39) * .1, 1)), height=160)
if st.button('Play recorded predictions'):
    placeholder = st.empty()
    for i in range(39):
        placeholder.markdown(f'**{.5 + i * .1:.1f}–{.7 + i * .1:.1f} s** · Gesture **{labels[i]:02d}** · confidence **{probabilities[i].max():.1%}**')
        time.sleep(.1)

if before is not None:
    st.subheader('Before and after personalization')
    baseline_labels = np.asarray(before['labels'])
    st.write(f'Predictions differ on {np.mean(labels != baseline_labels):.1%} of recording windows.')
    st.line_chart(pd.DataFrame({'Frozen encoder': baseline_labels, 'Personalized': labels}), height=160)

if truth is not None:
    st.subheader('Recording metrics')
    current = confusion_matrix(truth, labels)
    recall_den = current.sum(axis=1)
    precision_den = current.sum(axis=0)
    denom = recall_den + precision_den
    valid = denom > 0
    f1 = np.divide(2 * np.diag(current), denom, out=np.zeros(16, dtype=float), where=valid)
    m1, m2 = st.columns(2)
    m1.metric('Window accuracy', f'{np.mean(labels == truth):.1%}')
    m2.metric('Macro-F1, observed classes', f'{np.mean(f1[recall_den > 0]):.3f}')
    st.caption('Overlapping windows are dependent. A single recording is not an evaluation of generalization.')
    st.dataframe(pd.DataFrame(current, index=[f'True {i:02d}' for i in range(1,17)], columns=[f'Pred {i:02d}' for i in range(1,17)]), use_container_width=True)
