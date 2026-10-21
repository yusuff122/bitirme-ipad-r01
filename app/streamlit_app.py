import streamlit as st
import cv2
import tempfile
from pathlib import Path

st.set_page_config(page_title='IPAD R01 Anomaly Detection',layout='wide')
st.title('Industrial Conveyor Video Anomaly Detection')
st.caption('IPAD R01 · Video anomaly detection research prototype')
uploaded=st.file_uploader('Upload a video',type=['mp4','avi','mov','mkv'])
if uploaded:
    suffix=Path(uploaded.name).suffix or '.mp4'
    with tempfile.NamedTemporaryFile(suffix=suffix,delete=False) as tmp: tmp.write(uploaded.getbuffer()); path=tmp.name
    cap=cv2.VideoCapture(path)
    if not cap.isOpened(): st.error('Could not open this video.')
    else:
        fps=cap.get(cv2.CAP_PROP_FPS); count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); duration=count/fps if fps else 0
        col1,col2=st.columns([2,1]); col1.video(uploaded)
        col2.metric('FPS',f'{fps:.2f}' if fps else 'Unknown'); col2.metric('Frames',count); col2.metric('Resolution',f'{w} × {h}'); col2.metric('Duration',f'{duration:.2f} s')
        cap.release()
        st.info('No trained model run is configured yet. Anomaly timeline will appear after model inference is connected.')
else:
    st.info('Upload a video to inspect its preview and metadata. The application works without a dataset or trained model.')
