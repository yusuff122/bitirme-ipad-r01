from pathlib import Path
import cv2

def read_video(path):
    cap = cv2.VideoCapture(str(path)); frames = []
    if not cap.isOpened(): raise OSError(f'Cannot open video: {path}')
    fps = cap.get(cv2.CAP_PROP_FPS); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ok, frame = cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, {'fps': fps, 'frame_count': count, 'decoded_frames': len(frames)}

def make_clips(frames, length=16, stride=4):
    return [frames[i:i+length] for i in range(0, max(0, len(frames)-length+1), stride)]
