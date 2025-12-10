import mediapipe as mp
import cv2
import onnxruntime as ort
import numpy as np
from scipy.signal import welch
from collections import deque
import json
import matplotlib.pyplot as plt

# ---------- rPPG helper functions ----------
def crop_face(frame):
    with mp.solutions.face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True) as fm:
        results = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks is None:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        xaxis = [i.x for i in landmarks if i.x > 0]
        yaxis = [i.y for i in landmarks if i.y > 0]
        xmin, xmax = min(xaxis) * frame.shape[1], max(xaxis) * frame.shape[1]
        ymin, ymax = min(yaxis) * frame.shape[0], max(yaxis) * frame.shape[0]
        img = frame[round(ymin):round(ymax), round(xmin):round(xmax), ::-1].astype('float32')/255
        return cv2.resize(img, (36, 36), interpolation=cv2.INTER_AREA)

def load_state(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_model(path):
    model = ort.InferenceSession(path)
    def run(img, state, dt=1/30):
        result = model.run(None, {"arg_0.1": img[None, None], "onnx::Mul_37": [dt], **state})
        bvp, new_state = result[0][0,0], result[1:]
        return bvp, dict(zip(state, new_state))
    return run

def get_hr(y, sr=30, hr_min=30, hr_max=180):
    if len(y) < 2:
        return 0
    p, q = welch(y, sr, nfft=int(1e5 / sr), nperseg=np.min((len(y) - 1, 256)))
    mask = (p > hr_min / 60) & (p < hr_max / 60)
    if not any(mask):
        return 0
    return p[mask][np.argmax(q[mask])] * 60

# ---------- Initialize ----------
# path to model & state
MODEL_PATH = 'your_model.onnx'
STATE_PATH = 'state.json'

state = load_state(STATE_PATH)
run_model = load_model(MODEL_PATH)

cap = cv2.VideoCapture(0)
bvp_buffer = deque(maxlen=300)  # store last 10 seconds at 30 fps

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_ylim(-0.1, 0.1)
ax.set_xlim(0, 300)
ax.set_title("BVP Signal")
ax.set_ylabel("Amplitude")
ax.set_xlabel("Frames")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    face_img = crop_face(frame)
    if face_img is not None:
        bvp, state = run_model(face_img, state)
        bvp_buffer.append(bvp)
        hr = get_hr(np.array(bvp_buffer))
        
        # draw HR on frame
        cv2.putText(frame, f"HR: {int(hr)} bpm", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    
    cv2.imshow("Face + HR", frame)
    
    # Update matplotlib graph
    line.set_ydata(np.array(bvp_buffer))
    line.set_xdata(np.arange(len(bvp_buffer)))
    ax.set_xlim(0, max(50, len(bvp_buffer)))
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.close()
