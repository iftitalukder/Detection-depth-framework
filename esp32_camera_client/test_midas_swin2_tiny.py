# test_yolo_midas.py

import cv2
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import sys,  pathlib
repo_dir = pathlib.Path(__file__).resolve().parent / "yolov7"   # <-- repo folder beside this script
sys.path.insert(0, str(repo_dir))   
import models.yolo          # repo_dir/models/yolo.py
from models.experimental import attempt_load

# --------------------------------------------------------------------
# 1) Load MiDaS (we'll use the small model for speed)
# --------------------------------------------------------------------
print("Loading MiDaS_small from torch.hub…")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
midas.to("cpu").eval()
# prepare transforms
midas_transform = Compose([
    Resize(256),                # smaller for speed
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std =[0.229, 0.224, 0.225]),
])

# --------------------------------------------------------------------
# 2) Monkey-patch torch.load so YOLOv7 can load its custom Model
# --------------------------------------------------------------------
# This forces torch.load(..., weights_only=False) under the hood,
# allowing unpickling of models.yolo.Model.
orig_torch_load = torch.load
def torch_load_allow_model(path, *args, **kwargs):
    # ensure weights_only=False so custom globals are allowed
    kwargs.setdefault("weights_only", False)
    return orig_torch_load(path, *args, **kwargs)
torch.load = torch_load_allow_model

# allowlist the YOLOv7 Model class

import models.yolo  # noqa: F401  (ensures models.yolo.Model is imported)
torch.serialization.add_safe_globals([models.yolo.Model])

# --------------------------------------------------------------------
# 3) Load YOLOv7
# --------------------------------------------------------------------
from yolov7.models.experimental import attempt_load

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading YOLOv7…")
yolo = attempt_load("yolov7/yolov7.pt", map_location=DEVICE)
yolo.to(DEVICE).eval()

# --------------------------------------------------------------------
# 4) Inference loop
# --------------------------------------------------------------------
cap = cv2.VideoCapture(0)
CLOSE_THRESH = 1.0  # tune this: objects with depth < this are "too close"

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # ——— 4a) MiDaS depth
    input_batch = midas_transform(img).unsqueeze(0)
    with torch.no_grad():
        depth_pred = midas(input_batch)
    depth_map = depth_pred.squeeze().cpu().numpy()
    # normalize for display
    dmin, dmax = depth_map.min(), depth_map.max()
    depth_vis = ((depth_map - dmin) / (dmax - dmin) * 255).astype(np.uint8)

    # ——— 4b) YOLOv7 detection
    img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(DEVICE) / 255.0
    with torch.no_grad():
        preds = yolo(img_tensor)[0]

    # NMS
    preds = preds.cpu()
    # xyxy, conf, cls
    boxes = preds[:, :4]
    scores = preds[:, 4]
    classes = preds[:, 5].int()

    # draw
    out = frame_bgr.copy()
    for box, score, cls in zip(boxes, scores, classes):
        if score < 0.3:
            continue
        x1, y1, x2, y2 = map(int, box.tolist())
        # clamp
        x1, y1 = max(0,x1), max(0,y1)
        x2, y2 = min(depth_map.shape[1]-1, x2), min(depth_map.shape[0]-1, y2)

        # median depth in box
        med = np.median(depth_map[y1:y2, x1:x2])

        color = (0,255,0)
        label = f"{med:.2f}m"
        if med < CLOSE_THRESH:
            color = (0,0,255)
            label += " TOO CLOSE"

        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # show
    cv2.imshow("RGB + YOLOv7+MiDaS", out)
    cv2.imshow("depth", depth_vis)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
