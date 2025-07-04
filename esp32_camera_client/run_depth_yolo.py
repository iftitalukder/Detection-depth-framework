#!/usr/bin/env python
"""
Live depth + YOLOv7 in one window
• left  : camera RGB with detections
• right : MiDaS disparity (brighter → closer)
"""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent / "yolov7"))

import cv2
import numpy as np
import torch
from pathlib import Path
from torchvision import transforms

try:
    from yolov7.utils.plots import colors
except ImportError:
    # tiny local fallback
    def colors(i, bgr=False):
        palette = [(255,56,56),(255,157,151),(255,112,31),(255,178,29),
                   (207,210,49),(72,249,10),(0,226,252),(0,149,255),
                   (0,51,255),(77,85,255),(168,76,51),(246,51,102)]
        r,g,b = palette[i % len(palette)]
        return (b,g,r) if bgr else (r,g,b)

# ───────────────────────────── settings ──────────────────────────────
YOLO_WTS      = Path("yolov7/yolov7.pt")      # put the .pt here (or change path)
YOLO_IMG_SIZE = 640                           # inference resolution
CONF_THRES    = 0.25                          # YOLO object conf
IOU_THRES     = 0.45                          # YOLO NMS IoU
DEPTH_WARN    = 0.60                          # Δ→ warn threshold (MiDaS_small, 0-1 scale)
DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ─────────────────────────────────────────────────────────────────────

print(f"[INFO] using device: {DEVICE}")

# 1️⃣  MiDaS -- model + transform
print("[INFO] loading MiDaS_small from torch.hub …")
midas           = torch.hub.load("intel-isl/MiDaS", "MiDaS_small",  pretrained=True).to(DEVICE).eval()
midas_transforms= torch.hub.load("intel-isl/MiDaS", "transforms")
midas_tf        = midas_transforms.small_transform

# 2️⃣  YOLOv7
print("[INFO] loading YOLOv7 weights …")
# ---  if torch.load pickle-safety bites again, patch one line in yolov7/models/experimental.py
#      ckpt = torch.load(w, map_location=map_location, weights_only=False)  ← add weights_only=False
from yolov7.models.experimental import attempt_load
from yolov7.utils.general      import non_max_suppression, scale_coords
#from yolov7.utils.plots        import colors
from yolov7.utils.datasets  import letterbox          # << correct module

yolo        = attempt_load(str(YOLO_WTS), map_location=DEVICE)  # ckpt
yolo_names  = yolo.names if hasattr(yolo, "names") else [str(i) for i in range(1000)]
yolo_stride = int(yolo.stride.max())

# 3️⃣  webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
assert cap.isOpened(), "❌  Cannot open camera"

# ─────────────────────────── helper funcs ────────────────────────────
def infer_depth(rgb):
    """rgb (H,W,3, uint8) → disparity (H,W, float32, 0-1)"""
    inp = midas_tf(rgb).to(DEVICE)          # no extra unsqueeze
    with torch.no_grad():
        disp = midas(inp)
        disp = torch.nn.functional.interpolate(
            disp.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()
    disp = (disp - disp.min()) / max(disp.max() - disp.min(), 1e-6)
    return disp.astype("float32")


def infer_yolo(frame):
    """BGR → detections (after NMS, on original image scale)"""
    img0 = frame.copy()
    img  = letterbox(img0, YOLO_IMG_SIZE, stride=yolo_stride, auto=True)[0]
    img  = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img  = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img  = torch.from_numpy(img).to(DEVICE).unsqueeze(0)
    with torch.no_grad():
        pred = yolo(img)[0]
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES)[0]
    if pred is None:
        return []
    pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], img0.shape).round()
    return pred.cpu().numpy()

# ───────────────────────────── main loop ─────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    disp      = infer_depth(frame_rgb)                         # (H,W)
    detections= infer_yolo(frame)                              # (n,6)

    # annotate detections + compute “too close”
    for *xyxy, conf, cls in detections:
        x1, y1, x2, y2 = map(int, xyxy)
        box_disp = disp[y1:y2, x1:x2]
        if box_disp.size == 0:
            continue
        med = float(np.median(box_disp))

        c = int(cls)
        label = f"{yolo_names[c]} {conf:.2f}"
        if med > DEPTH_WARN:           # MiDaS: larger value ≈ closer
            label += "Stop! Object is TOO CLOSE "
            clr = (0, 0, 255)          # red
        else:
            clr = colors(c, True)      # color per class

        cv2.rectangle(frame, (x1, y1), (x2, y2), clr, 2)
        tf = cv2.FONT_HERSHEY_SIMPLEX
        w, h = cv2.getTextSize(label, tf, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - 18), (x1 + w, y1), clr, -1)
        cv2.putText(frame, label, (x1, y1 - 4), tf, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # depth visualisation
    disp_vis = (disp * 255).astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, cv2.COLORMAP_MAGMA)

    both = np.hstack((frame, disp_vis))
    cv2.imshow("YOLO  +  MiDaS depth", both)

    if cv2.waitKey(1) & 0xFF == 27:    # ESC
        break

cap.release()
cv2.destroyAllWindows()
