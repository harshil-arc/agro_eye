import os
import sys
import time
import platform
import psutil

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("=" * 60)
print(" AgroEye Diagnostic & Hardware Probe Tool")
print("=" * 60)

# --- 1. System & Memory ---
print("\n[Step 1/7] Probing System Environment & Memory...")
print(f"  OS Platform     : {platform.system()} ({platform.machine()})")
print(f"  Python Version  : {sys.version.split()[0]} ({sys.executable})")
try:
    mem = psutil.virtual_memory()
    print(f"  Total RAM       : {mem.total / (1024**2):.1f} MB")
    print(f"  Available RAM   : {mem.available / (1024**2):.1f} MB ({mem.percent}% used)")
    swap = psutil.swap_memory()
    print(f"  Swap Memory     : {swap.total / (1024**2):.1f} MB (Used: {swap.used / (1024**2):.1f} MB)")
except Exception as e:
    print(f"  Memory probe notice: {e}")

# --- 2. OpenCV GUI Probe ---
print("\n[Step 2/7] Probing OpenCV GUI & Display...")
try:
    import cv2
    import numpy as np
    print(f"  OpenCV Version  : {cv2.__version__}")
    print(f"  DISPLAY Env     : {os.environ.get('DISPLAY', 'NOT SET')}")
    print(f"  WAYLAND Env     : {os.environ.get('WAYLAND_DISPLAY', 'NOT SET')}")
    
    test_img = np.zeros((300, 500, 3), dtype=np.uint8)
    cv2.putText(test_img, "AgroEye OpenCV GUI OK", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.namedWindow("AgroEye GUI Test", cv2.WINDOW_NORMAL)
    cv2.imshow("AgroEye GUI Test", test_img)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    print("  [PASS] OpenCV GUI Window created & closed successfully.")
except Exception as e:
    print(f"  [FAIL] OpenCV GUI Window failed: {e}")

# --- 3. PyTorch Probe ---
print("\n[Step 3/7] Probing PyTorch...")
try:
    import torch
    print(f"  PyTorch Version : {torch.__version__}")
    t = torch.tensor([1.0, 2.0, 3.0])
    print(f"  Tensor Compute  : {t.sum().item()} (OK)")
    print("  [PASS] PyTorch core functional.")
except Exception as e:
    print(f"  [FAIL] PyTorch failed: {e}")

# --- 4. Camera Probe ---
print("\n[Step 4/7] Probing Video Devices / USB Camera...")
try:
    import glob
    video_nodes = glob.glob("/dev/video*")
    print(f"  Detected Nodes  : {sorted(video_nodes)}")
    
    cam_found = False
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  [PASS] Camera Index {idx}: OK (Captured frame {frame.shape[1]}x{frame.shape[0]})")
                cam_found = True
                cap.release()
                break
            cap.release()
    if not cam_found:
        print("  [WARN] No active camera stream captured on indices [0, 1, 2].")
except Exception as e:
    print(f"  [FAIL] Camera probe error: {e}")

# --- 5. Plant Pathology YOLO Model Probe ---
print("\n[Step 5/7] Probing Plant Pathology AI Model (disease_model.pt)...")
try:
    from ultralytics import YOLO
    from config import YOLO_MODEL_PATH
    print(f"  Model Path      : {YOLO_MODEL_PATH}")
    if os.path.exists(YOLO_MODEL_PATH):
        t0 = time.time()
        m = YOLO(YOLO_MODEL_PATH)
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        res = m(dummy_frame, verbose=False)
        print(f"  [PASS] Disease Model loaded & inferred in {time.time()-t0:.2f}s.")
    else:
        print(f"  [WARN] Disease model file not found at {YOLO_MODEL_PATH}")
except Exception as e:
    print(f"  [FAIL] Disease Model probe error: {e}")

# --- 6. Animal Intrusion YOLO Model Probe ---
print("\n[Step 6/7] Probing Animal Intrusion AI Model (yolo11m.pt)...")
try:
    from ultralytics import YOLO
    from config import ANIMAL_MODEL_PATH
    print(f"  Model Path      : {ANIMAL_MODEL_PATH}")
    # Candidate search
    found_path = None
    for cand in [ANIMAL_MODEL_PATH, "models/yolo11m.pt", "models/yolov8n.pt"]:
        if cand and os.path.exists(cand):
            found_path = cand
            break
    if found_path:
        t0 = time.time()
        print(f"  Loading {found_path}...")
        m_animal = YOLO(found_path)
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        res = m_animal(dummy_frame, verbose=False)
        print(f"  [PASS] Animal Model loaded & inferred in {time.time()-t0:.2f}s.")
    else:
        print(f"  [WARN] Animal model not found at candidate paths.")
except Exception as e:
    print(f"  [FAIL] Animal Model probe error: {e}")

# --- 7. Full Subsystems Integration Probe ---
print("\n[Step 7/7] Probing PlantDetectionSystem Object Instantiation...")
try:
    from main import PlantDetectionSystem
    app = PlantDetectionSystem()
    print("  [PASS] PlantDetectionSystem initialized cleanly without crashing.")
    app.stop()
except Exception as e:
    import traceback
    print(f"  [FAIL] PlantDetectionSystem initialization failed: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print(" Diagnostic Complete. If any step failed above, please share the output.")
print("=" * 60)
