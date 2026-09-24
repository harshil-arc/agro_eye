"""
================================================================================
PLANT DISEASE DETECTION - DIRECT WEBCAM RECOGNITION (PS 180)
================================================================================
Standalone real-time webcam recognition with YOLOv8 (17 Crop Disease Classes),
on-screen diagnosis, English/Hindi treatment recommendations, sensitivity tuning,
instant snapshot capture, and on-the-fly camera switching.

Automatically opens the external USB connected webcam (Index 1) first.
Press 'TAB' to switch between USB Webcam and Laptop Camera.

Usage:
    python webcam_recognition.py
    python webcam_recognition.py --camera-id 1
================================================================================
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(BASE_DIR / "models" / "disease_model.pt")
SNAPSHOTS_DIR = str(BASE_DIR / "data" / "snapshots")

# =====================================================================
# 17 CROP DISEASE DATABASE & TREATMENT REMEDIES
# =====================================================================
DISEASE_INFO = {
    "Apple Scab Leaf": {
        "hindi": "सेब का पपड़ी रोग (Apple Scab)",
        "cause": "Venturia inaequalis (Fungus)",
        "treatment": "Spray Captan (0.2%) or Mancozeb (0.25%). Prune infected leaves and maintain good air circulation."
    },
    "Apple rust leaf": {
        "hindi": "सेब का रतुआ रोग (Apple Rust)",
        "cause": "Gymnosporangium juniperi-virginianae (Fungus)",
        "treatment": "Apply Myclobutanil or Copper-based fungicides. Remove nearby juniper or red cedar host plants."
    },
    "Bell_pepper leaf spot": {
        "hindi": "शिमला मिर्च पत्ती धब्बा रोग (Bacterial Spot)",
        "cause": "Xanthomonas campestris (Bacteria)",
        "treatment": "Spray Copper oxychloride (2.5g/L) + Streptocycline (100 ppm). Use disease-free certified seeds."
    },
    "Corn Gray leaf spot": {
        "hindi": "मक्का ग्रे पत्ती धब्बा (Gray Leaf Spot)",
        "cause": "Cercospora zeae-maydis (Fungus)",
        "treatment": "Apply Azoxystrobin or Pyraclostrobin fungicides. Rotate crops and practice deep plowing."
    },
    "Corn leaf blight": {
        "hindi": "मक्का पत्ता झुलसा रोग (Northern Leaf Blight)",
        "cause": "Exserohilum turcicum (Fungus)",
        "treatment": "Spray Mancozeb (2.5g/L) or Propiconazole (1ml/L). Plant resistant hybrid seed varieties."
    },
    "Corn rust leaf": {
        "hindi": "मक्का रतुआ रोग (Common Rust)",
        "cause": "Puccinia sorghi (Fungus)",
        "treatment": "Apply Tebuconazole or Azoxystrobin early in the season. Ensure proper field drainage."
    },
    "Potato leaf early blight": {
        "hindi": "आलू अगेती झुलसा (Early Blight)",
        "cause": "Alternaria solani (Fungus)",
        "treatment": "Spray Chlorothalonil (2g/L) or Mancozeb (2.5g/L) at 7-10 day intervals. Avoid overhead irrigation."
    },
    "Potato leaf late blight": {
        "hindi": "आलू पछेती झुलसा (Late Blight)",
        "cause": "Phytophthora infestans (Oomycete/Fungus-like)",
        "treatment": "Spray Metalaxyl + Mancozeb (Ridomil MZ @ 2.5g/L) or Cymoxanil immediately. Destroy infected debris."
    },
    "Squash Powdery mildew leaf": {
        "hindi": "कद्दू/तोरई चूर्णिल आसिता (Powdery Mildew)",
        "cause": "Podosphaera xanthii (Fungus)",
        "treatment": "Apply wettable sulfur (2g/L), Hexaconazole (1ml/L), or Neem oil (5ml/L). Ensure adequate sunlight."
    },
    "Tomato Early blight leaf": {
        "hindi": "टमाटर अगेती झुलसा (Early Blight)",
        "cause": "Alternaria linariae (Fungus)",
        "treatment": "Apply Mancozeb or Copper fungicide. Remove lower infected leaves and mulch soil surface."
    },
    "Tomato Septoria leaf spot": {
        "hindi": "टमाटर सेप्टोरिया पत्ती धब्बा (Septoria Leaf Spot)",
        "cause": "Septoria lycopersici (Fungus)",
        "treatment": "Spray Chlorothalonil or Mancozeb every 7-10 days. Avoid splashing water on leaves during irrigation."
    },
    "Tomato leaf bacterial spot": {
        "hindi": "टमाटर जीवाणु धब्बा रोग (Bacterial Spot)",
        "cause": "Xanthomonas spp. (Bacteria)",
        "treatment": "Spray Copper hydroxide + Streptocycline. Sanitize tools and practice 2-year crop rotation."
    },
    "Tomato leaf late blight": {
        "hindi": "टमाटर पछेती झुलसा (Late Blight)",
        "cause": "Phytophthora infestans",
        "treatment": "Spray Dimethomorph or Metalaxyl + Mancozeb (2.5g/L) urgently. Remove and burn severely infected plants."
    },
    "Tomato leaf mosaic virus": {
        "hindi": "टमाटर मोज़ेक वायरस (Mosaic Virus)",
        "cause": "Tomato Mosaic Virus (ToMV)",
        "treatment": "No chemical cure. Remove and destroy infected plants immediately. Control aphids/whiteflies and disinfect tools."
    },
    "Tomato leaf yellow virus": {
        "hindi": "टमाटर पीली पत्ती मुड़न वायरस (Yellow Leaf Curl Virus)",
        "cause": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "treatment": "Control Whitefly vector using Imidacloprid (0.5ml/L) or Acetamiprid. Install yellow sticky traps."
    },
    "Tomato mold leaf": {
        "hindi": "टमाटर पत्ती फफूंद (Leaf Mold)",
        "cause": "Passalora fulva (Fungus)",
        "treatment": "Spray Copper fungicide or Difenoconazole. Increase greenhouse ventilation and lower relative humidity."
    },
    "grape leaf black rot": {
        "hindi": "अंगूर काला सड़न रोग (Black Rot)",
        "cause": "Guignardia bidwellii (Fungus)",
        "treatment": "Apply Mancozeb, Myclobutanil, or Captan before and after bloom. Prune mummified berries and old canes."
    }
}

def open_camera_stream(target_index: int):
    """
    Opens video capture for the given camera index using OS-optimal backend (DirectShow on Windows).
    """
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_V4L2, cv2.CAP_ANY]

    for backend in backends:
        try:
            cap = cv2.VideoCapture(target_index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    return cap, target_index
                cap.release()
        except Exception:
            continue

    return None, target_index

def draw_hud(frame, fps, conf_threshold, detections, model_name, camera_id):
    """Renders real-time HUD with diagnosis, remedies, and camera info."""
    h, w, _ = frame.shape

    # Top Status Banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cam_label = "USB Webcam (Cam 1)" if camera_id == 1 else f"Laptop Camera (Cam {camera_id})"
    title_text = f"Plant Disease Detection | {cam_label}"
    stats_text = f"FPS: {fps:.1f} | Conf: {int(conf_threshold*100)}% (+/-) | TAB: Switch Cam"

    cv2.putText(frame, title_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 128), 2, cv2.LINE_AA)
    cv2.putText(frame, stats_text, (w - 420, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)

    # Bottom Diagnosis & Treatment Panel
    if detections:
        panel_h = min(160, 45 + len(detections) * 35)
        overlay_bottom = frame.copy()
        cv2.rectangle(overlay_bottom, (0, h - panel_h), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay_bottom, 0.85, frame, 0.15, 0, frame)

        cv2.putText(
            frame,
            "[DIAGNOSIS & REMEDY]  (Press 'S' to Save Snapshot | 'Q' to Exit | 'TAB' to Switch Camera)",
            (15, h - panel_h + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 215, 255),
            2,
            cv2.LINE_AA
        )

        y_offset = h - panel_h + 52
        for name, data in list(detections.items())[:2]:
            info = DISEASE_INFO.get(name, {})
            treatment = info.get("treatment", "Consult local agricultural extension.")
            if len(treatment) > 85:
                treatment = treatment[:82] + "..."

            line1 = f"* {name} ({data['max_conf']*100:.0f}% conf) - Cause: {info.get('cause', 'N/A')}"
            line2 = f"  Cure: {treatment}"

            cv2.putText(frame, line1, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (50, 255, 50), 1, cv2.LINE_AA)
            cv2.putText(frame, line2, (20, y_offset + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            y_offset += 48
    else:
        guide_text = "Hold camera 10-20cm from leaf | [+/-] Sensitivity | [S] Save | [TAB] Switch Cam | [Q] Quit"
        cv2.putText(frame, guide_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    return frame

def run_webcam(model_path=None, camera_index=None, conf_threshold=0.20):
    """Directly opens webcam and runs real-time plant disease detection."""
    if model_path is None or not os.path.exists(model_path):
        model_path = DEFAULT_MODEL_PATH

    print(f"\n[INFO] Loading YOLOv8 Disease Model: '{model_path}'...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found!")

    model = YOLO(model_path)
    print(f"[SUCCESS] Model loaded with {len(model.names)} Disease Classes.")

    # Try opening external USB camera (1) first by default, then fallback to (0)
    indices_to_try = [camera_index] if camera_index is not None else [1, 0]
    
    cap = None
    active_cam_index = 0
    for idx in indices_to_try:
        print(f"[INFO] Attempting to connect to Camera Index {idx}...")
        cap, active_cam_index = open_camera_stream(idx)
        if cap is not None and cap.isOpened():
            cam_type = "USB Connected Webcam" if idx == 1 else "Laptop Integrated Camera"
            print(f"[SUCCESS] Connected to Camera Index {idx} ({cam_type})!")
            break

    if cap is None or not cap.isOpened():
        print("[ERROR] Failed to open any camera device. Please check USB webcam cable connection.")
        return

    print("\n" + "=" * 60)
    print("                LIVE WEBCAM CONTROLS")
    print("=" * 60)
    print("  'TAB'       : Switch Camera (between USB Webcam & Laptop)")
    print("  '+' or '='  : Increase sensitivity (lower threshold)")
    print("  '-' or '_'  : Decrease sensitivity (higher threshold)")
    print("  'S' or 'C'  : Save snapshot to data/snapshots/")
    print("  'Q' or ESC  : Exit camera stream")
    print("=" * 60 + "\n")

    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    cv2.namedWindow("Plant Disease Detection - PS 180", cv2.WINDOW_NORMAL)

    fps = 0.0
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            results = model.predict(frame, conf=conf_threshold, verbose=False)
            annotated_frame = results[0].plot()

            detections = {}
            if results and len(results) > 0:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names.get(cls_id, f"Class_{cls_id}")
                    if name not in detections:
                        detections[name] = {"count": 0, "max_conf": 0.0}
                    detections[name]["count"] += 1
                    if conf > detections[name]["max_conf"]:
                        detections[name]["max_conf"] = conf

            final_frame = draw_hud(annotated_frame, fps, conf_threshold, detections, model_path, active_cam_index)
            cv2.imshow("Plant Disease Detection - PS 180", final_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key == 9:  # TAB key -> switch camera
                new_index = 0 if active_cam_index == 1 else 1
                print(f"\n[INFO] Switching camera to Index {new_index}...")
                cap.release()
                new_cap, opened_idx = open_camera_stream(new_index)
                if new_cap is not None and new_cap.isOpened():
                    cap = new_cap
                    active_cam_index = opened_idx
                    cam_name = "USB Webcam" if active_cam_index == 1 else "Laptop Camera"
                    print(f"[SUCCESS] Switched to Camera Index {active_cam_index} ({cam_name})")
                else:
                    print(f"[WARNING] Could not open Camera Index {new_index}. Reverting...")
                    cap, active_cam_index = open_camera_stream(active_cam_index)
            elif key in (ord('+'), ord('=')):
                conf_threshold = max(0.05, conf_threshold - 0.05)
                print(f"[SETTING] Sensitivity increased! Conf Threshold: {conf_threshold*100:.0f}%")
            elif key in (ord('-'), ord('_')):
                conf_threshold = min(0.95, conf_threshold + 0.05)
                print(f"[SETTING] Sensitivity decreased! Conf Threshold: {conf_threshold*100:.0f}%")
            elif key in (ord('s'), ord('S'), ord('c'), ord('C')):
                timestamp = int(time.time())
                snap_path = os.path.join(SNAPSHOTS_DIR, f"diagnosis_{timestamp}.jpg")
                cv2.imwrite(snap_path, final_frame)
                print(f"\n[SNAPSHOT SAVED] Saved to: {snap_path}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera closed cleanly.")

def main():
    parser = argparse.ArgumentParser(description="Plant Disease Live Webcam Detection (PS 180)")
    parser.add_argument("--camera-id", type=int, default=1, help="Camera index (default: 1 for USB Webcam)")
    parser.add_argument("--conf", type=float, default=0.20, help="Confidence threshold (default: 0.20)")
    parser.add_argument("--model", type=str, default=None, help="Model path")
    args = parser.parse_args()

    run_webcam(model_path=args.model, camera_index=args.camera_id, conf_threshold=args.conf)

if __name__ == "__main__":
    main()
