"""
Standalone Utility: Publish Live Stream Metadata to Firebase Realtime Database
Path: /live_stream/<device_id>
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    FIREBASE_DATABASE_URL,
    STREAM_DEVICE_ID,
    HLS_STREAM_URL,
    STREAM_FPS,
    STREAM_WIDTH,
    STREAM_HEIGHT
)
from firebase.realtime_db import RealtimeDatabaseManager
from utils.logger import logger

def publish_metadata(
    stream_url: str = HLS_STREAM_URL,
    device_id: str = STREAM_DEVICE_ID,
    status: str = "live",
    fps: int = STREAM_FPS,
    resolution: str = f"{STREAM_WIDTH}x{STREAM_HEIGHT}"
):
    rtdb = RealtimeDatabaseManager()
    
    timestamp = int(time.time() * 1000)
    payload = {
        "stream_url": stream_url,
        "stream_status": status,
        "fps": fps,
        "resolution": resolution,
        "last_active": timestamp
    }
    
    logger.info("=" * 60)
    logger.info(" Publishing Live Stream Metadata to Firebase Realtime Database")
    logger.info("=" * 60)
    logger.info(f"Target Path : /live_stream/{device_id}")
    logger.info(f"Payload     :\n{json.dumps(payload, indent=2)}")
    logger.info("-" * 60)
    
    success = rtdb.update_live_stream_metadata(device_id, payload)
    if success:
        logger.info(f" Successfully published live stream metadata to /live_stream/{device_id}!")
    else:
        logger.error(f" Failed to publish live stream metadata to /live_stream/{device_id}.")
        
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish Live Stream Metadata to Firebase RTDB")
    parser.add_argument("--url", type=str, default=HLS_STREAM_URL, help="HLS / Video Stream URL (e.g., https://<vps>/live/agroeye_01/index.m3u8)")
    parser.add_argument("--device", type=str, default=STREAM_DEVICE_ID, help="Device ID (e.g. pi_agroeye_01)")
    parser.add_argument("--status", type=str, default="live", choices=["live", "standby", "offline"], help="Stream status")
    parser.add_argument("--fps", type=int, default=STREAM_FPS, help="Stream FPS")
    parser.add_argument("--resolution", type=str, default=f"{STREAM_WIDTH}x{STREAM_HEIGHT}", help="Resolution (e.g. 1280x720 or 640x480)")
    
    args = parser.parse_args()
    publish_metadata(
        stream_url=args.url,
        device_id=args.device,
        status=args.status,
        fps=args.fps,
        resolution=args.resolution
    )
