import cv2
import time
import argparse
import numpy as np
from pathlib import Path
import sys
import os

from src.preprocessing.preprocessor import FramePreprocessor

def main(video_path):
    print(f"Initializing Gridlock 10-Stage Pipeline Simulation...")
    preprocessor = FramePreprocessor(config_path="configs/config.yaml")
    
    Path("results").mkdir(exist_ok=True)
    Path("assets").mkdir(exist_ok=True)
    
    out_path = "results/demo_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    latencies = []
    
    if not os.path.exists(video_path):
        print(f"Video {video_path} not found. Running synthetic smoke test...")
        out = cv2.VideoWriter(out_path, fourcc, 5.0, (1280, 720))
        for i in range(10):
            frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            start_t = time.time()
            processed_frame, meta = preprocessor.process_frame(frame, i)
            latency = (time.time() - start_t) * 1000
            latencies.append(latency)
            print(f"Frame {i} processed | Latency: {latency:.2f}ms | Brightness: {meta['mean_brightness']:.1f}")
            if out.isOpened():
                out.write(cv2.resize(processed_frame, (1280, 720)))
    else:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 5.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        frame_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            start_t = time.time()
            processed_frame, meta = preprocessor.process_frame(frame, frame_id)
            latency = (time.time() - start_t) * 1000
            latencies.append(latency)
            
            print(f"Frame {frame_id} | Processing latency: {latency:.2f}ms")
            if out.isOpened():
                out.write(cv2.resize(processed_frame, (w, h)))
            frame_id += 1
            if frame_id > 50: break
            
        cap.release()
    if out is not None:
        out.release()
    print(f"Demo run complete. Avg inference latency: {np.mean(latencies) if latencies else 0.0:.2f}ms")
    print(f"Video saved to {out_path}")

def main_webcam(camera_id=0):
    print(f"Opening webcam (device {camera_id})...")
    preprocessor = FramePreprocessor(config_path="configs/config.yaml")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open webcam {camera_id}")
        return

    Path("results").mkdir(exist_ok=True)
    out_path = "results/webcam_demo.mp4"
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    latencies = []
    frame_id = 0

    print("Press 'q' to quit. Processing webcam feed...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        start_t = time.time()
        processed_frame, meta = preprocessor.process_frame(frame, frame_id)
        latency = (time.time() - start_t) * 1000
        latencies.append(latency)
        print(f"Frame {frame_id} | Latency: {latency:.2f}ms")
        if out.isOpened():
            out.write(cv2.resize(processed_frame, (w, h)))
        cv2.imshow("Gridlock AI - Webcam Demo", processed_frame)
        frame_id += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Webcam demo complete. Avg latency: {np.mean(latencies):.2f}ms")
    print(f"Video saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default=None, help="Path to video file")
    parser.add_argument("--camera", type=int, default=None, help="Webcam device ID (e.g. 0)")
    args = parser.parse_args()
    if args.camera is not None:
        main_webcam(args.camera)
    elif args.video:
        main(args.video)
    else:
        main("assets/sample_traffic.mp4")
