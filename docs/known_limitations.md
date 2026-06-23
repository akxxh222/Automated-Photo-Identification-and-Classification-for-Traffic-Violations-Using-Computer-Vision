# Known Limitations & Honest Assessment

This document transparently documents known gaps, trade-offs, and areas for future improvement. Every limitation is acknowledged with its impact and planned resolution.

---

## 1. Seatbelt Detector — No Dedicated Training Data

**Issue**: The seatbelt detector uses a heuristic approach (edge detection along the diagonal shoulder-to-hip path via MediaPipe pose landmarks) rather than a trained model. It works on clear driver-facing views but has limited accuracy on side-angle or occluded views.

**Root Cause**: No labeled seatbelt dataset was available among the 6 provided datasets. Seatbelt detection requires fine-grained in-car pose annotations with seatbelt visibility labels.

**Current Approach**: Edge response along the seatbelt diagonal path. If no strong diagonal edge is detected (indicating a strap), a violation is reported. Falls back to Hough line transform when MediaPipe is unavailable.

**Resolution Path**: Collect or source a labeled seatbelt dataset (e.g., SVIRO or DriveSeg) and train a dedicated YOLOv8 classifier on seatbelt ROIs. Priority: Medium.

---

## 2. Rider & Pillion Classes — Very Few Training Samples

**Issue**: The vehicle detector defines 8 classes, but classes 6 (`rider`) and 7 (`pillion`) have extremely limited training data:

| Class | Annotations | Quality |
|-------|------------|---------|
| car (0) | 643 | ✅ Hand-labeled |
| truck (1) | 116 | ✅ Hand-labeled |
| bus (2) | 130 | ✅ Hand-labeled |
| two_wheeler (3) | 272 | ✅ Hand-labeled |
| three_wheeler (4) | 410 | ✅ Hand-labeled |
| pedestrian (5) | 10,074 | ✅ Hand-labeled |
| **rider (6)** | **9** | ⚠️ Auto-generated |
| **pillion (7)** | **124** | ⚠️ Auto-generated |

**Impact**: Helmet and triple-riding detectors depend on rider/pillion detections. With only 9 rider and 124 pillion samples, the model may fail to generalize for these classes in unseen scenarios.

**Mitigation**: `scripts/generate_missing_classes.py` bootstraps labels via COCO-pretrained YOLOv8n person detection + vehicle bbox overlap analysis. This provides initial training signal but is noisy compared to hand-labeled data.

**Resolution Path**: Source a labeled two-wheeler occupant dataset (e.g., Indian driving dataset with rider/pillion annotations) and retrain. Priority: High.

---

## 3. Vehicle Detector mAP50 — 58.4% (Room for Improvement)

**Issue**: The unified 8-class YOLOv8m detector achieves 58.4% mAP50, which is lower than the individually trained detectors.

**Root Cause**: Training a single model across 8 diverse classes with imbalanced annotation counts (643 car vs 9 rider) forces the model to allocate capacity across all classes. The extreme class imbalance hurts overall performance.

**Comparison**: Triple riding detector (3 balanced classes, 7.9K images) achieves **93.9% mAP50** — demonstrating that the same architecture performs well with balanced data.

**Resolution Path**: (a) Class-balanced sampling during training, (b) more data for under-represented classes, (c) separate detectors for vehicle type vs occupant detection. Priority: Medium.

---

## 4. Wrong-Side Detector — Single Reference Direction

**Issue**: The wrong-side detector learns the dominant traffic flow direction from optical flow statistics. This works for single-direction roadways but fails at intersections, multi-lane roads, or when traffic is stopped.

**Current Approach**: Accumulates Farneback optical flow vectors over a sliding window (deque, maxlen=10 frames). Computes mean dominant vector and flags vehicles with cosine similarity below threshold.

**Resolution Path**: Add lane-level direction analysis, intersection-specific logic, and configurable direction masks per camera zone. Priority: Medium.

---

## 5. Red-Light Detector — Fixed ROI, No Traffic Light Detection Model

**Issue**: The red-light detector uses a hardcoded HSV ROI at `(550, 10, 630, 80)` for traffic light detection. This only works for that specific camera position.

**Current Approach**: HSV color segmentation in a fixed pixel region + debounce state machine (3 consecutive red frames to trigger). Stop-line crossing verification via polygon containment.

**Resolution Path**: (a) Configurable traffic light ROIs per camera via JSON zones, (b) dedicated traffic light detection model (YOLOv8 trained on traffic light dataset), (c) V2I integration with signal controllers. Priority: High.

---

## 6. Single-Camera Tracking — No Multi-Camera ReID

**Issue**: Tracking is per-camera only. A vehicle that appears in one camera's view and reappears in another gets a new ID. No vehicle re-identification across junctions.

**Current Approach**: ByteTrack with motion-based tracking within a single camera's field of view.

**Resolution Path**: Implement appearance-based ReID using a lightweight embedding model (e.g., OSNet or BoT) and a vector database (FAISS) for cross-camera matching. Priority: Low (Phase 2).

---

## 7. MediaPipe API Compatibility

**Issue**: MediaPipe 0.10.x changed its API, removing `mp.solutions` in favor of a task-based API (`PoseLandmarker` with model asset files). The seatbelt detector's `mp.solutions.pose.Pose()` import fails on newer versions.

**Current Approach**: Graceful fallback to Hough line transform edge detection when the pose model can't be loaded. Logs a warning and continues with reduced accuracy.

**Resolution Path**: Migrate to the new MediaPipe task API or pin `mediapipe<0.10.10`. Priority: Low.

---

## 8. Security — Basic API Key Auth Only

**Issue**: Authentication is via a static API key in the request header (`X-API-Key`). No rate limiting, IP allowlisting, key rotation, or OAuth. The `dev-key-123` fallback is used when `API_KEYS` env var is not set.

**Resolution Path**: (a) Set `API_KEYS` in production, (b) implement JWT/OAuth2 authentication, (c) add rate limiting middleware, (d) key rotation with expiry. Priority: Medium.

---

## 9. No CI/CD or Automated Testing Pipeline

**Issue**: Tests are minimal (13 unit tests, no integration/E2E tests). No CI/CD configuration exists. Model training, evaluation, and deployment are manual processes.

**Resolution Path**: (a) Expand test coverage to integration tests, (b) add GitHub Actions for CI, (c) automated model evaluation on PRs, (d) Docker registry deployment on merge to main. Priority: Medium.

---

## 10. Evidence Storage — Local Filesystem

**Issue**: Evidence artifacts (annotated frames, PDF challans) are stored on the local filesystem in `evidence_store/`. This doesn't scale for production with multiple cameras.

**Resolution Path**: Migrate to object storage (AWS S3, GCS, or MinIO) with signed URL access. Priority: Low (Phase 2).

---

## Summary

| Limitation | Impact | Priority | Effort |
|------------|--------|----------|--------|
| Seatbelt detector (heuristic) | Lower accuracy on side views | Medium | 2 weeks |
| Rider/pillion low samples (9/124) | Poor generalization | **High** | 4 weeks |
| Vehicle mAP50 at 58.4% | Lower overall accuracy | Medium | 3 weeks |
| Single-camera tracking | No multi-junction tracking | Low | 6 weeks |
| Fixed red-light ROI | Single camera only | **High** | 2 weeks |
| Basic API auth | Security risk | Medium | 1 week |
| No CI/CD | Manual deployment only | Medium | 1 week |

This project is a **working prototype** suitable for hackathon evaluation and pilot demonstrations. It demonstrates the full architecture, label harmonization, concurrent detection pipeline, and end-to-end enforcement workflow. Closing these gaps would require additional labeled data, production engineering, and security hardening.
