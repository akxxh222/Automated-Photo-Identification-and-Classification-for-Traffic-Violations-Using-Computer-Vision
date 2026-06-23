import os
import shutil
from pathlib import Path

from ultralytics import YOLO
from PIL import Image

COCO_PERSON_ID = 0

VEHICLE_CLASS_NAMES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "two_wheeler",
    4: "three_wheeler",
    5: "pedestrian",
    6: "rider",
    7: "pillion",
}

VEHICLE_CLASS_IDS = {v: k for k, v in VEHICLE_CLASS_NAMES.items()}
TWO_WHEELER_ID = VEHICLE_CLASS_IDS["two_wheeler"]
PEDESTRIAN_ID = VEHICLE_CLASS_IDS["pedestrian"]
RIDER_ID = VEHICLE_CLASS_IDS["rider"]
PILLION_ID = VEHICLE_CLASS_IDS["pillion"]
CAR_ID = VEHICLE_CLASS_IDS["car"]
TRUCK_ID = VEHICLE_CLASS_IDS["truck"]
BUS_ID = VEHICLE_CLASS_IDS["bus"]
THREE_WHEELER_ID = VEHICLE_CLASS_IDS["three_wheeler"]

VEHICLE_IDS = {CAR_ID, TRUCK_ID, BUS_ID, TWO_WHEELER_ID, THREE_WHEELER_ID}


def xywh_to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return cx, cy, bw, bh


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    xi1 = max(ax1, bx1)
    yi1 = max(ay1, by1)
    xi2 = min(ax2, bx2)
    yi2 = min(ay2, by2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_center_y(box):
    return (box[1] + box[3]) / 2.0


def main():
    model = YOLO("models/pretrained/yolov8n.pt")

    image_dirs = [
        Path("data/processed/vehicle/images/train"),
        Path("data/processed/vehicle/images/val"),
    ]

    label_dirs = [
        Path("data/processed/vehicle/labels/train"),
        Path("data/processed/vehicle/labels/val"),
    ]

    backup_dir = Path("data/processed/vehicle/labels_backup")
    print(f"Backing up existing labels to {backup_dir}...")
    for ld in label_dirs:
        if ld.exists():
            dest = backup_dir / ld.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in ld.iterdir():
                if f.suffix == ".txt":
                    shutil.copy2(f, dest / f.name)

    total_pedestrian = 0
    total_rider = 0
    total_pillion = 0

    for img_dir, lbl_dir in zip(image_dirs, label_dirs):
        if not img_dir.exists():
            continue
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(img_dir.rglob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            rel_path = img_path.relative_to(img_dir)
            label_path = lbl_dir / rel_path.with_suffix(".txt")
            label_path.parent.mkdir(parents=True, exist_ok=True)

            existing_boxes = []
            if label_path.exists():
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cls_id = int(parts[0])
                            cx, cy, bw, bh = map(float, parts[1:])
                            existing_boxes.append((cls_id, cx, cy, bw, bh))

            try:
                img = Image.open(img_path)
                img_w, img_h = img.size
            except Exception:
                continue

            person_boxes_abs = []
            dets = model(img_path, verbose=False, conf=0.25, iou=0.5)
            if dets and len(dets[0].boxes) > 0:
                for box in dets[0].boxes:
                    if int(box.cls) == COCO_PERSON_ID:
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        person_boxes_abs.append((x1, y1, x2, y2))

            if not person_boxes_abs:
                continue

            vehicle_boxes_abs = []
            for cls_id, cx, cy, bw, bh in existing_boxes:
                if cls_id in VEHICLE_IDS:
                    x1 = (cx - bw / 2) * img_w
                    y1 = (cy - bh / 2) * img_h
                    x2 = (cx + bw / 2) * img_w
                    y2 = (cy + bh / 2) * img_h
                    vehicle_boxes_abs.append((cls_id, x1, y1, x2, y2))

            two_wheeler_boxes = [b for b in vehicle_boxes_abs if b[0] == TWO_WHEELER_ID]
            car_boxes = [b for b in vehicle_boxes_abs if b[0] == CAR_ID]
            other_vehicle_boxes = [b for b in vehicle_boxes_abs if b[0] in {TRUCK_ID, BUS_ID, THREE_WHEELER_ID}]

            new_labels = []
            assigned_persons = set()

            for pi, person_box in enumerate(person_boxes_abs):

                overlaps_tw = [(iou(person_box, vb[1:]), vb) for vb in two_wheeler_boxes]
                best_tw = max(overlaps_tw, key=lambda x: x[0]) if overlaps_tw else (0, None)

                overlaps_car = [(iou(person_box, vb[1:]), vb) for vb in car_boxes]
                best_car = max(overlaps_car, key=lambda x: x[0]) if overlaps_car else (0, None)

                overlaps_other = [(iou(person_box, vb[1:]), vb) for vb in other_vehicle_boxes]
                best_other = max(overlaps_other, key=lambda x: x[0]) if overlaps_other else (0, None)

                on_tw = best_tw[0] > 0.15
                on_car = best_car[0] > 0.15
                on_other = best_other[0] > 0.15

                if on_tw:
                    tw_box = best_tw[1][1:]
                    person_cy = box_center_y(person_box)
                    tw_cy = box_center_y(tw_box)

                    if person_cy < tw_cy - 20:
                        assigned_class = RIDER_ID
                    else:
                        pillion_of_same_tw = [
                            pj for pj, pb in enumerate(person_boxes_abs)
                            if pj != pi and pj not in assigned_persons
                            and iou(pb, tw_box) > 0.15 and box_center_y(pb) >= tw_cy - 20
                        ]
                        other_rider_ahead = any(
                            box_center_y(person_boxes_abs[pj]) < tw_cy - 20
                            for pj in pillion_of_same_tw
                        )
                        if other_rider_ahead or person_cy >= tw_cy - 20:
                            assigned_class = PILLION_ID
                        else:
                            assigned_class = RIDER_ID
                elif on_car:
                    assigned_class = PEDESTRIAN_ID
                elif on_other:
                    assigned_class = PEDESTRIAN_ID
                else:
                    assigned_class = PEDESTRIAN_ID

                cx, cy, bw, bh = xywh_to_yolo(*person_box, img_w, img_h)
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))
                new_labels.append(f"{assigned_class} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                assigned_persons.add(pi)

                if assigned_class == PEDESTRIAN_ID:
                    total_pedestrian += 1
                elif assigned_class == RIDER_ID:
                    total_rider += 1
                elif assigned_class == PILLION_ID:
                    total_pillion += 1

            if new_labels:
                with open(label_path, "a") as f:
                    for line in new_labels:
                        f.write(line + "\n")

            if assigned_persons:
                split = "val" if "val" in str(img_dir) else "train"
                print(f"  [{split}] {img_path.name}: {len(assigned_persons)} persons classified "
                      f"(PED={sum(1 for _ in range(len(person_boxes_abs)) if _ in assigned_persons)})")

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"  Pedestrian: {total_pedestrian} instances added")
    print(f"  Rider:      {total_rider} instances added")
    print(f"  Pillion:    {total_pillion} instances added")
    print(f"  Total:      {total_pedestrian + total_rider + total_pillion} new labels")
    print(f"{'='*60}")
    print(f"Backup of original labels at: {backup_dir}")
    print("Run 'python scripts/evaluate_all.py' after training to verify improvement.")


if __name__ == "__main__":
    main()
