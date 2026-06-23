import random
import shutil
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

from src.preprocessing.label_harmonizer import harmonize_yolo_datasets, normalize_label


def prepare_yolo_dataset(task: str, extract_zips: bool = True, strict: bool = False, config_path="configs/label_map.yaml"):
    """Normalize one of the YOLO-style tasks into a single training directory."""
    return harmonize_yolo_datasets(
        task=task,
        config_path=config_path,
        strict=strict,
        extract_zips=extract_zips,
    )


def _find_plate_archive(source_roots):
    keywords = {"plate", "license", "licence", "car"}
    for source_root in source_roots:
        root = Path(source_root)
        if not root.exists():
            continue
        for archive in root.rglob("*.zip"):
            text = normalize_label(str(archive))
            if all(k not in text for k in {"plate", "license", "licence"}):
                continue
            if "car" not in text:
                continue
            return archive
    return None


def _extract_plate_archive(archive: Path, extract_root: Path) -> Path:
    target = extract_root / archive.stem
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target)
    return target


def _locate_plate_image(dataset_root: Path, xml_root: ET.Element) -> Path | None:
    filename = xml_root.findtext("filename")
    candidates = []
    if filename:
        candidates.append(dataset_root / "images" / Path(filename).name)
        candidates.append(dataset_root / Path(filename).name)

    stem = Path(filename).stem if filename else None
    if stem:
        candidates.extend(
            [
                dataset_root / "images" / f"{stem}.png",
                dataset_root / "images" / f"{stem}.jpg",
                dataset_root / "images" / f"{stem}.jpeg",
                dataset_root / "images" / f"{stem}.JPG",
                dataset_root / "images" / f"{stem}.PNG",
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    images_dir = dataset_root / "images"
    if images_dir.exists() and stem:
        matches = list(images_dir.glob(f"{stem}.*"))
        if matches:
            return matches[0]

    return None


def _parse_voc_boxes(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing size metadata in {xml_path}")

    width = float(size.findtext("width"))
    height = float(size.findtext("height"))
    boxes = []

    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        xmin = float(bbox.findtext("xmin"))
        ymin = float(bbox.findtext("ymin"))
        xmax = float(bbox.findtext("xmax"))
        ymax = float(bbox.findtext("ymax"))

        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_w = (xmax - xmin) / width
        box_h = (ymax - ymin) / height
        boxes.append((0, x_center, y_center, box_w, box_h))

    return boxes


def _write_yolo_label(label_path: Path, boxes):
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        for cls_id, x_center, y_center, box_w, box_h in boxes:
            f.write(
                f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n"
            )


def _split_samples(samples, train_ratio=0.8, val_ratio=0.1, seed=42):
    samples = list(samples)
    random.Random(seed).shuffle(samples)

    total = len(samples)
    if total == 0:
        return [], [], []

    train_end = max(1, int(total * train_ratio))
    val_end = max(train_end + 1 if total > 2 else train_end, int(total * (train_ratio + val_ratio)))
    val_end = min(val_end, total)

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    if not val_samples and len(train_samples) > 1:
        val_samples = [train_samples.pop()]
    if not test_samples and len(train_samples) > 2:
        test_samples = [train_samples.pop()]

    return train_samples, val_samples, test_samples


def prepare_plate_dataset(
    source_roots=None,
    output_dir="data/processed/license_plate",
    extract_root="data/raw/extracted/license_plate",
    seed=42,
):
    """Convert the VOC-style car plate archive into a YOLO dataset."""
    source_roots = [Path(p) for p in (source_roots or ["data/datasets", "data/raw"])]
    output_dir = Path(output_dir)
    extract_root = Path(extract_root)

    archive = _find_plate_archive(source_roots)
    if archive is None:
        raise FileNotFoundError(
            "Could not find the license plate archive under the configured source roots."
        )

    dataset_root = _extract_plate_archive(archive, extract_root)
    annotations_dir = dataset_root / "annotations"
    images_dir = dataset_root / "images"
    if not annotations_dir.exists() or not images_dir.exists():
        raise FileNotFoundError(
            f"Expected 'annotations' and 'images' folders in {dataset_root}, but they were not found."
        )

    samples = []
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        image_path = _locate_plate_image(dataset_root, ET.parse(xml_path).getroot())
        if image_path is None:
            continue
        boxes = _parse_voc_boxes(xml_path)
        if not boxes:
            continue
        samples.append({"xml": xml_path, "image": image_path, "boxes": boxes})

    if not samples:
        raise FileNotFoundError(f"No usable plate annotations were found in {annotations_dir}.")

    train_samples, val_samples, test_samples = _split_samples(samples, seed=seed)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    def _materialize(split_name, split_samples):
        images_out = output_dir / "images" / split_name
        labels_out = output_dir / "labels" / split_name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        for sample in split_samples:
            rel_name = f"plate_dataset/{sample['image'].name}"
            dst_image = images_out / rel_name
            dst_label = labels_out / Path(rel_name).with_suffix(".txt")
            dst_image.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample["image"], dst_image)
            _write_yolo_label(dst_label, sample["boxes"])

    _materialize("train", train_samples)
    _materialize("val", val_samples)
    _materialize("test", test_samples)

    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "license_plate"},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    return output_dir / "data.yaml"
