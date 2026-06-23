import argparse
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import yaml


SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "valid", "validation"),
    "test": ("test",),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class LabelHarmonizationError(ValueError):
    pass


def normalize_label(label):
    text = str(label).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def load_label_config(config_path="configs/label_map.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_alias_lookup(task_config):
    lookup = {}
    canonical_names = task_config["canonical_names"]

    for canonical in canonical_names:
        lookup[normalize_label(canonical)] = canonical

    for canonical, aliases in task_config.get("aliases", {}).items():
        if canonical not in canonical_names:
            raise LabelHarmonizationError(
                f"Alias group '{canonical}' is not listed in canonical_names."
            )
        for alias in aliases:
            lookup[normalize_label(alias)] = canonical

    return lookup


def parse_yolo_names(data):
    names = data.get("names", {})
    if isinstance(names, list):
        return {idx: name for idx, name in enumerate(names)}
    return {int(idx): name for idx, name in names.items()}


def resolve_split_path(dataset_root, data_yaml, split):
    raw_value = data_yaml.get(split)
    if raw_value is None:
        return None

    split_path = Path(raw_value)
    if split_path.is_absolute():
        return split_path

    direct = (dataset_root / split_path).resolve()
    if direct.exists():
        return direct

    without_parent_refs = Path(*[part for part in split_path.parts if part != ".."])
    repaired = (dataset_root / without_parent_refs).resolve()
    if repaired.exists():
        return repaired

    return direct


def find_existing_split_dir(dataset_root, data_yaml, canonical_split):
    for split_key in SPLIT_ALIASES[canonical_split]:
        split_dir = resolve_split_path(dataset_root, data_yaml, split_key)
        if split_dir and split_dir.exists():
            return split_dir

    for split_key in SPLIT_ALIASES[canonical_split]:
        candidate = dataset_root / "images" / split_key
        if candidate.exists():
            return candidate.resolve()

    return None


def label_path_for_image(image_path, image_split_dir, dataset_root):
    rel_image = image_path.relative_to(image_split_dir)

    if image_split_dir.name.lower() in {"train", "val", "valid", "validation", "test"}:
        labels_split_dir = image_split_dir.parent.parent / "labels" / image_split_dir.name
        candidate = labels_split_dir / rel_image.with_suffix(".txt")
        if candidate.exists():
            return candidate

    try:
        rel_from_root = image_path.relative_to(dataset_root)
        parts = list(rel_from_root.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            candidate = dataset_root / Path(*parts).with_suffix(".txt")
            if candidate.exists():
                return candidate
    except ValueError:
        pass

    candidate = image_path.with_suffix(".txt")
    if candidate.exists():
        return candidate

    return None


def path_matches_keywords(path, keywords):
    if not keywords:
        return True
    text = normalize_label(path)
    return any(normalize_label(keyword) in text for keyword in keywords)


def discover_dataset_yamls(source_roots, output_dir, include_keywords=None):
    output_dir = output_dir.resolve()
    yamls = []

    for source_root in source_roots:
        root = Path(source_root)
        if not root.exists():
            continue
        for candidate in list(root.rglob("*.yaml")) + list(root.rglob("*.yml")):
            if not path_matches_keywords(str(candidate), include_keywords):
                continue
            if output_dir in candidate.resolve().parents:
                continue
            with open(candidate, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if "names" in data and any(key in data for key in ("train", "val", "valid", "test")):
                yamls.append(candidate)

    return sorted(set(yamls))


def extract_archives(source_roots, extract_dir, include_keywords=None):
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted_roots = []

    for source_root in source_roots:
        root = Path(source_root)
        if not root.exists():
            continue
        for archive in root.rglob("*.zip"):
            if not path_matches_keywords(str(archive), include_keywords):
                continue
            target = extract_dir / archive.stem
            if not target.exists():
                print(f"Extracting {archive} -> {target}")
                with zipfile.ZipFile(archive, "r") as zf:
                    zf.extractall(target)
            extracted_roots.append(target)

    return extracted_roots


def remap_label_file(src_label, dst_label, class_id_map, counters, strict=True):
    dst_label.parent.mkdir(parents=True, exist_ok=True)

    remapped_lines = []
    if src_label is not None and src_label.exists():
        with open(src_label, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                parts = stripped.split()
                try:
                    old_class_id = int(float(parts[0]))
                except ValueError as exc:
                    raise LabelHarmonizationError(
                        f"Invalid YOLO class id in {src_label}:{line_no}: {parts[0]}"
                    ) from exc

                if old_class_id not in class_id_map:
                    message = f"Class id {old_class_id} in {src_label} is not present in data.yaml names."
                    if strict:
                        raise LabelHarmonizationError(message)
                    continue

                new_class_id, canonical_name = class_id_map[old_class_id]
                remapped_lines.append(" ".join([str(new_class_id), *parts[1:]]))
                counters[canonical_name] += 1

    with open(dst_label, "w", encoding="utf-8") as f:
        if remapped_lines:
            f.write("\n".join(remapped_lines) + "\n")


def copy_split(dataset_root, data_yaml, split, output_dir, class_id_map, counters, strict=True):
    image_split_dir = find_existing_split_dir(dataset_root, data_yaml, split)
    if image_split_dir is None:
        return 0

    copied = 0
    out_images = output_dir / "images" / split
    out_labels = output_dir / "labels" / split
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for image_path in image_split_dir.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        dataset_key = normalize_label(dataset_root.name)
        rel_image = image_path.relative_to(image_split_dir)
        safe_rel = Path(dataset_key) / rel_image
        dst_image = out_images / safe_rel
        dst_label = out_labels / safe_rel.with_suffix(".txt")

        src_label = label_path_for_image(image_path, image_split_dir, dataset_root)
        dst_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, dst_image)
        remap_label_file(src_label, dst_label, class_id_map, counters, strict=strict)
        copied += 1

    return copied


def build_dataset_class_map(dataset_yaml_path, canonical_names, alias_lookup, strict=True):
    with open(dataset_yaml_path, "r", encoding="utf-8") as f:
        data_yaml = yaml.safe_load(f) or {}

    dataset_names = parse_yolo_names(data_yaml)
    class_id_map = {}
    unknown = []

    for old_id, raw_name in dataset_names.items():
        normalized = normalize_label(raw_name)
        canonical = alias_lookup.get(normalized)
        if canonical is None:
            unknown.append(str(raw_name))
            continue
        class_id_map[old_id] = (canonical_names.index(canonical), canonical)

    if unknown and strict:
        raise LabelHarmonizationError(
            f"{dataset_yaml_path} has unmapped labels: {', '.join(sorted(unknown))}. "
            "Add them to configs/label_map.yaml aliases before training."
        )

    return data_yaml, class_id_map


def write_data_yaml(output_dir, canonical_names):
    data_yaml_path = output_dir / "data.yaml"
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(canonical_names)},
    }

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return data_yaml_path


def harmonize_yolo_datasets(
    task,
    config_path="configs/label_map.yaml",
    source_roots=None,
    output_dir=None,
    strict=True,
    extract_zips=False,
):
    config = load_label_config(config_path)
    task_config = config["tasks"][task]
    canonical_names = task_config["canonical_names"]
    alias_lookup = build_alias_lookup(task_config)
    include_keywords = task_config.get("include_keywords", [])

    source_roots = [Path(p) for p in (source_roots or task_config.get("sources", []))]
    output_dir = Path(output_dir or task_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if extract_zips:
        extracted = extract_archives(source_roots, Path("data/raw/extracted") / task, include_keywords)
        source_roots = [*source_roots, *extracted]

    dataset_yamls = discover_dataset_yamls(source_roots, output_dir, include_keywords)
    if not dataset_yamls and not extract_zips:
        extracted = extract_archives(source_roots, Path("data/raw/extracted") / task, include_keywords)
        if extracted:
            source_roots = [*source_roots, *extracted]
            dataset_yamls = discover_dataset_yamls(source_roots, output_dir, include_keywords)
    if not dataset_yamls:
        raise FileNotFoundError(
            f"No YOLO data.yaml files found for task '{task}'. "
            "Extract datasets under data/raw or data/datasets, or pass --extract-zips."
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counters = Counter()
    image_counts = Counter()

    for dataset_yaml_path in dataset_yamls:
        data_yaml, class_id_map = build_dataset_class_map(
            dataset_yaml_path, canonical_names, alias_lookup, strict=strict
        )
        raw_dataset_root = data_yaml.get("path")
        if raw_dataset_root:
            dataset_root = Path(raw_dataset_root)
            if not dataset_root.is_absolute():
                dataset_root = (dataset_yaml_path.parent / dataset_root).resolve()
        else:
            dataset_root = dataset_yaml_path.parent.resolve()

        for split in SPLIT_ALIASES:
            image_counts[split] += copy_split(
                dataset_root,
                data_yaml,
                split,
                output_dir,
                class_id_map,
                counters,
                strict=strict,
            )

    data_yaml_path = write_data_yaml(output_dir, canonical_names)
    print(f"Normalized {sum(image_counts.values())} images into {output_dir}")
    print(f"Images by split: {dict(image_counts)}")
    print(f"Labels by class: {dict(counters)}")
    print(f"Training YAML: {data_yaml_path}")
    return data_yaml_path


def get_normalized_data_yaml(task, config_path="configs/label_map.yaml", prepare=True):
    config = load_label_config(config_path)
    output_dir = Path(config["tasks"][task]["output_dir"])
    data_yaml_path = output_dir / "data.yaml"
    if data_yaml_path.exists():
        return data_yaml_path
    if not prepare:
        return None
    return harmonize_yolo_datasets(task=task, config_path=config_path)


def main():
    parser = argparse.ArgumentParser(description="Normalize mixed YOLO dataset labels before training.")
    parser.add_argument("--task", required=True, choices=["vehicle", "helmet", "license_plate", "triple_riding"])
    parser.add_argument("--config", default="configs/label_map.yaml")
    parser.add_argument("--source", action="append", dest="sources", help="Dataset root to scan. Can be repeated.")
    parser.add_argument("--output-dir")
    parser.add_argument("--extract-zips", action="store_true", help="Extract zip datasets under data/raw/extracted first.")
    parser.add_argument("--allow-unmapped", action="store_true", help="Skip unknown labels instead of failing.")
    args = parser.parse_args()

    harmonize_yolo_datasets(
        task=args.task,
        config_path=args.config,
        source_roots=args.sources,
        output_dir=args.output_dir,
        strict=not args.allow_unmapped,
        extract_zips=args.extract_zips,
    )


if __name__ == "__main__":
    main()
