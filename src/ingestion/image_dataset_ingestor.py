"""Hardcore Universal Image Dataset Ingestion Engine.
Scans the `datasets/` root directory and automatically parses subfolders containing
construction safety image datasets in YOLO, COCO, Pascal VOC, or flat image formats.
"""
import os
import glob
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from PIL import Image


@dataclass
class BoundingBox:
    class_name: str
    class_id: int
    # Normalized coordinates [0.0, 1.0]
    x1: float
    y1: float
    x2: float
    y2: float
    # Absolute pixel coordinates
    abs_x1: int
    abs_y1: int
    abs_x2: int
    abs_y2: int
    confidence: float = 1.0


@dataclass
class IngestedImage:
    dataset_name: str
    format: str
    image_id: str
    image_path: str
    width: int
    height: int
    channels: int
    objects: List[BoundingBox]


class ImageDatasetIngestor:
    """Universal parser and validator for multi-format construction image datasets."""

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    # Canonical construction safety class normalization map
    CLASS_CANONICAL_MAP = {
        # Worker / Person
        "person": "person",
        "worker": "person",
        "human": "person",
        "workers": "person",
        "pedestrian": "person",
        "0": "person",

        # PPE: Helmet / Hardhat
        "helmet": "helmet",
        "hard-hat": "helmet",
        "hardhat": "helmet",
        "safety_helmet": "helmet",
        "white-helmet": "helmet",
        "yellow-helmet": "helmet",
        "blue-helmet": "helmet",
        "red-helmet": "helmet",

        # PPE: Missing Helmet
        "no-helmet": "no_helmet",
        "no_helmet": "no_helmet",
        "no-hardhat": "no_helmet",
        "head": "no_helmet",
        "without-helmet": "no_helmet",

        # PPE: Vest
        "vest": "vest",
        "safety-vest": "vest",
        "safety_vest": "vest",
        "hi-vis": "vest",
        "reflective-vest": "vest",

        # PPE: Missing Vest
        "no-vest": "no_vest",
        "no_vest": "no_vest",
        "without-vest": "no_vest",

        # Heavy Machinery / Construction Vehicles
        "machinery": "machinery",
        "excavator": "machinery",
        "crane": "machinery",
        "bulldozer": "machinery",
        "truck": "machinery",
        "dump-truck": "machinery",
        "forklift": "machinery",
        "loader": "machinery",
        "wheel-loader": "machinery",
        "roller": "machinery",
        "vehicle": "machinery",
        "heavy-equipment": "machinery"
    }

    def __init__(self, datasets_root: str = "datasets", output_manifest_dir: str = "data/ingested"):
        self.datasets_root = datasets_root
        self.output_manifest_dir = output_manifest_dir
        os.makedirs(self.output_manifest_dir, exist_ok=True)

    def scan_datasets_directory(self) -> List[str]:
        """Finds all subdirectories within datasets_root."""
        if not os.path.exists(self.datasets_root):
            return []
        subdirs = []
        for entry in os.listdir(self.datasets_root):
            p = os.path.join(self.datasets_root, entry)
            if os.path.isdir(p) and not entry.startswith("."):
                subdirs.append(p)
        return sorted(subdirs)

    def detect_format(self, dataset_path: str) -> str:
        """Auto-detects format: 'yolo', 'coco', 'voc', or 'flat'."""
        # 1. COCO: Look for annotations json
        for root, _, files in os.walk(dataset_path):
            for f in files:
                if f.endswith(".json") and any(k in f.lower() for k in ["annotation", "coco", "instances", "train", "val"]):
                    try:
                        with open(os.path.join(root, f), "r", encoding="utf-8") as jf:
                            d = json.load(jf)
                        if isinstance(d, dict) and "annotations" in d and "images" in d:
                            return "coco"
                    except Exception:
                        pass

        # 2. Pascal VOC: Look for XMLs
        xml_files = glob.glob(os.path.join(dataset_path, "**", "*.xml"), recursive=True)
        if xml_files:
            return "voc"

        # 3. YOLO: Look for label txt files or yaml
        txt_files = glob.glob(os.path.join(dataset_path, "**", "*.txt"), recursive=True)
        # Filter out classes.txt or license files
        label_txts = [t for t in txt_files if not any(x in os.path.basename(t).lower() for x in ["classes", "readme", "license"])]
        yaml_files = glob.glob(os.path.join(dataset_path, "**", "*.yaml"), recursive=True) + glob.glob(os.path.join(dataset_path, "**", "*.yml"), recursive=True)
        if yaml_files or (label_txts and any("labels" in t for t in label_txts)):
            return "yolo"

        # 4. Flat images fallback
        img_count = self._count_images(dataset_path)
        if img_count > 0:
            return "flat"

        return "unknown"

    def _count_images(self, path: str) -> int:
        count = 0
        for root, _, files in os.walk(path):
            for f in files:
                if os.path.splitext(f.lower())[1] in self.IMAGE_EXTENSIONS:
                    count += 1
        return count

    def canonical_class(self, raw_name: str) -> str:
        """Maps any arbitrary dataset class name to canonical HazardMesh class."""
        clean = raw_name.strip().lower().replace(" ", "_").replace("-", "_")
        return self.CLASS_CANONICAL_MAP.get(clean, clean)

    def ingest_dataset(self, dataset_path: str) -> List[IngestedImage]:
        """Parses a dataset directory into a list of standardized IngestedImage objects."""
        ds_name = os.path.basename(os.path.normpath(dataset_path))
        fmt = self.detect_format(dataset_path)
        print(f"Ingesting [{ds_name}] | Detected format: {fmt.upper()}")

        if fmt == "yolo":
            return self._parse_yolo(dataset_path, ds_name)
        elif fmt == "coco":
            return self._parse_coco(dataset_path, ds_name)
        elif fmt == "voc":
            return self._parse_voc(dataset_path, ds_name)
        elif fmt == "flat":
            return self._parse_flat(dataset_path, ds_name)
        else:
            print(f"Warning: Unknown or empty dataset at {dataset_path}")
            return []

    def _get_image_dimensions(self, image_path: str) -> Optional[Tuple[int, int, int]]:
        """Safely reads image dimensions without loading entire image into memory."""
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                channels = len(img.getbands())
                return w, h, channels
        except Exception:
            return None

    # --------------------------------------------------------------------------
    # Format Parsers
    # --------------------------------------------------------------------------

    def _parse_yolo(self, dataset_path: str, ds_name: str) -> List[IngestedImage]:
        """Parses YOLO normalized bbox format."""
        # Find class map
        classes = self._load_yolo_classes(dataset_path)
        images = []

        # Find all images
        for root, _, files in os.walk(dataset_path):
            for f in files:
                ext = os.path.splitext(f.lower())[1]
                if ext in self.IMAGE_EXTENSIONS:
                    img_path = os.path.join(root, f)
                    dims = self._get_image_dimensions(img_path)
                    if not dims:
                        continue
                    w, h, ch = dims
                    stem = os.path.splitext(f)[0]

                    # Look for corresponding .txt label
                    label_path = self._find_yolo_label(root, stem)
                    boxes = []
                    if label_path and os.path.exists(label_path):
                        with open(label_path, "r", encoding="utf-8") as lf:
                            for line in lf:
                                parts = line.strip().split()
                                if len(parts) >= 5:
                                    cid = int(parts[0])
                                    cx, cy, bw, bh = map(float, parts[1:5])
                                    x1 = max(0.0, cx - bw / 2.0)
                                    y1 = max(0.0, cy - bh / 2.0)
                                    x2 = min(1.0, cx + bw / 2.0)
                                    y2 = min(1.0, cy + bh / 2.0)

                                    raw_name = classes.get(cid, str(cid))
                                    c_name = self.canonical_class(raw_name)

                                    boxes.append(BoundingBox(
                                        class_name=c_name,
                                        class_id=cid,
                                        x1=x1, y1=y1, x2=x2, y2=y2,
                                        abs_x1=int(x1 * w), abs_y1=int(y1 * h),
                                        abs_x2=int(x2 * w), abs_y2=int(y2 * h),
                                        confidence=1.0
                                    ))

                    images.append(IngestedImage(
                        dataset_name=ds_name,
                        format="yolo",
                        image_id=stem,
                        image_path=img_path,
                        width=w, height=h, channels=ch,
                        objects=boxes
                    ))
        return images

    def _load_yolo_classes(self, dataset_path: str) -> Dict[int, str]:
        # 1. Try finding data.yaml
        for root, _, files in os.walk(dataset_path):
            for f in files:
                if f.lower().endswith(".yaml") or f.lower().endswith(".yml"):
                    try:
                        with open(os.path.join(root, f), "r", encoding="utf-8") as yf:
                            content = yf.read()
                        import re
                        m = re.search(r"names:\s*\[(.*?)\]", content, re.DOTALL)
                        if m:
                            raw_items = [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
                            return {i: name for i, name in enumerate(raw_items)}
                        dict_matches = re.findall(r"(\d+):\s*['\"]?([a-zA-Z0-9_\-]+)['\"]?", content)
                        if dict_matches:
                            return {int(k): v for k, v in dict_matches}
                    except Exception:
                        pass

        # 2. Try finding classes.txt or labels.txt
        for root, _, files in os.walk(dataset_path):
            for f in files:
                if f.lower() in ["classes.txt", "labels.txt"]:
                    try:
                        with open(os.path.join(root, f), "r", encoding="utf-8") as cf:
                            lines = [l.strip() for l in cf if l.strip()]
                            return {i: name for i, name in enumerate(lines)}
                    except Exception:
                        pass

        # 3. Standard fallback for Roboflow Construction PPE datasets (e.g. ppe1)
        return {
            0: "helmet",
            1: "mask",
            2: "no_helmet",
            3: "no_mask",
            4: "no_vest",
            5: "vest",
            6: "person",
            7: "safety_cone",
            8: "machinery",
            9: "vehicle",
            10: "worker"
        }

    def _find_yolo_label(self, img_dir: str, stem: str) -> Optional[str]:
        # Same dir
        same_dir = os.path.join(img_dir, f"{stem}.txt")
        if os.path.exists(same_dir):
            return same_dir
        # Swap /images/ for /labels/
        labels_dir = img_dir.replace("images", "labels").replace("Images", "Labels")
        candidate = os.path.join(labels_dir, f"{stem}.txt")
        if os.path.exists(candidate):
            return candidate
        return None

    def _parse_coco(self, dataset_path: str, ds_name: str) -> List[IngestedImage]:
        """Parses COCO format JSON annotations."""
        images = []
        # Find COCO json
        coco_json_path = None
        for root, _, files in os.walk(dataset_path):
            for f in files:
                if f.endswith(".json") and any(k in f.lower() for k in ["annotation", "coco", "instances", "train", "val"]):
                    coco_json_path = os.path.join(root, f)
                    break
            if coco_json_path:
                break

        if not coco_json_path:
            return []

        try:
            with open(coco_json_path, "r", encoding="utf-8") as jf:
                coco_data = json.load(jf)
        except Exception as e:
            print(f"Failed to read COCO json {coco_json_path}: {e}")
            return []

        cat_map = {c["id"]: self.canonical_class(c["name"]) for c in coco_data.get("categories", [])}

        # Map annotations to image_id
        ann_by_img = {}
        for ann in coco_data.get("annotations", []):
            iid = ann["image_id"]
            ann_by_img.setdefault(iid, []).append(ann)

        base_img_dir = os.path.dirname(coco_json_path)

        for img_info in coco_data.get("images", []):
            iid = img_info["id"]
            file_name = img_info["file_name"]
            # Search image on disk
            img_path = os.path.join(base_img_dir, file_name)
            if not os.path.exists(img_path):
                # Search recursively
                matches = glob.glob(os.path.join(dataset_path, "**", file_name), recursive=True)
                img_path = matches[0] if matches else None

            if not img_path or not os.path.exists(img_path):
                continue

            w = img_info.get("width")
            h = img_info.get("height")
            if not w or not h:
                dims = self._get_image_dimensions(img_path)
                if not dims:
                    continue
                w, h, _ = dims

            boxes = []
            for a in ann_by_img.get(iid, []):
                bbox = a.get("bbox", [])
                if len(bbox) == 4:
                    bx, by, bw, bh = bbox
                    x1 = max(0.0, bx / w)
                    y1 = max(0.0, by / h)
                    x2 = min(1.0, (bx + bw) / w)
                    y2 = min(1.0, (by + bh) / h)
                    cid = a.get("category_id", 0)
                    cname = cat_map.get(cid, "object")

                    boxes.append(BoundingBox(
                        class_name=cname,
                        class_id=cid,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        abs_x1=int(bx), abs_y1=int(by),
                        abs_x2=int(bx + bw), abs_y2=int(by + bh),
                        confidence=float(a.get("score", 1.0))
                    ))

            images.append(IngestedImage(
                dataset_name=ds_name,
                format="coco",
                image_id=str(iid),
                image_path=img_path,
                width=w, height=h, channels=3,
                objects=boxes
            ))
        return images

    def _parse_voc(self, dataset_path: str, ds_name: str) -> List[IngestedImage]:
        """Parses Pascal VOC XML annotations."""
        images = []
        xml_files = glob.glob(os.path.join(dataset_path, "**", "*.xml"), recursive=True)

        for xf in xml_files:
            try:
                tree = ET.parse(xf)
                root = tree.getroot()

                filename = root.findtext("filename")
                if not filename:
                    stem = os.path.splitext(os.path.basename(xf))[0]
                    filename = f"{stem}.jpg"

                # Find image
                img_path = os.path.join(os.path.dirname(xf), "..", "JPEGImages", filename)
                if not os.path.exists(img_path):
                    matches = glob.glob(os.path.join(dataset_path, "**", filename), recursive=True)
                    img_path = matches[0] if matches else None

                if not img_path or not os.path.exists(img_path):
                    continue

                size_elem = root.find("size")
                if size_elem is not None:
                    w = int(size_elem.findtext("width", 0))
                    h = int(size_elem.findtext("height", 0))
                else:
                    dims = self._get_image_dimensions(img_path)
                    if not dims:
                        continue
                    w, h, _ = dims

                boxes = []
                for obj in root.findall("object"):
                    name = obj.findtext("name", "object")
                    cname = self.canonical_class(name)
                    bnd = obj.find("bndbox")
                    if bnd is not None:
                        xmin = float(bnd.findtext("xmin", 0))
                        ymin = float(bnd.findtext("ymin", 0))
                        xmax = float(bnd.findtext("xmax", w))
                        ymax = float(bnd.findtext("ymax", h))

                        boxes.append(BoundingBox(
                            class_name=cname,
                            class_id=0,
                            x1=max(0.0, xmin / w), y1=max(0.0, ymin / h),
                            x2=min(1.0, xmax / w), y2=min(1.0, ymax / h),
                            abs_x1=int(xmin), abs_y1=int(ymin),
                            abs_x2=int(xmax), abs_y2=int(ymax),
                            confidence=1.0
                        ))

                stem = os.path.splitext(os.path.basename(xf))[0]
                images.append(IngestedImage(
                    dataset_name=ds_name,
                    format="voc",
                    image_id=stem,
                    image_path=img_path,
                    width=w, height=h, channels=3,
                    objects=boxes
                ))
            except Exception:
                continue
        return images

    def _parse_flat(self, dataset_path: str, ds_name: str) -> List[IngestedImage]:
        """Parses raw unannotated image files."""
        images = []
        for root, _, files in os.walk(dataset_path):
            for f in files:
                ext = os.path.splitext(f.lower())[1]
                if ext in self.IMAGE_EXTENSIONS:
                    img_path = os.path.join(root, f)
                    dims = self._get_image_dimensions(img_path)
                    if not dims:
                        continue
                    w, h, ch = dims
                    stem = os.path.splitext(f)[0]
                    images.append(IngestedImage(
                        dataset_name=ds_name,
                        format="flat",
                        image_id=stem,
                        image_path=img_path,
                        width=w, height=h, channels=ch,
                        objects=[]
                    ))
        return images

    def ingest_all(self) -> Dict[str, Any]:
        """Scans and parses all dataset folders inside datasets_root."""
        subdirs = self.scan_datasets_directory()
        if not subdirs:
            print(f"No dataset subdirectories found in '{self.datasets_root}'.")
            return {"status": "empty", "total_images": 0, "datasets": []}

        all_images: List[IngestedImage] = []
        dataset_summaries = []

        for sd in subdirs:
            parsed = self.ingest_dataset(sd)
            all_images.extend(parsed)

            # Class counts
            c_counts = {}
            for img in parsed:
                for obj in img.objects:
                    c_counts[obj.class_name] = c_counts.get(obj.class_name, 0) + 1

            dataset_summaries.append({
                "dataset_name": os.path.basename(sd),
                "format": self.detect_format(sd),
                "image_count": len(parsed),
                "class_counts": c_counts
            })

        # Save unified manifest
        manifest_path = os.path.join(self.output_manifest_dir, "image_manifest.json")
        summary_payload = {
            "total_images": len(all_images),
            "datasets": dataset_summaries,
            "images": [asdict(img) for img in all_images]
        }

        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(summary_payload, mf, indent=2)

        print(f"\n[Ingestion Complete] Ingested {len(all_images)} total images across {len(subdirs)} dataset(s).")
        print(f"Saved manifest: {manifest_path}")

        return summary_payload
