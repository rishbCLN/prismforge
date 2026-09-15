"""Large Vehicle Dataset Ingestor.

Supports two dataset formats:
  1. large_vehicle/ — Pascal VOC XML annotations (tractor, truck)
     Structure: Annotations/Annotations/{tractor,truck}/*.xml
                Tractor/*.jpg, Truck/*.jpg
  2. large_dataset/ — YOLO-format txt labels (person=class 0 used only)
     Structure: images/*.jpg, labels/*.txt

Outputs a unified manifest dict:
  {
    "total_images": int,
    "vehicle_images": int,
    "entries": [
      {
        "image_path": str,
        "image_width": int,
        "image_height": int,
        "vehicles": [{"class": "tractor"|"truck", "bbox_norm": [cx, cy, w, h]}],
        "persons":  [{"bbox_norm": [cx, cy, w, h]}],
        "dataset_source": "large_vehicle"|"large_dataset",
      }, ...
    ]
  }
"""

import os
import json
import glob
import math
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VOC XML helpers
# ---------------------------------------------------------------------------
try:
    import xml.etree.ElementTree as ET
    _ET_AVAILABLE = True
except ImportError:
    _ET_AVAILABLE = False


def _parse_voc_xml(xml_path: str) -> Optional[Dict[str, Any]]:
    """Parse Pascal VOC XML annotation file.

    Returns dict with keys: filename, width, height, objects.
    objects is a list of {name, xmin, ymin, xmax, ymax}.
    Returns None if parse fails.
    """
    if not _ET_AVAILABLE:
        logger.error("xml.etree.ElementTree not available")
        return None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        logger.warning(f"Failed to parse VOC XML {xml_path}: {e}")
        return None

    size = root.find("size")
    width = int(float(size.findtext("width", "0") or "0")) if size is not None else 0
    height = int(float(size.findtext("height", "0") or "0")) if size is not None else 0
    filename = root.findtext("filename", os.path.basename(xml_path).replace(".xml", ".jpg"))

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").lower().strip()
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))
        except (ValueError, TypeError):
            continue
        objects.append({
            "name": name,
            "xmin": xmin, "ymin": ymin,
            "xmax": xmax, "ymax": ymax,
        })

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "objects": objects,
    }


def _abs_to_norm_yolo(xmin: float, ymin: float, xmax: float, ymax: float,
                      img_w: int, img_h: int) -> List[float]:
    """Convert absolute (xmin, ymin, xmax, ymax) to YOLO [cx, cy, w, h] normalized."""
    if img_w <= 0 or img_h <= 0:
        return [0.5, 0.5, 0.0, 0.0]
    cx = ((xmin + xmax) / 2.0) / img_w
    cy = ((ymin + ymax) / 2.0) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return [
        max(0.0, min(1.0, cx)),
        max(0.0, min(1.0, cy)),
        max(0.0, min(1.0, w)),
        max(0.0, min(1.0, h)),
    ]


# ---------------------------------------------------------------------------
# Main Ingestor
# ---------------------------------------------------------------------------

class LargeVehicleIngestor:
    """Ingests large_vehicle (VOC XML) and optionally large_dataset (YOLO).

    Args:
        large_vehicle_dir: Path to datasets/large_vehicle/
        large_dataset_dir: Optional path to datasets/large_dataset/
        use_large_dataset: Whether to also ingest YOLO-format large_dataset (default True)
    """

    VEHICLE_CLASSES = ("tractor", "truck", "crane", "excavator", "bulldozer",
                       "forklift", "vehicle", "large vehicle")

    def __init__(
        self,
        large_vehicle_dir: str = "datasets/large_vehicle",
        large_dataset_dir: Optional[str] = "datasets/large_dataset",
        use_large_dataset: bool = True,
    ):
        self.large_vehicle_dir = large_vehicle_dir
        self.large_dataset_dir = large_dataset_dir
        self.use_large_dataset = use_large_dataset

    # ------------------------------------------------------------------
    # large_vehicle/ — Pascal VOC
    # ------------------------------------------------------------------

    def _ingest_voc_split(self, vehicle_class: str) -> List[Dict[str, Any]]:
        """Ingest one class subfolder (tractor or truck) from VOC XML dataset."""
        ann_dir = os.path.join(
            self.large_vehicle_dir, "Annotations", "Annotations", vehicle_class
        )
        img_dir = os.path.join(
            self.large_vehicle_dir, vehicle_class.capitalize()
        )
        # fallback: truck folder may be named "Truck" exactly
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(self.large_vehicle_dir, vehicle_class.title())

        if not os.path.isdir(ann_dir):
            logger.warning(f"Annotation dir not found: {ann_dir}")
            return []

        xml_files = glob.glob(os.path.join(ann_dir, "*.xml"))
        entries = []

        for xml_path in xml_files:
            parsed = _parse_voc_xml(xml_path)
            if parsed is None:
                continue

            filename = parsed["filename"]
            img_path = os.path.join(img_dir, filename)
            if not os.path.isfile(img_path):
                # Try without leading path
                img_path = os.path.join(img_dir, os.path.basename(filename))

            img_w = parsed["width"]
            img_h = parsed["height"]

            vehicles = []
            persons = []
            for obj in parsed["objects"]:
                name = obj["name"]
                if name in self.VEHICLE_CLASSES or name == vehicle_class:
                    norm = _abs_to_norm_yolo(
                        obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"],
                        img_w, img_h
                    )
                    vehicles.append({"class": vehicle_class, "bbox_norm": norm})
                elif name in ("person", "worker", "human"):
                    norm = _abs_to_norm_yolo(
                        obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"],
                        img_w, img_h
                    )
                    persons.append({"bbox_norm": norm})

            # Every VOC image has at least the vehicle we know about
            # If the XML didn't yield any vehicles via object tags, add one from filename context
            if not vehicles and parsed["objects"]:
                # All objects are the known vehicle class
                for obj in parsed["objects"]:
                    norm = _abs_to_norm_yolo(
                        obj["xmin"], obj["ymin"], obj["xmax"], obj["ymax"],
                        img_w, img_h
                    )
                    vehicles.append({"class": vehicle_class, "bbox_norm": norm})

            entries.append({
                "image_path": img_path,
                "image_width": img_w,
                "image_height": img_h,
                "vehicles": vehicles,
                "persons": persons,
                "dataset_source": "large_vehicle",
            })

        logger.info(f"  VOC {vehicle_class}: {len(entries)} images")
        return entries

    def ingest_large_vehicle(self) -> List[Dict[str, Any]]:
        """Ingest all classes from large_vehicle/ VOC dataset."""
        if not os.path.isdir(self.large_vehicle_dir):
            logger.warning(f"large_vehicle dir not found: {self.large_vehicle_dir}")
            return []

        entries = []
        for cls in ("tractor", "truck"):
            entries.extend(self._ingest_voc_split(cls))
        return entries

    # ------------------------------------------------------------------
    # large_dataset/ — YOLO format
    # ------------------------------------------------------------------

    def ingest_large_dataset(self) -> List[Dict[str, Any]]:
        """Ingest large_dataset/ YOLO-format labels.
        Only uses person annotations (class 0) for worker position context.
        """
        if not self.use_large_dataset or not self.large_dataset_dir:
            return []
        if not os.path.isdir(self.large_dataset_dir):
            logger.warning(f"large_dataset dir not found: {self.large_dataset_dir}")
            return []

        img_dir = os.path.join(self.large_dataset_dir, "images")
        lbl_dir = os.path.join(self.large_dataset_dir, "labels")

        if not os.path.isdir(lbl_dir):
            logger.warning(f"large_dataset labels dir not found: {lbl_dir}")
            return []

        txt_files = glob.glob(os.path.join(lbl_dir, "*.txt"))
        entries = []

        for txt_path in txt_files:
            basename = os.path.splitext(os.path.basename(txt_path))[0]
            # Try common image extensions
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".JPG"):
                candidate = os.path.join(img_dir, basename + ext)
                if os.path.isfile(candidate):
                    img_path = candidate
                    break

            persons = []
            try:
                with open(txt_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id = int(parts[0])
                        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        # class 0 = person in most construction YOLO datasets
                        if cls_id == 0:
                            persons.append({"bbox_norm": [cx, cy, w, h]})
            except Exception as e:
                logger.debug(f"Failed to read {txt_path}: {e}")
                continue

            # Only include entries with at least one person annotation
            if persons:
                entries.append({
                    "image_path": img_path or "",
                    "image_width": 0,   # unknown without loading image
                    "image_height": 0,
                    "vehicles": [],     # no vehicle labels in this dataset
                    "persons": persons,
                    "dataset_source": "large_dataset",
                })

        logger.info(f"  large_dataset (persons only): {len(entries)} images with persons")
        return entries

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def ingest_all(self) -> Dict[str, Any]:
        """Ingest both datasets and return unified manifest."""
        print("=" * 60)
        print("  LARGE VEHICLE DATASET INGESTOR")
        print("=" * 60)

        voc_entries = self.ingest_large_vehicle()
        yolo_entries = self.ingest_large_dataset()

        all_entries = voc_entries + yolo_entries

        # Stats
        total_vehicles = sum(len(e["vehicles"]) for e in all_entries)
        total_persons = sum(len(e["persons"]) for e in all_entries)
        vehicle_images = sum(1 for e in all_entries if e["vehicles"])

        class_dist: Dict[str, int] = {}
        for e in all_entries:
            for v in e["vehicles"]:
                c = v["class"]
                class_dist[c] = class_dist.get(c, 0) + 1

        manifest = {
            "total_images": len(all_entries),
            "vehicle_images": vehicle_images,
            "total_vehicle_annotations": total_vehicles,
            "total_person_annotations": total_persons,
            "vehicle_class_distribution": class_dist,
            "entries": all_entries,
        }

        print(f"Total images:      {len(all_entries)}")
        print(f"Images w/ vehicle: {vehicle_images}")
        print(f"Vehicle annotations: {total_vehicles}")
        print(f"Person annotations:  {total_persons}")
        print(f"Vehicle classes:   {class_dist}")

        return manifest
