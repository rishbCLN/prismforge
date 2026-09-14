"""VigiAI Structured Detection Export Utility.
Serializes perception frame detections to standardized JSON and JSON Lines formats.
"""
from typing import List, Dict, Any, Optional
import os
import json
from src.perception.yolo import Detection
from src.ingestion.video import VideoMetadata


def export_detections_to_json(
    detections_by_frame: List[Dict[str, Any]],
    output_path: str,
    metadata: Optional[VideoMetadata] = None,
    clip_id: Optional[str] = None
) -> str:
    """Exports structured frame detections and video metadata to a JSON file.

    Args:
        detections_by_frame: List of dicts, each with {'frame_id', 'timestamp', 'detections'}.
        output_path: Target JSON file path.
        metadata: Optional VideoMetadata object.
        clip_id: Optional identifier for the clip.

    Returns:
        Absolute path to the exported JSON file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    total_detections = sum(len(f.get("detections", [])) for f in detections_by_frame)
    doc: Dict[str, Any] = {
        "clip_id": clip_id or (os.path.splitext(os.path.basename(output_path))[0]),
        "total_frames": len(detections_by_frame),
        "total_detections": total_detections,
        "metadata": metadata.to_dict() if metadata else None,
        "frames": detections_by_frame
    }

    temp_path = output_path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(temp_path, output_path)

    return os.path.abspath(output_path)


def load_detections_from_json(json_path: str) -> Dict[str, Any]:
    """Loads exported detections JSON document.

    Args:
        json_path: Path to the detections JSON file.

    Returns:
        Parsed dictionary containing metadata and frames.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Detections file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)
