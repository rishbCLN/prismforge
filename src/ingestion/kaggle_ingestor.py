"""Kaggle Dataset Ingestion Module.
Uses the user's Kaggle API Token (KGAT) to search, download, and extract construction safety datasets.
"""
import os
import sys
import pathlib
import zipfile
from typing import List, Dict, Any, Optional

DEFAULT_KAGGLE_TOKEN = "KGAT_30f03f4d2199abd465d1842def6b1201"


def configure_kaggle_auth(token: Optional[str] = None):
    """Ensures Kaggle API authentication credentials are set up."""
    tok = token or os.environ.get("KAGGLE_API_TOKEN") or DEFAULT_KAGGLE_TOKEN
    os.environ["KAGGLE_API_TOKEN"] = tok

    # Ensure ~/.kaggle/access_token exists
    try:
        home = pathlib.Path.home()
        kdir = home / ".kaggle"
        kdir.mkdir(exist_ok=True)
        tfile = kdir / "access_token"
        if not tfile.exists() or tfile.read_text().strip() != tok:
            tfile.write_text(f"{tok}\n")
    except Exception as e:
        print(f"[Warning] Could not write ~/.kaggle/access_token: {e}")


# Initialize auth upon import
configure_kaggle_auth()


class KaggleIngestor:
    """Manages searching, downloading, and unpacking datasets from Kaggle."""

    CURATED_DATASETS = [
        {
            "slug": "mrifatrashid/construction-site-safety-video",
            "title": "Construction Site Safety Video Footage",
            "description": "Real-world construction site video footage showing heavy machinery, active workers, and safety gear.",
            "type": "video",
            "size": "~43 MB",
            "recommended": True
        },
        {
            "slug": "snehilsanyal/construction-site-safety-image-dataset-roboflow",
            "title": "Construction Site Safety Image Dataset (Roboflow)",
            "description": "Annotated construction images identifying hardhats, safety vests, machinery, and workers.",
            "type": "image_annotated",
            "size": "~216 MB",
            "recommended": True
        },
        {
            "slug": "ndomalau/personal-protective-equipment-ppe-dataset",
            "title": "Personal Protective Equipment (PPE) Dataset",
            "description": "High-resolution imagery of industrial and construction workers with PPE compliance/violations.",
            "type": "image",
            "size": "~249 MB",
            "recommended": False
        },
        {
            "slug": "adilshamim8/safety-helmet-detection-for-construction-sites",
            "title": "Safety Helmet Detection for Construction Sites",
            "description": "Targeted dataset focusing on helmet vs no-helmet detection on active construction personnel.",
            "type": "image",
            "size": "~1.1 GB",
            "recommended": False
        }
    ]

    def __init__(self, token: Optional[str] = None):
        self.token = token or DEFAULT_KAGGLE_TOKEN
        configure_kaggle_auth(self.token)
        self._api = None

    def _get_api(self):
        if self._api is None:
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                api = KaggleApi()
                api.authenticate()
                self._api = api
            except Exception as e:
                raise RuntimeError(f"Failed to authenticate with Kaggle API: {e}")
        return self._api

    def check_connection(self) -> Dict[str, Any]:
        """Checks if the Kaggle API token is active and valid."""
        try:
            api = self._get_api()
            res = api.dataset_list(search="construction safety", page=1)
            count = len(res) if res else 0
            return {
                "status": "connected",
                "token_prefix": self.token[:12] + "...",
                "sample_results_count": count
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "token_prefix": self.token[:12] + "..." if self.token else "None"
            }

    def search_datasets(self, query: str = "construction site safety", max_results: int = 10) -> List[Dict[str, Any]]:
        """Searches Kaggle for safety datasets matching query."""
        api = self._get_api()
        raw_results = api.dataset_list(search=query, page=1) or []
        datasets = []
        for r in raw_results[:max_results]:
            total_bytes = getattr(r, "total_bytes", 0) or 0
            size_mb = f"{total_bytes / (1024 * 1024):.1f} MB" if total_bytes else "Unknown"
            datasets.append({
                "slug": getattr(r, "ref", str(r)),
                "title": getattr(r, "title", ""),
                "size_bytes": total_bytes,
                "size_human": size_mb,
                "downloads": getattr(r, "download_count", 0),
                "votes": getattr(r, "vote_count", 0),
                "last_updated": str(getattr(r, "last_updated", ""))
            })
        return datasets

    def download_dataset(self, dataset_slug: str, download_dir: str = "data/ingested/kaggle") -> str:
        """Downloads and extracts a Kaggle dataset into download_dir/<slug>."""
        api = self._get_api()
        safe_slug = dataset_slug.replace("/", "_")
        target_path = os.path.join(download_dir, safe_slug)
        os.makedirs(target_path, exist_ok=True)

        print(f"Downloading Kaggle dataset: {dataset_slug} to {target_path}...")
        api.dataset_download_files(dataset_slug, path=target_path, unzip=True, quiet=False)
        print(f"Download and extraction completed: {target_path}")
        return target_path
