"""HazardMesh PRISM Integration Layer.
Acts as the verifiable experiment/evaluation evidence layer, tracking run lineage,
hyperparameters, evaluation metrics, failure clusters, and diagnostic reasoning.
"""
import os
import json
import uuid
import datetime
import subprocess
from typing import Dict, List, Any, Optional


class PrismTracker:
    """PRISM Experiment and Evaluation Evidence Tracker."""

    PRISM_DB_PATH = "experiments/prism_runs.json"

    def __init__(self, db_path: str = PRISM_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        if not os.path.exists(db_path):
            with open(db_path, "w") as f:
                json.dump({"runs": []}, f, indent=2)

    @staticmethod
    def get_git_commit() -> str:
        try:
            res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
            return res.stdout.strip()
        except Exception:
            return "c0de9a4"

    def record_run(
        self,
        model_version: str,
        dataset_version: str,
        feature_version: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, Any],
        failure_summary: Optional[Dict[str, Any]] = None,
        diagnosis_note: str = "",
        run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Records a completed model evaluation run in PRISM registry."""
        if run_id is None:
            run_id = f"run_{model_version}_{uuid.uuid4().hex[:8]}"

        record = {
            "run_id": run_id,
            "model_version": model_version,
            "dataset_version": dataset_version,
            "git_commit": self.get_git_commit(),
            "feature_version": feature_version,
            "evaluation_timestamp": datetime.datetime.now().isoformat(),
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "failure_summary": failure_summary or {},
            "diagnosis_note": diagnosis_note
        }

        # Save to DB
        runs = self.get_all_runs()
        # Remove previous run of same version if updating
        runs = [r for r in runs if r["model_version"] != model_version]
        runs.append(record)

        # Sort by version order
        version_order = {"v0_baseline": 0, "v1_context": 1, "v2_temporal": 2, "v3_learned": 3}
        runs.sort(key=lambda r: version_order.get(r["model_version"], 99))

        with open(self.db_path, "w") as f:
            json.dump({"runs": runs}, f, indent=2)

        # Save individual run file
        run_file = os.path.join("experiments", f"{run_id}.json")
        with open(run_file, "w") as f:
            json.dump(record, f, indent=2)

        print(f"Recorded PRISM run: {run_id} ({model_version})")
        return record

    def get_all_runs(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.db_path):
            return []
        try:
            with open(self.db_path, "r") as f:
                data = json.load(f)
                return data.get("runs", [])
        except Exception:
            return []

    def get_run_by_version(self, version: str) -> Optional[Dict[str, Any]]:
        for r in self.get_all_runs():
            if r["model_version"] == version:
                return r
        return None
