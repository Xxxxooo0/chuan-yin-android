#!/usr/bin/env python3
"""Compare vendor DLA tensor dumps with enterprise precision vectors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare(actual: np.ndarray, expected: np.ndarray, data_range: float) -> Dict[str, Any]:
    actual64 = actual.astype(np.float64, copy=False).reshape(-1)
    expected64 = expected.astype(np.float64, copy=False).reshape(-1)
    finite = bool(np.isfinite(actual64).all())
    if not finite:
        return {
            "finite": False,
            "max_abs": None,
            "mean_abs": None,
            "rmse": None,
            "relative_l2": None,
            "cosine": None,
            "psnr": None,
        }
    diff = actual64 - expected64
    rmse = float(np.sqrt(np.mean(diff * diff)))
    expected_norm = float(np.linalg.norm(expected64))
    denominator = float(np.linalg.norm(actual64) * expected_norm)
    return {
        "finite": True,
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "rmse": rmse,
        "relative_l2": float(np.linalg.norm(diff) / max(expected_norm, 1e-12)),
        "cosine": float(np.dot(actual64, expected64) / denominator) if denominator > 0.0 else 1.0,
        "psnr": None if rmse == 0.0 else float(20.0 * math.log10(data_range / rmse)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--vendor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-abs", type=float, default=None)
    parser.add_argument("--rmse", type=float, default=None)
    args = parser.parse_args()

    baseline_dir = args.baseline_dir.resolve()
    vendor_dir = args.vendor_dir.resolve()
    manifest_path = baseline_dir / "precision_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    max_abs_limit = float(args.max_abs if args.max_abs is not None else manifest["thresholds"]["max_abs"])
    rmse_limit = float(args.rmse if args.rmse is not None else manifest["thresholds"]["rmse"])
    value_range = manifest.get("frame_value_range", [0.0, 1.0])
    data_range = float(value_range[1] - value_range[0])
    results: List[Dict[str, Any]] = []
    first_failed = None

    for stage in manifest["stages"]:
        for output in stage["expected_outputs"]:
            expected_path = baseline_dir / output["file"]
            vendor_path = vendor_dir / output["vendor_file"]
            result: Dict[str, Any] = {
                "stage": stage["id"],
                "model": stage["model"],
                "tensor": output["name"],
                "expected_file": output["file"],
                "vendor_file": output["vendor_file"],
            }
            if not expected_path.is_file():
                result.update({"passed": False, "error": "missing_baseline_output"})
            elif expected_path.stat().st_size != output["bytes"]:
                result.update({"passed": False, "error": "baseline_byte_size_mismatch"})
            elif sha256(expected_path) != output["sha256"]:
                result.update({"passed": False, "error": "baseline_sha256_mismatch"})
            elif not vendor_path.is_file():
                result.update({"passed": False, "error": "missing_vendor_output"})
            elif vendor_path.stat().st_size != output["bytes"]:
                result.update(
                    {
                        "passed": False,
                        "error": "byte_size_mismatch",
                        "expected_bytes": output["bytes"],
                        "actual_bytes": vendor_path.stat().st_size,
                    }
                )
            else:
                expected = np.fromfile(str(expected_path), dtype="<f4")
                actual = np.fromfile(str(vendor_path), dtype="<f4")
                metrics = compare(actual, expected, data_range)
                passed = bool(
                    metrics["finite"]
                    and metrics["max_abs"] <= max_abs_limit
                    and metrics["rmse"] <= rmse_limit
                )
                result.update(
                    {
                        "passed": passed,
                        "shape": output["shape"],
                        "vendor_sha256": sha256(vendor_path),
                        "metrics": metrics,
                    }
                )
            if not result["passed"] and first_failed is None:
                first_failed = {"stage": stage["id"], "tensor": output["name"]}
            results.append(result)
            metrics_text = result.get("metrics") or {}
            print(
                "{} {} passed={} max_abs={} rmse={} psnr={}".format(
                    stage["id"],
                    output["name"],
                    result["passed"],
                    metrics_text.get("max_abs"),
                    metrics_text.get("rmse"),
                    metrics_text.get("psnr") if output.get("is_reconstructed_frame") else "n/a",
                )
            )

    report = {
        "model": manifest["model"],
        "baseline_manifest_sha256": sha256(manifest_path),
        "thresholds": {"max_abs": max_abs_limit, "rmse": rmse_limit},
        "all_passed": all(result["passed"] for result in results),
        "first_failed": first_failed,
        "results": results,
    }
    output_path = (args.output or vendor_dir / "precision_report.json").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print("all_passed={}".format(report["all_passed"]))
    print("first_failed={}".format(first_failed))
    print("report={}".format(output_path))
    raise SystemExit(0 if report["all_passed"] else 2)


if __name__ == "__main__":
    main()
