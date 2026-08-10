#!/usr/bin/env python3
"""Audit the published MTK offline model bundle without running inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "module_offline_manifest.json": {
        "temporal_from_frame_big",
        "temporal_from_feature_big",
        "i_encoder_analysis_big",
        "p_encoder_analysis_big",
    },
    "entropy_offline_manifest.json": {
        "i_hyper_enc_continuous",
        "i_hyper_prior_shared",
        "i_prior_stage0_params",
        "i_prior_reduce",
        "i_prior_stage1_continuous",
        "i_prior_stage2_continuous",
        "i_prior_stage3_continuous",
        "p_hyper_enc_continuous",
        "p_hyper_prior_shared",
        "p_prior_stage0_params",
        "p_prior_stage1_continuous",
    },
    "decoder_offline_manifest.json": {
        "i_featuredec_norm_rewrite_fp32",
        "i_generator_stage1_norm_rewrite_fp32",
        "i_generator_stage2_norm_rewrite_fp32",
        "i_generator_stage3_norm_rewrite_fp32",
        "i_generator_stage4_norm_rewrite_fp32",
        "i_generator_final_norm_rewrite_fp32",
        "p_featuredec_latent_norm_rewrite_fp32",
        "p_featuredec_mlp0_norm_rewrite_fp32",
        "p_featuredec_mlp1_norm_rewrite_fp32",
        "p_generator_stage1_norm_rewrite_fp32",
        "p_generator_stage2_norm_rewrite_fp32",
        "p_generator_stage3_norm_rewrite_fp32",
        "p_generator_stage4_norm_rewrite_fp32",
        "p_generator_final_norm_rewrite_fp32",
    },
    "decoder_merged_offline_manifest.json": {
        "i_decoder_synthesis_merged_fp32",
        "p_decoder_synthesis_merged_fp32",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--android-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.android_root.resolve()
    assets = root / "app" / "src" / "mtkOffline" / "assets"
    failures: list[str] = []
    report: dict[str, object] = {"assets_root": str(assets), "manifests": {}}

    for manifest_name, expected_names in EXPECTED.items():
        path = assets / "offline_models" / manifest_name
        if not path.is_file():
            failures.append(f"missing manifest: {path}")
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        records = {item["name"]: item for item in manifest.get("models", [])}
        missing = sorted(expected_names - set(records))
        extra = sorted(set(records) - expected_names)
        if missing:
            failures.append(f"{manifest_name} missing models: {missing}")

        verified = 0
        for name in sorted(expected_names & set(records)):
            record = records[name]
            asset = assets / record["asset"]
            if not asset.is_file():
                failures.append(f"{name} missing asset: {asset}")
                continue
            actual_sha = sha256(asset)
            if actual_sha != record.get("dla_sha256"):
                failures.append(f"{name} SHA mismatch: {actual_sha} != {record.get('dla_sha256')}")
                continue
            if not record.get("offline_compile_verified"):
                failures.append(f"{name} is not marked offline_compile_verified")
                continue
            if (
                manifest_name == "decoder_merged_offline_manifest.json"
                and not record.get("precision_verified")
            ):
                failures.append(f"{name} is not marked precision_verified")
                continue
            verified += 1

        report["manifests"][manifest_name] = {
            "expected": len(expected_names),
            "present": len(expected_names & set(records)),
            "verified": verified,
            "missing": missing,
            "extra": extra,
        }

    report["expected_models"] = sum(len(items) for items in EXPECTED.values())
    report["failures"] = failures
    report["passed"] = not failures
    output = assets / "offline_models" / "offline_bundle_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"wrote {output}")
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
