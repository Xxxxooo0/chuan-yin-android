#!/usr/bin/env bash
set -euo pipefail

ROOT="${ANDROID_ROOT:-/media/ltelab/D/weilingfeng/GVC-RT_clean_android}"
SDK_ROOT="${NEUROPILOT_SDK_ROOT:-$ROOT/neuropilot-sdk-premium-8.0.11-build20260211}"
PY="${PYTHON:-/media/ltelab/D/weilingfeng/conda_envs/weilingfeng/bin/python}"
OUT="$ROOT/outputs/ip_recon_offline_audit"
ANALYZER="$ROOT/server_tools/analyze_recon_neuron_support.py"

mkdir -p "$OUT"

run_group() {
  local name="$1"
  local assets="$2"
  local labels="$3"
  local group_out="$OUT/$name"

  echo "[offline-audit] group=$name assets=$assets labels=$labels"
  "$PY" -u "$ANALYZER" \
    --android-root "$ROOT" \
    --assets-dir "$assets" \
    --output-dir "$group_out" \
    --ncc-tflite "$SDK_ROOT" \
    --arch mdla5.3 \
    --labels "$labels" \
    --opt-bw \
    --relax-fp32 \
    --compile-dla
}

run_group \
  i_featuredec_full \
  "$ROOT/app/src/main/assets/featuredec_i_nhwc" \
  all

run_group \
  i_featuredec_split \
  "$ROOT/app/src/main/assets/featuredec_i_split_nhwc" \
  all

run_group \
  ip_recon_segments \
  "$ROOT/app/src/main/assets/recon_diagnostic" \
  "i_latent_conv_in_fp32,i_latent_conv_in_nhwc_fp32,i_latent_decoder_fp32,i_recon_full_fp32,p_latent_decoder_fp32,p_recon_mlp_dcb0_fp32,p_recon_mlp_dcb1_fp32,p_decoder_stage1_conv_only_fp32,p_decoder_stage2_blocks_only_fp32,p_upsampler_original_fp32,p_decoder_stage3_blocks_only_fp32,p_decoder_stage4_blocks_explicit_fp32,p_recon_final_head_no_ada_fp32"

"$PY" - "$OUT" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = []
for manifest in sorted(root.glob("*/recon_neuron_diagnostics.json")):
    group = manifest.parent.name
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for record in payload.get("records", []):
        compile_ok = (
            record.get("compile_dla_rc") == 0
            and bool(record.get("dla"))
            and Path(record["dla"]).is_file()
        )
        diagnostics = []
        for line in record.get("diagnostic_lines", []) + record.get("dla_diagnostic_lines", []):
            if line not in diagnostics:
                diagnostics.append(line)
        records.append({
            "group": group,
            "label": record.get("label"),
            "tflite": record.get("tflite"),
            "tflite_sha256": record.get("tflite_sha256"),
            "check_target_rc": record.get("check_target_rc"),
            "exec_plan_rc": record.get("exec_plan_rc"),
            "compile_dla_rc": record.get("compile_dla_rc"),
            "offline_compile_ok": compile_ok,
            "dla": record.get("dla"),
            "dla_sha256": record.get("dla_sha256"),
            "unsupported_or_warning_lines": diagnostics,
            "logs": {
                "check_target": record.get("check_target_log"),
                "exec_plan": record.get("exec_plan_log"),
                "compile_dla": record.get("compile_dla_log"),
            },
        })

summary = {
    "target": "mdla5.3",
    "mode": "offline_dla_compile",
    "flags": ["--opt-bw", "--relax-fp32"],
    "checked": len(records),
    "offline_compile_ok": sum(1 for r in records if r["offline_compile_ok"]),
    "offline_compile_failed": sum(1 for r in records if not r["offline_compile_ok"]),
    "records": records,
}
json_path = root / "ip_recon_offline_audit.json"
json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

lines = [
    "# I/P Recon Offline Compile Audit",
    "",
    f"- checked: {summary['checked']}",
    f"- offline compile ok: {summary['offline_compile_ok']}",
    f"- offline compile failed: {summary['offline_compile_failed']}",
    "",
    "| Group | Model | DLA | check | plan | compile | Unsupported / warning |",
    "|---|---|---:|---:|---:|---:|---|",
]
for record in records:
    diagnostic = "<br>".join(record["unsupported_or_warning_lines"]) or "none"
    lines.append(
        f"| {record['group']} | {record['label']} | "
        f"{'OK' if record['offline_compile_ok'] else 'FAIL'} | "
        f"{record['check_target_rc']} | {record['exec_plan_rc']} | "
        f"{record['compile_dla_rc']} | {diagnostic} |"
    )
md_path = root / "ip_recon_offline_audit.md"
md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"wrote {json_path}")
print(f"wrote {md_path}")
print(
    f"checked={summary['checked']} "
    f"offline_compile_ok={summary['offline_compile_ok']} "
    f"offline_compile_failed={summary['offline_compile_failed']}"
)
for record in records:
    state = "OK" if record["offline_compile_ok"] else "FAIL"
    print(
        f"{state} group={record['group']} model={record['label']} "
        f"check={record['check_target_rc']} plan={record['exec_plan_rc']} "
        f"compile={record['compile_dla_rc']}"
    )
    for diagnostic in record["unsupported_or_warning_lines"][:8]:
        print(f"  - {diagnostic}")
PY

echo "[offline-audit] report=$OUT/ip_recon_offline_audit.md"
echo "[offline-audit] manifest=$OUT/ip_recon_offline_audit.json"
