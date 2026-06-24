# Clean GVC-RT Export Tools

Run these scripts only on the server/PyTorch environment. The local PC is used
for editing, APK build, and adb installation only.

Clean v1 is intentionally FP32-only. The earlier FP16 export converted several
graphs to FP32 after recording an FP16 baseline, making the comparison invalid.
Use the command below exactly; FP16 is deferred until every graph has a
source-matched FP16 export path.

```bash
cd /media/ltelab/D/weilingfeng/GVC-RT_inference
python /media/ltelab/D/weilingfeng/GVC-RT_clean_android/server_tools/export_clean_gvcrt_modules.py \
  --source-root /media/ltelab/D/weilingfeng/GVC-RT_inference \
  --output-assets /media/ltelab/D/weilingfeng/GVC-RT_clean_android/app/src/main/assets \
  --height 256 \
  --width 512 \
  --qp 0 \
  --precision fp32
```

The generated assets are:

- `models/*.onnx`
- `baseline/inputs/*.f32le`
- `baseline/tensors/*.f32le`
- `baseline/entropy/*_cdf*.i32le`, packed symbols, and z symbols
- `baseline/bitstream/i_rans_payload.bin`, `p_rans_payload.bin`, and `encoded_ip.gvc`
- `gvcrt_clean_manifest.json`

Copy the generated `app/src/main/assets` directory back into the Android clean
project before building the APK.
