# Repository Guidelines

## Project Structure & Module Organization

This repository is a clean Android deployment for GVC-RT inference and diagnostics. The Android app lives in `app/`.

- `app/src/main/java/com/gvcrt/clean/`: Kotlin app logic, ONNX/TFLite runners, module tests, speed benchmarks, memory sampling, and image inference.
- `app/src/main/cpp/`: native C++/JNI code, rANS integration, MTK TFLite bridge, and fused-kernel experiments.
- `app/src/main/assets/`: runtime assets packaged into the APK. Large baseline/model assets under `models/` and `baseline/` are intentionally ignored unless explicitly required.
- `app/src/main/jniLibs/`: native shared libraries required by the Android runtime.
- `server_tools/`: server-side export, validation, and diagnostic scripts. These are for the remote Linux/server environment, not local PC inference.
- `docs/`: project notes and supporting documentation.

## Build, Test, and Development Commands

Run commands from the repository root:

```powershell
.\gradlew.bat :app:compileDebugKotlin
```

Compiles Kotlin and catches most Android source errors quickly.

```powershell
.\gradlew.bat :app:assembleDebug
```

Builds the debug APK, including CMake/JNI code and packaged assets.

```powershell
.\sdk\platform-tools\adb.exe install -r .\app\build\outputs\apk\debug\app-debug.apk
```

Installs the APK on a connected Android device.

## Coding Style & Naming Conventions

Use Kotlin for app orchestration and C++17 for JNI/native kernels. Keep changes scoped to the active module. Kotlin classes use `PascalCase`; functions and variables use `camelCase`; benchmark labels should be stable snake-like strings, for example `native_p_recon_stage_precision_probe`. Prefer explicit tensor shapes and asset paths in logs.

All `README.md` files in this repository must be written in Chinese. Commands, paths, API names, and identifiers may remain in their original form.

## Testing Guidelines

This workstation is edit/build only for model inference. Do not run local PyTorch, ONNX Runtime, TFLite, precision, or performance inference tests on the PC. Use:

- Android device via `adb` for APK tests, speed tests, memory logs, and Android output dumps.
- Remote server scripts in `server_tools/` for PyTorch/export/precision validation.

Always record the exact adb extra or server command used and include key hashes when comparing model outputs.

## Commit & Pull Request Guidelines

Commit history uses short imperative messages, for example `Add recon diagnostics and MTK TFLite runtime` and `Optimize native rANS encoding path`. Keep commits focused and avoid bundling unrelated experiments. Before committing, run `.\gradlew.bat :app:assembleDebug` and check `git status --short` for ignored SDK, MTK tool, build, and cache files.

PRs or handoffs should include: purpose, tested Android/server commands, device/backend used, important precision or speed results, and any known fallback or unsupported operator behavior.

## Security & Configuration Tips

Do not commit local SDKs, server credentials, generated zips, build outputs, or downloaded MTK tool archives. Keep paths like `sdk/`, `mtk/`, `outputs/`, `app/build/`, and `local.properties` ignored.
