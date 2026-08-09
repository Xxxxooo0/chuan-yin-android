param(
    [Parameter(Mandatory = $true)] [string] $DlaInputPackage,
    [Parameter(Mandatory = $true)] [string] $TfliteModelDirectory,
    [Parameter(Mandatory = $true)] [string] $PackageName,
    [Parameter(Mandatory = $true)] [string] $OutputArchive
)

$ErrorActionPreference = "Stop"

function Get-Sha256([string] $Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

$temporary = Join-Path ([System.IO.Path]::GetTempPath()) ("gvc_rt_tflite_" + [guid]::NewGuid())
$extractRoot = Join-Path $temporary "extract"
$packageRoot = Join-Path $temporary $PackageName
New-Item -ItemType Directory -Path $extractRoot, $packageRoot | Out-Null

try {
    tar -xzf $DlaInputPackage -C $extractRoot
    if ($LASTEXITCODE -ne 0) { throw "failed to extract $DlaInputPackage" }

    $sourceRoot = @(Get-ChildItem -LiteralPath $extractRoot -Directory)
    if ($sourceRoot.Count -ne 1) { throw "expected one package root in $DlaInputPackage" }
    $sourceRoot = $sourceRoot[0].FullName

    Copy-Item -LiteralPath (Join-Path $sourceRoot "inputs") -Destination $packageRoot -Recurse
    Copy-Item -LiteralPath $TfliteModelDirectory -Destination (Join-Path $packageRoot "models") -Recurse

    $manifest = Get-Content -Raw -LiteralPath (Join-Path $sourceRoot "manifest.json") | ConvertFrom-Json
    $inputManifest = Get-Content -Raw -LiteralPath (Join-Path $sourceRoot "input_manifest.json") | ConvertFrom-Json
    $manifest.package = $PackageName
    if ($manifest.PSObject.Properties.Name -contains "target") {
        $manifest.target = "TFLite intermediate for NeuroPilot 7.2.4 / MDLA 5.0 validation"
    }
    $manifest | Add-Member -NotePropertyName artifact_format -NotePropertyValue "TFLite" -Force
    $manifest | Add-Member -NotePropertyName current_export_converter -NotePropertyValue "mtk_converter 8.16.0" -Force
    $manifest | Add-Member -NotePropertyName enterprise_runtime -NotePropertyValue "NeuroPilot 7.2.4; Neuron 3.8.275480; MDLA 5.0" -Force
    $manifest.PSObject.Properties.Remove("checkpoint")
    $manifest.PSObject.Properties.Remove("checkpoint_sha256")

    foreach ($model in $manifest.models) {
        $modelPath = Join-Path $packageRoot ("models\{0}.tflite" -f $model.name)
        if (-not (Test-Path -LiteralPath $modelPath)) { throw "missing model: $modelPath" }
        $model.file = "models/$($model.name).tflite"
        $model.bytes = (Get-Item -LiteralPath $modelPath).Length
        $model.sha256 = Get-Sha256 $modelPath
        $model.PSObject.Properties.Remove("offline_compile_verified")
        $model.PSObject.Properties.Remove("mdla_only")
    }
    $manifest | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath (Join-Path $packageRoot "manifest.json") -Encoding utf8

    $inputManifest.package = $PackageName
    $inputManifest.purpose = "vendor_tflite_compatibility_and_precision_input_collection"
    $inputManifest | Add-Member -NotePropertyName artifact_format -NotePropertyValue "TFLite" -Force
    $inputManifest.PSObject.Properties.Remove("source_model_package_sha256")
    $inputManifest.PSObject.Properties.Remove("source_model_manifest_sha256")
    $inputManifest | ConvertTo-Json -Depth 40 | Set-Content -LiteralPath (Join-Path $packageRoot "input_manifest.json") -Encoding utf8

    @"
# $PackageName

This package contains FP32 NHWC TFLite models and fixed 270p/QP0 input tensors for enterprise compatibility and performance testing.

The models were generated with MTK Converter 8.16.0. Validate them with the matching NeuroPilot 7.2.4 host toolchain and compile for `mdla5.0` before device execution.

For every stage in `input_manifest.json`, preserve the listed tensor shape, order, and FP32 little-endian layout. Save returned tensors using the listed `vendor_file` names.
"@ | Set-Content -LiteralPath (Join-Path $packageRoot "README.md") -Encoding utf8

    $checksums = Get-ChildItem -LiteralPath $packageRoot -Recurse -File |
        Where-Object Name -ne "SHA256SUMS.txt" |
        Sort-Object FullName |
        ForEach-Object {
            $relative = $_.FullName.Substring($packageRoot.Length + 1).Replace("\", "/")
            "$(Get-Sha256 $_.FullName)  $relative"
        }
    $checksums | Set-Content -LiteralPath (Join-Path $packageRoot "SHA256SUMS.txt") -Encoding ascii

    $outputParent = Split-Path -Parent $OutputArchive
    New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
    if (Test-Path -LiteralPath $OutputArchive) { throw "output already exists: $OutputArchive" }
    tar -czf $OutputArchive -C $temporary $PackageName
    if ($LASTEXITCODE -ne 0) { throw "failed to create $OutputArchive" }

    Write-Output "archive=$OutputArchive"
    Write-Output "archive_sha256=$(Get-Sha256 $OutputArchive)"
    Write-Output "models=$($manifest.models.Count)"
    Write-Output "inputs=$(@(Get-ChildItem (Join-Path $packageRoot 'inputs') -Recurse -File).Count)"
}
finally {
    if (Test-Path -LiteralPath $temporary) {
        Remove-Item -LiteralPath $temporary -Recurse -Force
    }
}
