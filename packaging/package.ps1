$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
cargo build --release --manifest-path "$repo/Cargo.toml"
if ($LASTEXITCODE -ne 0) { throw 'Rust build failed' }
$package = Join-Path $repo ("local/packages/OpenSV-windows-x64-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Path "$package/training/opensv_train/web" -Force | Out-Null
Copy-Item -LiteralPath "$repo/target/release/opensv.exe", "$repo/LICENSE", "$PSScriptRoot/README.md", "$PSScriptRoot/start.cmd" -Destination $package
Copy-Item -LiteralPath "$repo/training/pyproject.toml", "$repo/training/uv.lock", "$repo/training/.python-version", "$repo/training/README.md" -Destination "$package/training"
Get-ChildItem -LiteralPath "$repo/training/opensv_train" -Filter '*.py' | Copy-Item -Destination "$package/training/opensv_train"
Copy-Item -Path "$repo/training/opensv_train/web/*.html" -Destination "$package/training/opensv_train/web"
Compress-Archive -LiteralPath $package -DestinationPath "$package.zip"
Write-Output "$package.zip"
