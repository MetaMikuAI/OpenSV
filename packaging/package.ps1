$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$target = 'x86_64-pc-windows-msvc'
cargo build --release --locked --target $target --manifest-path "$repo/Cargo.toml"
if ($LASTEXITCODE -ne 0) { throw 'Rust build failed' }
uv run --no-project --python 3.13 python "$PSScriptRoot/build_bundle.py" --engine "$repo/target/$target/release/opensv.exe" --target $target
if ($LASTEXITCODE -ne 0) { throw 'Bundle packaging failed' }
