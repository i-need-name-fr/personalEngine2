param(
    [switch]$Restore,
    [string]$DxcPath = ""
)

$ErrorActionPreference = 'Stop'

function Find-NewDxc {
    if ($DxcPath -ne "" -and (Test-Path $DxcPath)) { return (Resolve-Path $DxcPath).Path }

    $local = Join-Path $PSScriptRoot 'dxc\dxcompiler.dll'
    if (Test-Path $local) { return $local }

    $candidates = @()
    if ($env:VULKAN_SDK) { $candidates += (Join-Path $env:VULKAN_SDK 'Bin\dxcompiler.dll') }
    if (Test-Path 'C:\VulkanSDK') {
        $candidates += Get-ChildItem 'C:\VulkanSDK' -Directory | Sort-Object Name -Descending | ForEach-Object { Join-Path $_.FullName 'Bin\dxcompiler.dll' }
    }
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    return $null
}

function Get-Version($path) {
    return (Get-Item $path).VersionInfo.FileVersion
}

$roots = @(
    "$env:USERPROFILE\.vscode\extensions",
    "$env:USERPROFILE\.vscode-insiders\extensions",
    "$env:USERPROFILE\.cursor\extensions",
    "$env:USERPROFILE\.vscode-oss\extensions"
)

$targets = @()
foreach ($root in $roots) {
    if (Test-Path $root) {
        $targets += Get-ChildItem $root -Directory -Filter 'antaalt.shader-validator-*' |
            ForEach-Object { Join-Path $_.FullName 'bin\win32-x64\dxcompiler.dll' } |
            Where-Object { Test-Path $_ }
    }
}

if ($targets.Count -eq 0) {
    Write-Host 'The Shader validator extension (antaalt.shader-validator) was not found. Install it in VS Code first.' -ForegroundColor Red
    exit 1
}

$source = $null
if (-not $Restore) {
    $source = Find-NewDxc
    if (-not $source) {
        Write-Host 'No new dxcompiler.dll found. Install the Vulkan SDK, or put the file into tools\dxc\dxcompiler.dll, or pass -DxcPath <file>.' -ForegroundColor Red
        exit 1
    }
    Write-Host ("New DXC: {0} (version {1})" -f $source, (Get-Version $source))
}

foreach ($target in $targets) {
    Write-Host ""
    $extensionDir = Split-Path (Split-Path (Split-Path $target))
    Write-Host ("Extension: {0}" -f (Split-Path $extensionDir -Leaf))
    $backup = "$target.bak"

    if ($Restore) {
        if (-not (Test-Path $backup)) {
            Write-Host '  nothing to restore (no .bak file)'
            continue
        }
        Move-Item $target "$target.prev" -Force
        Copy-Item $backup $target -Force
        Write-Host ("  restored the original DXC (version {0})" -f (Get-Version $target)) -ForegroundColor Green
        continue
    }

    if ((Test-Path $target) -and ((Get-FileHash $target).Hash -eq (Get-FileHash $source).Hash)) {
        Write-Host ("  already up to date (version {0})" -f (Get-Version $target)) -ForegroundColor Green
        continue
    }

    if (-not (Test-Path $backup)) {
        Move-Item $target $backup
        Write-Host '  the original DXC is saved as dxcompiler.dll.bak'
    } else {
        Move-Item $target "$target.prev" -Force
    }
    Copy-Item $source $target -Force
    Write-Host ("  replaced with DXC {0}" -f (Get-Version $target)) -ForegroundColor Green
}

Write-Host ""
Write-Host 'Done. In VS Code run "Developer: Reload Window" so the shader server restarts.'
