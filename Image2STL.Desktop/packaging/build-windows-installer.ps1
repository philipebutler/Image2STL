$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
dotnet publish "$root/Image2STL.Desktop.csproj" -c Release -r win-x64 --self-contained true /p:PublishSingleFile=true

$publishDir = Join-Path $root "bin/Release/net10.0/win-x64/publish"
$nsis = Get-Command makensis -ErrorAction SilentlyContinue
if ($nsis) {
    $scriptPath = Join-Path $PSScriptRoot "image2stl-installer.nsi"
    if (Test-Path $scriptPath) {
        & $nsis.Source /DPUBLISH_DIR="$publishDir" $scriptPath
    } else {
        Write-Host "Publish completed at $publishDir (NSIS script not found)."
    }
} else {
    Write-Host "Publish completed at $publishDir (makensis not installed)."
}
