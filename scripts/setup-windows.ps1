param(
    [string]$PythonCommand = "py",
    [string]$VenvPath = ".venv",
    [switch]$System,
    [switch]$SkipOptionalFormats
)

$ErrorActionPreference = "Stop"

function Resolve-PythonCommand {
    param([string]$Cmd)

    $candidate = Get-Command $Cmd -ErrorAction SilentlyContinue
    if ($null -ne $candidate) {
        return $Cmd
    }

    if ($Cmd -eq "py") {
        $candidate = Get-Command python -ErrorAction SilentlyContinue
        if ($null -ne $candidate) {
            Write-Host "'py' launcher not found; falling back to 'python'."
            return "python"
        }
    }

    throw "Python command '$Cmd' was not found on PATH. Pass -PythonCommand with a valid command or full path."
}

function Test-GitAvailable {
    $gitCmd = Get-Command git -ErrorAction SilentlyContinue
    if ($null -eq $gitCmd) {
        throw "Git is required to install TripoSR from source. Install Git for Windows and retry."
    }
}

$PythonCommand = Resolve-PythonCommand -Cmd $PythonCommand
Test-GitAvailable

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..")
Set-Location $projectRoot

if ($System) {
    $InstallPython = $PythonCommand
    Write-Host "Using system Python interpreter:"
    & $InstallPython -c "import sys; print(sys.executable); print(sys.version)"
}
else {
    if (-not [System.IO.Path]::IsPathRooted($VenvPath)) {
        $VenvPath = Join-Path $projectRoot $VenvPath
    }

    $venvPython = Join-Path $VenvPath "Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        Write-Host "Creating virtual environment at: $VenvPath"
        & $PythonCommand -m venv $VenvPath
    }

    $InstallPython = $venvPython
    Write-Host "Using virtual environment Python interpreter:"
    & $InstallPython -c "import sys; print(sys.executable); print(sys.version)"
}

Write-Host "Upgrading pip tooling..."
& $InstallPython -m pip install --upgrade pip setuptools wheel

Write-Host "Installing core dependencies..."
& $InstallPython -m pip install pillow torch transformers huggingface-hub trimesh pymeshlab numpy

$triposrDir = Join-Path $projectRoot ".vendor\TripoSR"
New-Item -ItemType Directory -Path (Join-Path $projectRoot ".vendor") -Force | Out-Null

if (-not (Test-Path (Join-Path $triposrDir ".git"))) {
    Write-Host "Cloning TripoSR source into $triposrDir..."
    git clone https://github.com/VAST-AI-Research/TripoSR.git $triposrDir
}
else {
    Write-Host "Updating existing TripoSR source checkout..."
    git -C $triposrDir pull --ff-only
}

$triposrRequirements = Join-Path $triposrDir "requirements.txt"
if (-not (Test-Path $triposrRequirements)) {
    throw "TripoSR requirements file not found at $triposrRequirements"
}

Write-Host "Installing TripoSR runtime requirements..."
& $InstallPython -m pip install -r $triposrRequirements

$sitePackages = & $InstallPython -c "import site; print(site.getsitepackages()[0])"
$pthFile = Join-Path $sitePackages "triposr_local.pth"
Write-Host "Linking TripoSR source via .pth: $pthFile"
Set-Content -Path $pthFile -Value $triposrDir

if (-not $SkipOptionalFormats) {
    Write-Host "Installing optional image format dependencies..."
    & $InstallPython -m pip install pillow-heif pillow-avif-plugin
}

Write-Host "Verifying required imports..."
& $InstallPython -c "import importlib, sys; modules=['torch','PIL','tsr','transformers','huggingface_hub','trimesh','pymeshlab','numpy']; missing=[]
for m in modules:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append((m, str(e)))
if missing:
    print('Missing or broken modules detected:')
    [print(f' - {m}: {e}') for m,e in missing]
    sys.exit(1)
print('All required imports succeeded.')"

Write-Host "Running Image2STL local environment check..."
& $InstallPython -m image2stl.cli run --json '{"command":"check_environment","mode":"local"}'

Write-Host ""
Write-Host "Setup complete."
Write-Host "In the desktop app, set Python Command to: $InstallPython"
