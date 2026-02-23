using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows.Input;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.Input;

namespace Image2STL.Desktop.ViewModels;

public class MainWindowViewModel : ViewModelBase
{
    private static readonly JsonSerializerOptions ProjectJsonOptions = new() { WriteIndented = true };
    private static readonly HashSet<string> SupportedExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".heic",
        ".heif",
        ".webp",
        ".avif",
    };

    public ObservableCollection<ImageThumbnailViewModel> Images { get; } = new();
    public string? CurrentProjectPath { get; private set; }

    private string _status = "Drag and drop 3-5 images or use Add Images.";
    public string Status
    {
        get => _status;
        set => SetProperty(ref _status, value);
    }

    private bool _isLocalMode = true;
    public bool IsLocalMode
    {
        get => _isLocalMode;
        set
        {
            if (SetProperty(ref _isLocalMode, value) && value)
            {
                SetProperty(ref _isCloudMode, false, nameof(IsCloudMode));
            }
        }
    }

    private bool _isCloudMode;
    public bool IsCloudMode
    {
        get => _isCloudMode;
        set
        {
            if (SetProperty(ref _isCloudMode, value) && value)
            {
                SetProperty(ref _isLocalMode, false, nameof(IsLocalMode));
            }
        }
    }

    private decimal _scaleMm = 150m;
    public decimal ScaleMm
    {
        get => _scaleMm;
        set => SetProperty(ref _scaleMm, value);
    }

    private string _scaleAxis = "longest";
    public string ScaleAxis
    {
        get => _scaleAxis;
        set => SetProperty(ref _scaleAxis, value ?? "longest");
    }

    private bool _isProcessing;
    public bool IsProcessing
    {
        get => _isProcessing;
        set => SetProperty(ref _isProcessing, value);
    }

    private double _progress;
    public double Progress
    {
        get => _progress;
        set => SetProperty(ref _progress, value);
    }

    private string _progressText = "";
    public string ProgressText
    {
        get => _progressText;
        set => SetProperty(ref _progressText, value);
    }

    private bool _showTimeWarning;
    public bool ShowTimeWarning
    {
        get => _showTimeWarning;
        set => SetProperty(ref _showTimeWarning, value);
    }

    private string _timeWarningText = "";
    public string TimeWarningText
    {
        get => _timeWarningText;
        set => SetProperty(ref _timeWarningText, value);
    }

    private bool _hasError;
    public bool HasError
    {
        get => _hasError;
        set => SetProperty(ref _hasError, value);
    }

    private string _errorMessage = "";
    public string ErrorMessage
    {
        get => _errorMessage;
        set => SetProperty(ref _errorMessage, value);
    }

    private string _errorSuggestion = "";
    public string ErrorSuggestion
    {
        get => _errorSuggestion;
        set => SetProperty(ref _errorSuggestion, value);
    }

    private bool _showEnvironmentStatus;
    public bool ShowEnvironmentStatus
    {
        get => _showEnvironmentStatus;
        set => SetProperty(ref _showEnvironmentStatus, value);
    }

    private string _environmentSummary = "";
    public string EnvironmentSummary
    {
        get => _environmentSummary;
        set => SetProperty(ref _environmentSummary, value);
    }

    private string _environmentDetails = "";
    public string EnvironmentDetails
    {
        get => _environmentDetails;
        set => SetProperty(ref _environmentDetails, value);
    }

    private bool _isCheckingEnvironment;
    public bool IsCheckingEnvironment
    {
        get => _isCheckingEnvironment;
        set => SetProperty(ref _isCheckingEnvironment, value);
    }

    private string _cloudApiKey = "";
    public string CloudApiKey
    {
        get => _cloudApiKey;
        set => SetProperty(ref _cloudApiKey, value);
    }

    private string _cloudApiKeyEnvVar = "MESHY_API_KEY";
    public string CloudApiKeyEnvVar
    {
        get => _cloudApiKeyEnvVar;
        set => SetProperty(ref _cloudApiKeyEnvVar, string.IsNullOrWhiteSpace(value) ? "MESHY_API_KEY" : value.Trim());
    }

    private string _pythonExecutable = "python3";
    public string PythonExecutable
    {
        get => _pythonExecutable;
        set => SetProperty(ref _pythonExecutable, string.IsNullOrWhiteSpace(value) ? "python3" : value.Trim());
    }

    private string? _currentOperationId;
    private Process? _activeEngineProcess;

    public ICommand NewProjectCommand { get; }
    public ICommand SaveProjectCommand { get; }
    public ICommand GenerateCommand { get; }
    public ICommand CheckLocalSetupCommand { get; }
    public ICommand CheckCloudSetupCommand { get; }

    public string ExportFileName =>
        string.IsNullOrWhiteSpace(CurrentProjectPath)
            ? "model.stl"
            : Path.GetFileNameWithoutExtension(CurrentProjectPath) + ".stl";

    public MainWindowViewModel()
    {
        NewProjectCommand = new RelayCommand(NewProject);
        SaveProjectCommand = new RelayCommand(() =>
        {
            if (!string.IsNullOrWhiteSpace(CurrentProjectPath))
            {
                SaveProject(CurrentProjectPath);
            }
        });
        GenerateCommand = new AsyncRelayCommand(GenerateAsync);
        CheckLocalSetupCommand = new AsyncRelayCommand(CheckLocalSetupAsync);
        CheckCloudSetupCommand = new AsyncRelayCommand(CheckCloudSetupAsync);
    }

    public void AddImages(IEnumerable<string> filePaths)
    {
        var skipped = 0;
        foreach (var filePath in filePaths)
        {
            if (!File.Exists(filePath))
            {
                skipped++;
                continue;
            }

            if (!SupportedExtensions.Contains(Path.GetExtension(filePath)))
            {
                skipped++;
                continue;
            }

            Bitmap? bitmap = null;
            try
            {
                using var stream = File.OpenRead(filePath);
                bitmap = Bitmap.DecodeToWidth(stream, 260);
            }
            catch
            {
                // Format not decodable by Avalonia (e.g. HEIC/HEIF) – accept
                // the file with a small placeholder bitmap so it still appears
                // in the image gallery and can be sent to the Python engine.
                bitmap = CreatePlaceholderBitmap();
            }

            Images.Add(new ImageThumbnailViewModel(filePath, Path.GetFileName(filePath), bitmap));
        }

        ClearError();
        UpdateStatus(skipped);
    }

    public void RemoveImage(string filePath)
    {
        var item = Images.FirstOrDefault(image => image.FilePath == filePath);
        if (item is null)
        {
            return;
        }

        item.Thumbnail.Dispose();
        Images.Remove(item);
        UpdateStatus(0);
    }

    public void NewProject()
    {
        ClearImages();
        CurrentProjectPath = null;
        ClearError();
        Status = "Started new project. Add 3-5 images.";
    }

    public void SaveProject(string projectPath)
    {
        try
        {
            var project = new DesktopProject(
                Images.Select(image => image.FilePath).ToList(),
                IsLocalMode ? "local" : "cloud",
                (double)ScaleMm,
                ScaleAxis,
                string.IsNullOrWhiteSpace(CloudApiKey) ? null : CloudApiKey,
                string.IsNullOrWhiteSpace(CloudApiKeyEnvVar) ? "MESHY_API_KEY" : CloudApiKeyEnvVar,
                string.IsNullOrWhiteSpace(PythonExecutable) ? "python3" : PythonExecutable);
            File.WriteAllText(projectPath, JsonSerializer.Serialize(project, ProjectJsonOptions));
            CurrentProjectPath = projectPath;
            Status = $"Saved project to {Path.GetFileName(projectPath)}.";
        }
        catch (Exception ex)
        {
            Status = $"Unable to save project file: {ex.Message}";
        }
    }

    public void LoadProject(string projectPath)
    {
        try
        {
            var project = JsonSerializer.Deserialize<DesktopProject>(File.ReadAllText(projectPath));
            if (project is null)
            {
                throw new InvalidDataException("Project file could not be parsed or is empty.");
            }
            ClearImages();
            ClearError();
            AddImages(project.Images);

            IsLocalMode = project.ReconstructionMode != "cloud";
            IsCloudMode = project.ReconstructionMode == "cloud";
            if (project.ScaleMm > 0)
            {
                ScaleMm = (decimal)project.ScaleMm;
            }
            if (!string.IsNullOrWhiteSpace(project.ScaleAxis))
            {
                ScaleAxis = project.ScaleAxis;
            }
            if (!string.IsNullOrWhiteSpace(project.CloudApiKey))
            {
                CloudApiKey = project.CloudApiKey;
            }
            if (!string.IsNullOrWhiteSpace(project.CloudApiKeyEnvVar))
            {
                CloudApiKeyEnvVar = project.CloudApiKeyEnvVar;
            }
            if (!string.IsNullOrWhiteSpace(project.PythonExecutable))
            {
                PythonExecutable = project.PythonExecutable;
            }

            CurrentProjectPath = projectPath;
            Status = $"{Status} Project loaded.";
        }
        catch (Exception ex)
        {
            Status = $"Unable to load project file: {ex.Message}";
        }
    }

    public void StartGeneration()
    {
        if (Images.Count < 3)
        {
            ShowError("Not enough images", "Add at least 3 images before generating.");
            return;
        }

        if (Images.Count > 5)
        {
            ShowError("Too many images", "Use 3-5 images for best results.");
            return;
        }

        ClearError();
        IsProcessing = true;
        Progress = 0;
        _currentOperationId = Guid.NewGuid().ToString();
        ProgressText = "Starting reconstruction...";
        Status = $"Generating 3D model ({(IsLocalMode ? "local" : "cloud")} mode)...";
    }

    public async Task GenerateAsync()
    {
        StartGeneration();
        if (HasError)
        {
            return;
        }

        var outputPath = Path.Combine(Path.GetTempPath(), $"image2stl-{Guid.NewGuid():N}.obj");
        var mode = IsLocalMode ? "local" : "cloud";
        var payload = JsonSerializer.Serialize(new
        {
            command = "reconstruct",
            mode,
            images = Images.Select(image => image.FilePath).ToList(),
            outputPath,
            operationId = _currentOperationId,
            apiKey = string.IsNullOrWhiteSpace(CloudApiKey) ? null : CloudApiKey,
            apiKeyEnvVar = string.IsNullOrWhiteSpace(CloudApiKeyEnvVar) ? "MESHY_API_KEY" : CloudApiKeyEnvVar,
        });

        try
        {
            var repoRoot = FindRepositoryRoot();
            var startInfo = new ProcessStartInfo
            {
                FileName = GetPythonExecutable(),
                WorkingDirectory = repoRoot,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            startInfo.ArgumentList.Add("-m");
            startInfo.ArgumentList.Add("image2stl.cli");
            startInfo.ArgumentList.Add("run");
            startInfo.ArgumentList.Add("--json");
            startInfo.ArgumentList.Add(payload);

            using var process = Process.Start(startInfo);
            if (process is null)
            {
                IsProcessing = false;
                ShowError("Unable to start Python engine", $"Ensure '{GetPythonExecutable()}' is installed and on PATH.");
                return;
            }

            _activeEngineProcess = process;
            var parsedAny = false;
            while (true)
            {
                var line = await process.StandardOutput.ReadLineAsync();
                if (line is null)
                {
                    break;
                }
                if (string.IsNullOrWhiteSpace(line) || !line.StartsWith("{"))
                {
                    continue;
                }

                HandleEngineMessage(line);
                parsedAny = true;
            }

            var stderr = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            if (!parsedAny)
            {
                IsProcessing = false;
                ShowError(
                    "Python engine did not return status messages",
                    string.IsNullOrWhiteSpace(stderr) ? "No diagnostic output was produced." : stderr.Trim());
                return;
            }

            if (process.ExitCode != 0 && !HasError)
            {
                IsProcessing = false;
                ShowError("Reconstruction failed", string.IsNullOrWhiteSpace(stderr) ? "Python engine exited with an error." : stderr.Trim());
            }
        }
        catch (Exception ex)
        {
            IsProcessing = false;
            ShowError("Reconstruction failed", ex.Message);
        }
        finally
        {
            _activeEngineProcess = null;
        }
    }

    public void ExportSTL(string exportPath)
    {
        try
        {
            Status = $"Exported STL to {Path.GetFileName(exportPath)}.";
        }
        catch (Exception ex)
        {
            ShowError("Export failed", ex.Message);
        }
    }

    public void CancelOperation()
    {
        if (_activeEngineProcess is not null && !_activeEngineProcess.HasExited)
        {
            try
            {
                _activeEngineProcess.Kill(entireProcessTree: true);
            }
            catch
            {
                // Ignore process termination errors
            }
        }

        IsProcessing = false;
        Progress = 0;
        ProgressText = "";
        ShowTimeWarning = false;
        Status = "Operation cancelled.";
    }

    public async Task CheckLocalSetupAsync()
    {
        if (IsCheckingEnvironment)
        {
            return;
        }

        IsCheckingEnvironment = true;
        ShowEnvironmentStatus = true;
        EnvironmentSummary = "Checking local Python setup...";
        EnvironmentDetails = "Running image2stl check_environment command.";
        Status = "Checking local setup...";
        ClearError();

        try
        {
            var repoRoot = FindRepositoryRoot();
            var payload = JsonSerializer.Serialize(new { command = "check_environment", mode = "local" });
            var startInfo = new ProcessStartInfo
            {
                FileName = GetPythonExecutable(),
                WorkingDirectory = repoRoot,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            startInfo.ArgumentList.Add("-m");
            startInfo.ArgumentList.Add("image2stl.cli");
            startInfo.ArgumentList.Add("run");
            startInfo.ArgumentList.Add("--json");
            startInfo.ArgumentList.Add(payload);

            using var process = Process.Start(startInfo);
            if (process is null)
            {
                ShowError("Unable to start Python process", $"Ensure '{GetPythonExecutable()}' is installed and on PATH.");
                EnvironmentSummary = "Setup check failed.";
                EnvironmentDetails = $"Could not start {GetPythonExecutable()} process.";
                return;
            }

            var stdout = await process.StandardOutput.ReadToEndAsync();
            var stderr = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            var parsedAny = false;
            foreach (var line in stdout.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                if (!line.StartsWith("{"))
                {
                    continue;
                }

                HandleEngineMessage(line);
                parsedAny = true;
            }

            if (!parsedAny)
            {
                EnvironmentSummary = "Setup check did not return JSON output.";
                EnvironmentDetails = string.IsNullOrWhiteSpace(stderr)
                    ? "No diagnostic output was produced."
                    : stderr.Trim();
            }

            if (process.ExitCode != 0)
            {
                ShowError("Python setup check failed", string.IsNullOrWhiteSpace(stderr) ? "Command exited with an error." : stderr.Trim());
            }
        }
        catch (Exception ex)
        {
            ShowError("Python setup check failed", ex.Message);
            EnvironmentSummary = "Setup check failed.";
            EnvironmentDetails = ex.Message;
        }
        finally
        {
            IsCheckingEnvironment = false;
        }
    }

    public async Task CheckCloudSetupAsync()
    {
        if (IsCheckingEnvironment)
        {
            return;
        }

        IsCheckingEnvironment = true;
        ShowEnvironmentStatus = true;
        EnvironmentSummary = "Checking cloud API setup...";
        EnvironmentDetails = "Validating Meshy.ai key configuration.";
        Status = "Checking cloud setup...";
        ClearError();

        try
        {
            var repoRoot = FindRepositoryRoot();
            var payload = JsonSerializer.Serialize(new
            {
                command = "check_environment",
                mode = "cloud",
                apiKey = string.IsNullOrWhiteSpace(CloudApiKey) ? null : CloudApiKey,
                apiKeyEnvVar = string.IsNullOrWhiteSpace(CloudApiKeyEnvVar) ? "MESHY_API_KEY" : CloudApiKeyEnvVar,
            });
            var startInfo = new ProcessStartInfo
            {
                FileName = GetPythonExecutable(),
                WorkingDirectory = repoRoot,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            startInfo.ArgumentList.Add("-m");
            startInfo.ArgumentList.Add("image2stl.cli");
            startInfo.ArgumentList.Add("run");
            startInfo.ArgumentList.Add("--json");
            startInfo.ArgumentList.Add(payload);

            using var process = Process.Start(startInfo);
            if (process is null)
            {
                ShowError("Unable to start Python process", $"Ensure '{GetPythonExecutable()}' is installed and on PATH.");
                EnvironmentSummary = "Cloud setup check failed.";
                EnvironmentDetails = $"Could not start {GetPythonExecutable()} process.";
                return;
            }

            var stdout = await process.StandardOutput.ReadToEndAsync();
            var stderr = await process.StandardError.ReadToEndAsync();
            await process.WaitForExitAsync();

            var parsedAny = false;
            foreach (var line in stdout.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                if (!line.StartsWith("{"))
                {
                    continue;
                }

                HandleEngineMessage(line);
                parsedAny = true;
            }

            if (!parsedAny)
            {
                EnvironmentSummary = "Cloud setup check did not return JSON output.";
                EnvironmentDetails = string.IsNullOrWhiteSpace(stderr)
                    ? "No diagnostic output was produced."
                    : stderr.Trim();
            }

            if (process.ExitCode != 0)
            {
                ShowError("Cloud setup check failed", string.IsNullOrWhiteSpace(stderr) ? "Command exited with an error." : stderr.Trim());
            }
        }
        catch (Exception ex)
        {
            ShowError("Cloud setup check failed", ex.Message);
            EnvironmentSummary = "Cloud setup check failed.";
            EnvironmentDetails = ex.Message;
        }
        finally
        {
            IsCheckingEnvironment = false;
        }
    }

    public void HandleEngineMessage(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            var type = root.GetProperty("type").GetString();
            var command = root.TryGetProperty("command", out var commandProp)
                ? commandProp.GetString() ?? string.Empty
                : string.Empty;

            if (type == "progress")
            {
                var progressValue = root.GetProperty("progress").GetDouble();
                Progress = progressValue * 100;
                ProgressText = root.TryGetProperty("status", out var statusProp)
                    ? statusProp.GetString() ?? ""
                    : "";

                if (root.TryGetProperty("estimatedSecondsRemaining", out var estProp))
                {
                    var est = estProp.GetInt32();
                    if (est > 600)
                    {
                        ShowTimeWarning = true;
                        TimeWarningText = $"⚠ Estimated processing time exceeds 10 minutes ({est / 60} min remaining)";
                    }
                    else
                    {
                        ShowTimeWarning = false;
                    }
                }

                if (ProgressText.Contains("TripoSR model", StringComparison.OrdinalIgnoreCase)
                    || ProgressText.Contains("downloading", StringComparison.OrdinalIgnoreCase))
                {
                    ShowEnvironmentStatus = true;
                    EnvironmentSummary = "Model status: local TripoSR load/download in progress";
                    EnvironmentDetails = ProgressText;
                }
            }
            else if (type == "success")
            {
                if (command == "check_environment")
                {
                    ApplyEnvironmentCheckStatus(root);
                    Status = root.TryGetProperty("mode", out var modeProp) && modeProp.GetString() == "cloud"
                        ? "Cloud setup check complete."
                        : "Local setup check complete.";
                }
                else
                {
                    IsProcessing = false;
                    Progress = 100;
                    ShowTimeWarning = false;
                    if (root.TryGetProperty("stats", out var stats)
                        && stats.TryGetProperty("model", out var model)
                        && model.TryGetProperty("cacheStatusBeforeLoad", out var cacheStatusElement))
                    {
                        var cacheStatus = cacheStatusElement.GetString() ?? "unknown";
                        ShowEnvironmentStatus = true;
                        EnvironmentSummary = cacheStatus switch
                        {
                            "cached" => "Model status: TripoSR already cached",
                            "not_cached" => "Model status: first-run download was required",
                            _ => "Model status: cache state unknown",
                        };
                        EnvironmentDetails = model.TryGetProperty("downloadLikelyRequired", out var dl)
                            ? $"cacheStatusBeforeLoad={cacheStatus}, downloadLikelyRequired={dl}"
                            : $"cacheStatusBeforeLoad={cacheStatus}";
                    }
                    Status = "Reconstruction complete. Ready for export.";
                }
            }
            else if (type == "error")
            {
                IsProcessing = false;
                ShowTimeWarning = false;
                var message = root.TryGetProperty("message", out var msgProp) ? msgProp.GetString() ?? "Unknown error" : "Unknown error";
                var suggestion = root.TryGetProperty("suggestion", out var sugProp) ? sugProp.GetString() ?? "" : "";
                if (root.TryGetProperty("missingDependencies", out var missing) && missing.ValueKind == JsonValueKind.Array)
                {
                    var missingPackages = new List<string>();
                    var installTargets = new List<string>();
                    var requiresTripoSrSourceSetup = false;
                    foreach (var item in missing.EnumerateArray())
                    {
                        var moduleName = item.TryGetProperty("module", out var moduleProp)
                            ? moduleProp.GetString()
                            : null;
                        if (item.TryGetProperty("package", out var package))
                        {
                            var packageName = package.GetString();
                            if (!string.IsNullOrWhiteSpace(packageName))
                            {
                                missingPackages.Add(packageName);
                            }
                        }
                        var installTarget = item.TryGetProperty("installTarget", out var targetProp)
                            ? targetProp.GetString()
                            : null;
                        if (installTarget == "__TRIPOSR_SOURCE_CHECKOUT__" || moduleName == "tsr")
                        {
                            requiresTripoSrSourceSetup = true;
                            continue;
                        }
                        if (!string.IsNullOrWhiteSpace(installTarget))
                        {
                            installTargets.Add(installTarget);
                        }
                    }

                    if (missingPackages.Count > 0)
                    {
                        suggestion = string.IsNullOrWhiteSpace(suggestion)
                            ? $"Missing packages: {string.Join(", ", missingPackages)}"
                            : $"{suggestion} Missing packages: {string.Join(", ", missingPackages)}";
                        if (installTargets.Count > 0)
                        {
                            suggestion = $"{suggestion} Install with: {GetPythonExecutable()} -m pip install {string.Join(" ", installTargets)}";
                        }
                        if (requiresTripoSrSourceSetup)
                        {
                            suggestion = $"{suggestion} TripoSR uses source checkout. Run: ./scripts/setup-macos.sh (macOS) or scripts/setup-windows.ps1 (Windows).";
                        }
                        ShowEnvironmentStatus = true;
                        EnvironmentSummary = "Local setup issue detected";
                        var detailLines = new List<string>
                        {
                            $"Missing required packages: {string.Join(", ", missingPackages)}"
                        };
                        if (installTargets.Count > 0)
                        {
                            detailLines.Add($"Run: {GetPythonExecutable()} -m pip install {string.Join(" ", installTargets)}");
                        }
                        if (requiresTripoSrSourceSetup)
                        {
                            detailLines.Add("TripoSR requires source checkout wiring; run setup script to configure .vendor/TripoSR and .pth linkage.");
                        }
                        EnvironmentDetails = string.Join("\n", detailLines);
                    }
                }
                ShowError(message, suggestion);
            }
        }
        catch (JsonException)
        {
            // Ignore malformed JSON messages from the engine
        }
    }

    private void ApplyEnvironmentCheckStatus(JsonElement root)
    {
        ShowEnvironmentStatus = true;

        var mode = root.TryGetProperty("mode", out var modeProp) ? modeProp.GetString() ?? "local" : "local";
        var summary = mode == "cloud" ? "Cloud setup check complete." : "Local setup check complete.";
        var details = new List<string>();

        if (root.TryGetProperty("python", out var python))
        {
            var pyVersion = python.TryGetProperty("version", out var versionProp) ? versionProp.GetString() : "unknown";
            var pyExe = python.TryGetProperty("executable", out var exeProp) ? exeProp.GetString() : "unknown";
            details.Add($"Python {pyVersion} ({pyExe})");
        }

        if (root.TryGetProperty("local", out var local))
        {
            var localOk = local.TryGetProperty("ok", out var okProp) && okProp.GetBoolean();
            summary = localOk ? "Local setup: ready" : "Local setup: missing required dependencies";

            if (local.TryGetProperty("missing", out var missing) && missing.ValueKind == JsonValueKind.Array)
            {
                var missingPackages = new List<string>();
                var installTargets = new List<string>();
                var requiresTripoSrSourceSetup = false;
                foreach (var item in missing.EnumerateArray())
                {
                    var moduleName = item.TryGetProperty("module", out var moduleProp)
                        ? moduleProp.GetString()
                        : null;
                    if (item.TryGetProperty("package", out var packageProp))
                    {
                        var packageName = packageProp.GetString();
                        if (!string.IsNullOrWhiteSpace(packageName))
                        {
                            missingPackages.Add(packageName);
                        }
                    }
                    var installTarget = item.TryGetProperty("installTarget", out var targetProp)
                        ? targetProp.GetString()
                        : null;
                    if (installTarget == "__TRIPOSR_SOURCE_CHECKOUT__" || moduleName == "tsr")
                    {
                        requiresTripoSrSourceSetup = true;
                        continue;
                    }
                    if (!string.IsNullOrWhiteSpace(installTarget))
                    {
                        installTargets.Add(installTarget);
                    }
                }

                if (missingPackages.Count > 0)
                {
                    details.Add($"Missing required packages: {string.Join(", ", missingPackages)}");
                    if (installTargets.Count > 0)
                    {
                        details.Add($"Install with: {GetPythonExecutable()} -m pip install {string.Join(" ", installTargets)}");
                    }
                    if (requiresTripoSrSourceSetup)
                    {
                        details.Add("TripoSR is source-only; run setup script to configure local source checkout linkage.");
                    }
                }
            }
        }

        if (root.TryGetProperty("cloud", out var cloud))
        {
            var configured = cloud.TryGetProperty("apiKeyConfigured", out var configuredProp) && configuredProp.GetBoolean();
            var envVar = cloud.TryGetProperty("apiKeyEnvVar", out var envVarProp)
                ? envVarProp.GetString() ?? "MESHY_API_KEY"
                : "MESHY_API_KEY";
            summary = configured ? "Cloud setup: API key configured" : "Cloud setup: API key missing";
            details.Add(configured
                ? "Meshy.ai API key is available for cloud mode."
                : "No Meshy.ai API key found. Enter a key or set the environment variable.");
            details.Add($"API key environment variable: {envVar}");
        }

        if (root.TryGetProperty("model", out var model))
        {
            var cache = model.TryGetProperty("cacheStatusBeforeLoad", out var cacheProp)
                ? cacheProp.GetString() ?? "unknown"
                : "unknown";
            var download = model.TryGetProperty("downloadLikelyRequired", out var downloadProp)
                ? downloadProp.ToString()
                : "unknown";
            details.Add($"TripoSR cache: {cache}; first-run download likely: {download}");
        }

        EnvironmentSummary = summary;
        EnvironmentDetails = string.Join("\n", details);
    }

    private static string FindRepositoryRoot()
    {
        var current = new DirectoryInfo(AppContext.BaseDirectory);
        while (current is not null)
        {
            var hasEngine = Directory.Exists(Path.Combine(current.FullName, "image2stl"));
            var hasReadme = File.Exists(Path.Combine(current.FullName, "README.md"));
            if (hasEngine && hasReadme)
            {
                return current.FullName;
            }

            current = current.Parent;
        }

        return Directory.GetCurrentDirectory();
    }

    private string GetPythonExecutable()
    {
        return string.IsNullOrWhiteSpace(PythonExecutable) ? "python3" : PythonExecutable.Trim();
    }

    private void ShowError(string message, string suggestion)
    {
        HasError = true;
        ErrorMessage = message;
        ErrorSuggestion = suggestion;
    }

    private void ClearError()
    {
        HasError = false;
        ErrorMessage = "";
        ErrorSuggestion = "";
    }

    private void ClearImages()
    {
        foreach (var image in Images)
        {
            image.Thumbnail.Dispose();
        }

        Images.Clear();
    }

    private void UpdateStatus(int skipped)
    {
        var baseStatus = Images.Count switch
        {
            < 3 => $"Loaded {Images.Count} image(s). Add at least {3 - Images.Count} more.",
            > 5 => $"Loaded {Images.Count} image(s). MVP target is 3-5 images.",
            _ => $"Loaded {Images.Count} image(s). Ready for reconstruction.",
        };
        Status = skipped > 0 ? $"{baseStatus} Skipped {skipped} invalid image file(s)." : baseStatus;
    }

    private static Bitmap CreatePlaceholderBitmap()
    {
        // Minimal 1x1 grey PNG used as a placeholder thumbnail for image
        // formats that Avalonia cannot decode natively (e.g. HEIC/HEIF).
        byte[] png =
        [
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
            0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53,
            0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, // IDAT chunk
            0x54, 0x08, 0xD7, 0x63, 0x90, 0x90, 0x90, 0x00,
            0x00, 0x00, 0x04, 0x00, 0x01, 0x9A, 0x60, 0xE1,
            0xD5, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, // IEND chunk
            0x44, 0xAE, 0x42, 0x60, 0x82,
        ];
        using var ms = new MemoryStream(png);
        return new Bitmap(ms);
    }
}

public record ImageThumbnailViewModel(string FilePath, string FileName, Bitmap Thumbnail);

public record DesktopProject(
    List<string> Images,
    string ReconstructionMode = "local",
    double ScaleMm = 150.0,
    string ScaleAxis = "longest",
    string? CloudApiKey = null,
    string CloudApiKeyEnvVar = "MESHY_API_KEY",
    string PythonExecutable = "python3");
