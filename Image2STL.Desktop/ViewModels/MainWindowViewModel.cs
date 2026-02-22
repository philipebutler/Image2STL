using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text.Json;
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

    private string? _currentOperationId;

    public ICommand NewProjectCommand { get; }
    public ICommand SaveProjectCommand { get; }
    public ICommand GenerateCommand { get; }

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
        GenerateCommand = new RelayCommand(StartGeneration);
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
                ScaleAxis);
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
        IsProcessing = false;
        Progress = 0;
        ProgressText = "";
        ShowTimeWarning = false;
        Status = "Operation cancelled.";
    }

    public void HandleEngineMessage(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            var type = root.GetProperty("type").GetString();

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
            }
            else if (type == "success")
            {
                IsProcessing = false;
                Progress = 100;
                ShowTimeWarning = false;
                Status = "Reconstruction complete. Ready for export.";
            }
            else if (type == "error")
            {
                IsProcessing = false;
                ShowTimeWarning = false;
                var message = root.TryGetProperty("message", out var msgProp) ? msgProp.GetString() ?? "Unknown error" : "Unknown error";
                var suggestion = root.TryGetProperty("suggestion", out var sugProp) ? sugProp.GetString() ?? "" : "";
                ShowError(message, suggestion);
            }
        }
        catch (JsonException)
        {
            // Ignore malformed JSON messages from the engine
        }
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
    string ScaleAxis = "longest");
