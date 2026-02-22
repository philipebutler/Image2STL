using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text.Json;
using Avalonia.Media.Imaging;

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
    };

    public ObservableCollection<ImageThumbnailViewModel> Images { get; } = new();
    public string? CurrentProjectPath { get; private set; }

    private string _status = "Drag and drop 3-5 images or use Add Images.";
    public string Status
    {
        get => _status;
        set => SetProperty(ref _status, value);
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
        Status = "Started new project. Add 3-5 images.";
    }

    public void SaveProject(string projectPath)
    {
        try
        {
            var project = new DesktopProject(Images.Select(image => image.FilePath).ToList());
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
            AddImages(project.Images);
            CurrentProjectPath = projectPath;
            Status = $"{Status} Project loaded.";
        }
        catch (Exception ex)
        {
            Status = $"Unable to load project file: {ex.Message}";
        }
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

public record DesktopProject(List<string> Images);
