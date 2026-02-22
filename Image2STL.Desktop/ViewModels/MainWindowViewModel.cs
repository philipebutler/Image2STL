using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text.Json;
using Avalonia.Media.Imaging;

namespace Image2STL.Desktop.ViewModels;

public class MainWindowViewModel : ViewModelBase
{
    private static readonly HashSet<string> SupportedExtensions = new(System.StringComparer.OrdinalIgnoreCase)
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
        foreach (var filePath in filePaths.Where(File.Exists))
        {
            if (!SupportedExtensions.Contains(Path.GetExtension(filePath)))
            {
                skipped += 1;
                continue;
            }

            try
            {
                using var stream = File.OpenRead(filePath);
                var bitmap = Bitmap.DecodeToWidth(stream, 260);
                Images.Add(new ImageThumbnailViewModel(filePath, Path.GetFileName(filePath), bitmap));
            }
            catch
            {
                skipped += 1;
            }
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
        var project = new DesktopProject(Images.Select(image => image.FilePath).ToList());
        File.WriteAllText(projectPath, JsonSerializer.Serialize(project, new JsonSerializerOptions { WriteIndented = true }));
        CurrentProjectPath = projectPath;
        Status = $"Saved project to {Path.GetFileName(projectPath)}.";
    }

    public void LoadProject(string projectPath)
    {
        var project = JsonSerializer.Deserialize<DesktopProject>(File.ReadAllText(projectPath)) ?? new DesktopProject([]);
        ClearImages();
        AddImages(project.Images);
        CurrentProjectPath = projectPath;
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
}

public record ImageThumbnailViewModel(string FilePath, string FileName, Bitmap Thumbnail);

public record DesktopProject(List<string> Images);
