using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using Avalonia.Media.Imaging;

namespace Image2STL.Desktop.ViewModels;

public class MainWindowViewModel : ViewModelBase
{
    public ObservableCollection<ImageThumbnailViewModel> Images { get; } = new();

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
            try
            {
                using var stream = File.OpenRead(filePath);
                var bitmap = Bitmap.DecodeToWidth(stream, 260);
                Images.Add(new ImageThumbnailViewModel(Path.GetFileName(filePath), bitmap));
            }
            catch
            {
                skipped += 1;
            }
        }

        var baseStatus = Images.Count switch
        {
            < 3 => $"Loaded {Images.Count} image(s). Add at least {3 - Images.Count} more.",
            > 5 => $"Loaded {Images.Count} image(s). MVP target is 3-5 images.",
            _ => $"Loaded {Images.Count} image(s). Ready for reconstruction.",
        };
        Status = skipped > 0 ? $"{baseStatus} Skipped {skipped} invalid image file(s)." : baseStatus;
    }
}

public record ImageThumbnailViewModel(string FileName, Bitmap Thumbnail);
