using System.Collections.Generic;
using System.IO;
using System.Linq;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.Platform.Storage;
using Image2STL.Desktop.ViewModels;

namespace Image2STL.Desktop.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        DragDrop.SetAllowDrop(this, true);
        AddHandler(DragDrop.DragOverEvent, OnDragOver);
        AddHandler(DragDrop.DragLeaveEvent, OnDragLeave);
        AddHandler(DragDrop.DropEvent, OnDrop);
    }

    private async void AddImages_Click(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var topLevel = TopLevel.GetTopLevel(this);
        if (topLevel?.StorageProvider is null)
        {
            return;
        }

        var files = await topLevel.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            AllowMultiple = true,
            Title = "Select source images",
            FileTypeFilter =
            [
                new FilePickerFileType("Supported Images")
                {
                    Patterns = ["*.jpg", "*.jpeg", "*.png", "*.heic", "*.heif", "*.webp", "*.avif"],
                },
            ],
        });

        AddPickedFiles(files);
    }

    private void NewProject_Click(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is MainWindowViewModel vm)
        {
            vm.NewProject();
        }
    }

    private async void OpenProject_Click(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var topLevel = TopLevel.GetTopLevel(this);
        if (topLevel?.StorageProvider is null || DataContext is not MainWindowViewModel vm)
        {
            return;
        }

        var files = await topLevel.StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            AllowMultiple = false,
            Title = "Open project",
            FileTypeFilter = [ProjectFileType],
        });

        var projectPath = files.FirstOrDefault()?.TryGetLocalPath();
        if (!string.IsNullOrWhiteSpace(projectPath) && File.Exists(projectPath))
        {
            vm.LoadProject(projectPath);
        }
    }

    private void SaveProject_Click(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainWindowViewModel vm)
        {
            return;
        }

        if (!string.IsNullOrWhiteSpace(vm.CurrentProjectPath))
        {
            vm.SaveProject(vm.CurrentProjectPath);
            return;
        }

        SaveProjectAs_Click(sender, e);
    }

    private async void SaveProjectAs_Click(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var topLevel = TopLevel.GetTopLevel(this);
        if (topLevel?.StorageProvider is null || DataContext is not MainWindowViewModel vm)
        {
            return;
        }

        var file = await topLevel.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Save project",
            SuggestedFileName = "project.i2sproj",
            FileTypeChoices = [ProjectFileType],
        });

        var projectPath = file?.TryGetLocalPath();
        if (!string.IsNullOrWhiteSpace(projectPath))
        {
            vm.SaveProject(projectPath);
        }
    }

    private void Generate_Click(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is MainWindowViewModel vm)
        {
            vm.StartGeneration();
        }
    }

    private async void ExportSTL_Click(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var topLevel = TopLevel.GetTopLevel(this);
        if (topLevel?.StorageProvider is null || DataContext is not MainWindowViewModel vm)
        {
            return;
        }

        var file = await topLevel.StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Export STL",
            SuggestedFileName = vm.ExportFileName,
            FileTypeChoices =
            [
                new FilePickerFileType("STL Files") { Patterns = ["*.stl"] },
            ],
        });

        var exportPath = file?.TryGetLocalPath();
        if (!string.IsNullOrWhiteSpace(exportPath))
        {
            vm.ExportSTL(exportPath);
        }
    }

    private void Cancel_Click(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is MainWindowViewModel vm)
        {
            vm.CancelOperation();
        }
    }

    private void OnDragOver(object? sender, DragEventArgs e)
    {
        var hasFiles = e.DataTransfer.Contains(DataFormat.File);
        e.DragEffects = hasFiles ? DragDropEffects.Copy : DragDropEffects.None;
        if (hasFiles)
        {
            DropZone.BorderBrush = new SolidColorBrush(Color.FromRgb(0, 120, 215));
        }
        e.Handled = true;
    }

    private void OnDragLeave(object? sender, DragEventArgs e)
    {
        DropZone.BorderBrush = Brushes.Transparent;
        e.Handled = true;
    }

    private void OnDrop(object? sender, DragEventArgs e)
    {
        DropZone.BorderBrush = Brushes.Transparent;
        AddPickedFiles(e.DataTransfer.TryGetFiles() ?? []);
        e.Handled = true;
    }

    private void AddPickedFiles(IEnumerable<IStorageItem> files)
    {
        if (DataContext is not MainWindowViewModel vm)
        {
            return;
        }

        vm.AddImages(files.Select(file => file.TryGetLocalPath()).OfType<string>());
    }

    private void RemoveImage_Click(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        if (DataContext is not MainWindowViewModel vm || sender is not Button { Tag: string filePath })
        {
            return;
        }

        vm.RemoveImage(filePath);
    }

    private static readonly FilePickerFileType ProjectFileType = new("Image2STL Project")
    {
        Patterns = ["*.i2sproj", "*.json"],
    };
}
