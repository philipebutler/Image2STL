using System.Collections.Generic;
using System.Linq;
using Avalonia.Controls;
using Avalonia.Input;
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
                    Patterns = ["*.jpg", "*.jpeg", "*.png", "*.heic"],
                },
            ],
        });

        AddPickedFiles(files);
    }

    private void OnDragOver(object? sender, DragEventArgs e)
    {
        e.DragEffects = e.DataTransfer.Contains(DataFormat.File) ? DragDropEffects.Copy : DragDropEffects.None;
        e.Handled = true;
    }

    private void OnDrop(object? sender, DragEventArgs e)
    {
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
}
