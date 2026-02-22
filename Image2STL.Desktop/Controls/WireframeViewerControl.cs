using System;
using System.Numerics;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;

namespace Image2STL.Desktop.Controls;

public class WireframeViewerControl : Control
{
    private static readonly (int A, int B)[] Edges =
    {
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    };

    private static readonly Vector3[] Cube =
    {
        new(-1, -1, -1), new(1, -1, -1), new(1, 1, -1), new(-1, 1, -1),
        new(-1, -1, 1), new(1, -1, 1), new(1, 1, 1), new(-1, 1, 1),
    };

    private bool _isDragging;
    private Point _lastPoint;
    private float _yaw = 0.5f;
    private float _pitch = 0.3f;
    private float _zoom = 1f;

    public WireframeViewerControl()
    {
        Focusable = true;
        ClipToBounds = true;
    }

    public override void Render(DrawingContext context)
    {
        base.Render(context);
        context.FillRectangle(Brushes.Black, Bounds);

        if (Bounds.Width <= 1 || Bounds.Height <= 1)
        {
            return;
        }

        var projected = new Point[Cube.Length];
        var center = new Point(Bounds.Width / 2, Bounds.Height / 2);
        var perspectiveScale = Math.Min(Bounds.Width, Bounds.Height) * 0.33f * _zoom;

        var yawRotation = Matrix4x4.CreateRotationY(_yaw);
        var pitchRotation = Matrix4x4.CreateRotationX(_pitch);
        var rotation = pitchRotation * yawRotation;

        for (var i = 0; i < Cube.Length; i++)
        {
            var vertex = Vector3.Transform(Cube[i], rotation);
            var z = vertex.Z + 4f;
            var scale = perspectiveScale / z;
            projected[i] = new Point(center.X + (vertex.X * scale), center.Y - (vertex.Y * scale));
        }

        var pen = new Pen(Brushes.DeepSkyBlue, 2);
        foreach (var edge in Edges)
        {
            context.DrawLine(pen, projected[edge.A], projected[edge.B]);
        }
    }

    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        if (e.GetCurrentPoint(this).Properties.IsLeftButtonPressed)
        {
            _isDragging = true;
            _lastPoint = e.GetPosition(this);
            e.Pointer.Capture(this);
            e.Handled = true;
        }
    }

    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        base.OnPointerReleased(e);
        _isDragging = false;
        e.Pointer.Capture(null);
    }

    protected override void OnPointerMoved(PointerEventArgs e)
    {
        base.OnPointerMoved(e);
        if (!_isDragging)
        {
            return;
        }

        var current = e.GetPosition(this);
        var delta = current - _lastPoint;
        _lastPoint = current;
        _yaw += (float)(delta.X * 0.01);
        _pitch += (float)(delta.Y * 0.01);
        InvalidateVisual();
    }

    protected override void OnPointerWheelChanged(PointerWheelEventArgs e)
    {
        base.OnPointerWheelChanged(e);
        _zoom = Math.Clamp(_zoom + (float)(e.Delta.Y * 0.1), 0.4f, 4f);
        InvalidateVisual();
    }
}
