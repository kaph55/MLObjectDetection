using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Win32;
using System;
using System.IO;
using IOPath = System.IO.Path;
using System.Linq;
using ObjectConsole;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Windows.Controls;
using System.Collections;
using MLModel1_ConsoleApp1;

namespace ObjectConsole.Wpf
{
    public partial class MainWindow : Window
    {
        private readonly string _modelRelativePath = "Models\\MLModel1.mlnet";
        private readonly MLContext _mlContext;
        // model loader not needed when using generated MLModel1.Predict API
        private byte[]? _currentImageBytes;
        private string? _currentImagePath;

        // Controls (resolved at runtime with FindName in case generated fields are not present)
        private System.Windows.Controls.TextBox? SummaryTextBox;
        private System.Windows.Controls.ListBox? PredictionsListBox;
        private System.Windows.Controls.TextBlock? ConfidenceValueText;
        private System.Windows.Controls.Slider? ConfidenceSliderControl;
        private System.Windows.Controls.TextBox? OutputTextBox;
        private System.Windows.Controls.TextBlock? StatusTextBlock;
        private System.Windows.Controls.Image? PreviewImageControl;
        private System.Windows.Controls.Canvas? OverlayCanvasControl;
        // cache brushes per label
        private readonly System.Collections.Generic.Dictionary<string, SolidColorBrush> _labelBrushes = new();
        // last prediction results so slider can re-filter without re-running model
        private float[] _lastBoxes = Array.Empty<float>();
        private float[] _lastScores = Array.Empty<float>();
        private string[] _lastLabels = Array.Empty<string>();

        public MainWindow()
        {
            InitializeComponent();
            // Resolve named controls in XAML (use FindName to avoid missing generated fields)
            SummaryTextBox = FindName("SummaryText") as System.Windows.Controls.TextBox;
            PredictionsListBox = FindName("PredictionsList") as System.Windows.Controls.ListBox;
            ConfidenceValueText = FindName("ConfidenceValue") as System.Windows.Controls.TextBlock;
            ConfidenceSliderControl = FindName("ConfidenceSlider") as System.Windows.Controls.Slider;
            OutputTextBox = FindName("OutputText") as System.Windows.Controls.TextBox;
            StatusTextBlock = FindName("StatusText") as System.Windows.Controls.TextBlock;
            PreviewImageControl = FindName("PreviewImage") as System.Windows.Controls.Image;
            OverlayCanvasControl = FindName("OverlayCanvas") as System.Windows.Controls.Canvas;

            // Hook slider event if needed
            if (ConfidenceSliderControl != null)
                ConfidenceSliderControl.ValueChanged += ConfidenceSlider_ValueChanged;
            // Set initial text
            if (StatusTextBlock != null) StatusTextBlock.Text = "Ready";
            _mlContext = new MLContext();
            
        }

        private void EnsureOverlayMatchesImage()
        {
            if (PreviewImageControl == null || OverlayCanvasControl == null) return;
            if (!(PreviewImageControl.Source is BitmapSource bmp)) return;

            // Force layout update then set overlay size
            PreviewImageControl.UpdateLayout();
            var w = PreviewImageControl.ActualWidth;
            var h = PreviewImageControl.ActualHeight;
            if (w > 0 && h > 0)
            {
                OverlayCanvasControl.Width = w;
                OverlayCanvasControl.Height = h;
            }
        }

        private void ConfidenceSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (ConfidenceValueText != null) ConfidenceValueText.Text = e.NewValue.ToString("0.00");
            // If we have previous predictions, update the overlay immediately
            if (_lastBoxes.Length >= 4 && _lastScores.Length >= 1)
            {
                EnsureOverlayMatchesImage();
                DrawBoxes(_lastBoxes, _lastScores, (float)e.NewValue, _lastLabels.Length > 0 ? _lastLabels : null);
            }
        }

        private void OpenButton_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog
            {
                Filter = "Images|*.png;*.jpg;*.jpeg;*.bmp|All files|*.*"
            };

            if (dlg.ShowDialog() == true)
            {
                var path = dlg.FileName;
                LoadPreview(path);
                _currentImageBytes = File.ReadAllBytes(path);
                _currentImagePath = path;
                if (StatusTextBlock != null) StatusTextBlock.Text = $"Loaded {IOPath.GetFileName(path)}";
                if (OutputTextBox != null) OutputTextBox.Text = "";
                OverlayCanvasControl?.Children.Clear();
            }
        }

        private void LoadPreview(string path)
        {
            var bmp = new BitmapImage();
            bmp.BeginInit();
            bmp.CacheOption = BitmapCacheOption.OnLoad;
            bmp.UriSource = new Uri(path);
            bmp.EndInit();
            if (PreviewImageControl != null) PreviewImageControl.Source = bmp;

            // After layout update, size the overlay canvas to match the displayed image size
            Dispatcher.BeginInvoke(new Action(() =>
            {
                if (PreviewImageControl != null && OverlayCanvasControl != null)
                {
                    // Use the rendered size (ActualWidth/ActualHeight) but clamp to MaxWidth/MaxHeight if set
                    double w = PreviewImageControl.ActualWidth;
                    double h = PreviewImageControl.ActualHeight;
                    if (double.IsNaN(w) || w <= 0) w = PreviewImageControl.Width;
                    if (double.IsNaN(h) || h <= 0) h = PreviewImageControl.Height;
                    OverlayCanvasControl.Width = Math.Min(w, PreviewImageControl.MaxWidth > 0 ? PreviewImageControl.MaxWidth : w);
                    OverlayCanvasControl.Height = Math.Min(h, PreviewImageControl.MaxHeight > 0 ? PreviewImageControl.MaxHeight : h);
                }
            }));
        }

        private void DetectButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                if (string.IsNullOrEmpty(_currentImagePath))
                {
                    MessageBox.Show("Open an image first.", "No image", MessageBoxButton.OK, MessageBoxImage.Information);
                    return;
                }

                // Use generated consumption API to run prediction (ensures required input columns like "Labels")
                var modelInput = new MLModel1.ModelInput
                {
                    Labels = new[] { "unknown" },
                    Image = MLImage.CreateFromFile(_currentImagePath),
                    Box = Array.Empty<float>()
                };

                var output = MLModel1.Predict(modelInput);

                if (OutputTextBox != null)
                {
                    OutputTextBox.Text = "Predictions:" + Environment.NewLine;
                    if (output.PredictedBoundingBoxes != null)
                        OutputTextBox.AppendText($"Boxes: {output.PredictedBoundingBoxes.Length}\n");
                    if (output.Score != null)
                        OutputTextBox.AppendText($"Scores: {output.Score.Length}\n");
                }

                var boxes = output.PredictedBoundingBoxes ?? Array.Empty<float>();
                var scores = output.Score ?? Array.Empty<float>();
                var labels = output.PredictedLabel ?? Array.Empty<string>();

                if (boxes.Length >= 4 && scores.Length >= 1)
                {
                    if (SummaryTextBox != null) SummaryTextBox.Text = $"Antal fundet: {boxes.Length / 4}";
                    if (PredictionsListBox != null)
                    {
                        PredictionsListBox.Items.Clear();
                        for (int i = 0; i < scores.Length; i++)
                            PredictionsListBox.Items.Add($"Score: {scores[i]:0.00} | Box #{i}");
                    }

                    var threshold = ConfidenceSliderControl?.Value ?? 0.3;
                    // Ensure overlay canvas is sized to the rendered image before drawing
                    EnsureOverlayMatchesImage();
                    // cache results so slider updates re-filter without running the model again
                    _lastBoxes = boxes;
                    _lastScores = scores;
                    _lastLabels = labels;
                    // update predictions list with label info
                    if (PredictionsListBox != null)
                    {
                        PredictionsListBox.Items.Clear();
                        for (int i = 0; i < scores.Length; i++)
                        {
                            var lab = (i < labels.Length) ? labels[i] ?? "" : "";
                            PredictionsListBox.Items.Add($"Accuracy: {scores[i] * 100:0.0}% | {lab} | Box #{i}");

                        }
                    }
                    DrawBoxes(boxes, scores, (float)threshold, labels);
                }

                if (StatusTextBlock != null) StatusTextBlock.Text = "Detektion færdig.";
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.ToString(), "Error", MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        public class ImageInput
        {
            public byte[] Image { get; set; } = Array.Empty<byte>();
        }

        private void DrawBoxes(float[] boxes, float[] scores, float scoreThreshold = 0.3f, string[]? labels = null)
        {
            // Boxes: [xTop, yTop, xBottom, yBottom, ...]
            if (OverlayCanvasControl == null)
                return;
            OverlayCanvasControl.Children.Clear();

            if (PreviewImageControl == null) return;
            if (!(PreviewImageControl.Source is BitmapSource bmp)) return;
            // Convert image pixel size to WPF device-independent units (DIPs) using DPI
            double imgWidth = bmp.PixelWidth * (96.0 / (bmp.DpiX > 0 ? bmp.DpiX : 96.0));
            double imgHeight = bmp.PixelHeight * (96.0 / (bmp.DpiY > 0 ? bmp.DpiY : 96.0));

            // Use overlay actual size (DIPs) as the container for rendering calculations
            double containerW = OverlayCanvasControl.ActualWidth > 0 ? OverlayCanvasControl.ActualWidth : PreviewImageControl.ActualWidth;
            double containerH = OverlayCanvasControl.ActualHeight > 0 ? OverlayCanvasControl.ActualHeight : PreviewImageControl.ActualHeight;
            if (containerW <= 0 || containerH <= 0)
            {
                // fallback: set overlay to image pixel size
                OverlayCanvasControl.Width = imgWidth;
                OverlayCanvasControl.Height = imgHeight;
                containerW = imgWidth;
                containerH = imgHeight;
            }
            // Calculate rendered image size (Uniform stretch)
            double scale = Math.Min(containerW / imgWidth, containerH / imgHeight);
            double renderW = imgWidth * scale;
            double renderH = imgHeight * scale;
            double offsetX = (containerW - renderW) / 2.0;
            double offsetY = (containerH - renderH) / 2.0;

            int boxCount = boxes.Length / 4;
            for (int i = 0; i < boxCount; i++)
            {
                int baseIdx = i * 4;
                float xTop = boxes[baseIdx + 0];
                float yTop = boxes[baseIdx + 1];
                float xBottom = boxes[baseIdx + 2];
                float yBottom = boxes[baseIdx + 3];

                float score = (i < scores.Length) ? scores[i] : 1f;
                if (score < scoreThreshold) continue;

                // Determine if boxes are normalized (0..1) or in pixels
                bool normalized = (Math.Abs(xTop) <= 1.01 && Math.Abs(yTop) <= 1.01 && Math.Abs(xBottom) <= 1.01 && Math.Abs(yBottom) <= 1.01);

                double xTopPx = normalized ? xTop * imgWidth : xTop;
                double yTopPx = normalized ? yTop * imgHeight : yTop;
                double xBottomPx = normalized ? xBottom * imgWidth : xBottom;
                double yBottomPx = normalized ? yBottom * imgHeight : yBottom;

                double left = offsetX + xTopPx * scale;
                double top = offsetY + yTopPx * scale;
                double width = Math.Max(1, (xBottomPx - xTopPx) * scale);
                double height = Math.Max(1, (yBottomPx - yTopPx) * scale);

                // determine brush for this box (by label if available)
                string label = (labels != null && i < labels.Length) ? labels[i] ?? string.Empty : string.Empty;
                var brush = GetBrushForLabel(label);

                var rect = new Rectangle
                {
                    Stroke = brush,
                    StrokeThickness = 2,
                    Width = width,
                    Height = height,
                    RadiusX = 4,
                    RadiusY = 4
                };

                System.Windows.Controls.Canvas.SetLeft(rect, left);
                System.Windows.Controls.Canvas.SetTop(rect, top);
                OverlayCanvasControl.Children.Add(rect);

                var labelText = new System.Windows.Controls.TextBlock
                {
                    Text = (label.Length > 0) ? $"{label} {score * 100:0.0}%" : $"{score * 100:0.0}%",
                    Foreground = Brushes.Yellow,
                    Background = brush,
                    FontSize = 12,
                    Padding = new Thickness(2)
                };
                System.Windows.Controls.Canvas.SetLeft(labelText, left);
                System.Windows.Controls.Canvas.SetTop(labelText, Math.Max(0, top - 18));
                OverlayCanvasControl.Children.Add(labelText);
            }
        }

        private SolidColorBrush GetBrushForLabel(string label)
        {
            if (string.IsNullOrEmpty(label)) return Brushes.Red;
            if (_labelBrushes.TryGetValue(label, out var b)) return b;
            // pick a deterministic color from label hash
            var rnd = new Random(label.GetHashCode());
            var color = Color.FromRgb((byte)rnd.Next(64, 224), (byte)rnd.Next(64, 224), (byte)rnd.Next(64, 224));
            var brush = new SolidColorBrush(color);
            brush.Freeze();
            _labelBrushes[label] = brush;
            return brush;
        }
    }
}