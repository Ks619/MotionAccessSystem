using OpenCvSharp;


namespace MotionAccessSystem.Vision;


public sealed class FaceDetector : IDisposable
{
  private readonly CascadeClassifier _cascade;
  public int MinSize { get; }

  public FaceDetector(string cascadePath, int minSize = 80)
  {
    if (!File.Exists(cascadePath))
      throw new FileNotFoundException("Cascade file not found", cascadePath);

    _cascade = new CascadeClassifier(cascadePath);
    MinSize = minSize;
  }

  public List<Rect> Detect(Mat frame)
  {
    using var gray = new Mat();
    Cv2.CvtColor(frame, gray, ColorConversionCodes.BGR2GRAY);
    Cv2.EqualizeHist(gray, gray);

    var faces = _cascade.DetectMultiScale(image: gray, scaleFactor: 1.2, minNeighbors: 3,
                                              flags: HaarDetectionTypes.ScaleImage, minSize: new Size(MinSize, MinSize));
    return faces.ToList();
  }

  public void Dispose()
  {
    _cascade?.Dispose();
    GC.SuppressFinalize(this);
  }
}