using MotionAccessSystem.Capture;
using OpenCvSharp;


namespace MotionAccessSystem.Vision;

public sealed class MotionDetector : IDisposable
{
  private readonly BackgroundSubtractorMOG2 _bg;
  private readonly int _minArea;
  private readonly int _minMotionFrames;
  private readonly int _cooldownFrames;
  private int _streak = 0;
  private int _cooldown = 0;

  public MotionDetector(int history = 500, int varThreshold = 16, bool detectShadows = true,
                        int minArea = 1500, int minMotionFrames = 3, int cooldownFrames = 10)
  {
    _bg = BackgroundSubtractorMOG2.Create(history, varThreshold, detectShadows);
    _minArea = minArea;
    _minMotionFrames = minMotionFrames;
    _cooldownFrames = cooldownFrames;

  }

  public (bool triggered, Mat mask) Detect(Mat frame)
  {
    var mask = new Mat();
    _bg.Apply(frame, mask);

    Cv2.MedianBlur(mask, mask, 5);
    Cv2.Threshold(mask, mask, 127, 255, ThresholdTypes.Binary);

    var contours = Cv2.FindContoursAsMat(mask, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

    bool motionNow = false;

    foreach (var contour in contours)
    {
      double area = Cv2.ContourArea(contour);
      if (area >= _minArea)
      {
        motionNow = true;
        break;
      }
    }

    // required?
    foreach (var c in contours)
      c.Dispose();

    if (_cooldown > 0)
    {
      _cooldown--;
      return (false, mask);
    }

    if (motionNow)
      _streak++;
    else
      _streak = 0;

    if (_streak >= _minMotionFrames)
    {
      _streak = 0;
      _cooldown = _cooldownFrames;
      return (true, mask);
    }

    return (false, mask);
  }

  public void Dispose()
  {
    _bg?.Dispose();
    GC.SuppressFinalize(this);
  }
}