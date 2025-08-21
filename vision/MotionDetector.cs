using OpenCvSharp;

namespace MotionAccessSystem.Vision;

public sealed class MotionDetector : IDisposable
{
  private readonly BackgroundSubtractorMOG2 _bg;   // MOG2 background model (learns static scene)
  private readonly int _minArea;                   // minimal contour area to consider as motion
  private readonly int _minMotionFrames;           // frames-in-a-row required to assert motion
  private readonly int _cooldownFrames;            // frames to ignore after a trigger (debounce)
  private int _streak = 0;                         // consecutive frames with motion
  private int _cooldown = 0;                       // remaining cooldown frames

  public MotionDetector(int history = 500, int varThreshold = 16, bool detectShadows = true,
                        int minArea = 500, int minMotionFrames = 3, int cooldownFrames = 10)
  {
    _bg = BackgroundSubtractorMOG2.Create(history, varThreshold, detectShadows); // init MOG2
    _minArea = minArea;
    _minMotionFrames = minMotionFrames;
    _cooldownFrames = cooldownFrames;
  }

  public (bool triggered, Mat mask) Detect(Mat frame)
  {
    var mask = new Mat();                 // binary foreground mask (returned to caller)
    _bg.Apply(frame, mask);               // update model + compute foreground

    Cv2.MedianBlur(mask, mask, 5);        // denoise isolated pixels
    Cv2.Threshold(mask, mask, 127, 255, ThresholdTypes.Binary); // force clean 0/255 mask

    var contours = Cv2.FindContoursAsMat(mask, RetrievalModes.External, ContourApproximationModes.ApproxSimple);
                                          // extract blobs of motion

    bool motionNow = false;

    foreach (var contour in contours)
    {
      double area = Cv2.ContourArea(contour);
      if (area >= _minArea)               // accept only sufficiently large blobs
      {
        motionNow = true;
        break;
      }
    }

    foreach (var c in contours) c.Dispose(); // avoid Mat leaks from contours

    if (_cooldown > 0)                   // during cooldown we suppress new triggers
    {
      _cooldown--;
      return (false, mask);
    }

    _streak = motionNow ? _streak + 1 : 0; // track consecutive motion frames

    if (_streak >= _minMotionFrames)     // require persistence to avoid false positives
    {
      _streak = 0;
      _cooldown = _cooldownFrames;       // start debounce window
      return (true, mask);               // fire a single motion event
    }

    return (false, mask);                // no trigger this frame
  }

  public void Dispose()
  {
    _bg?.Dispose();                      // release native OpenCV resources
    GC.SuppressFinalize(this);           // standard Dispose pattern optimization
  }
}
