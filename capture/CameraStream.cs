using OpenCvSharp;

namespace MotionAccessSystem.Capture;

public sealed class CameraStream : IDisposable
{
  private readonly VideoCapture _cap;

  public CameraStream(int index = 0, int width = 1280, int height = 720)
  {
    // Windows DirectShow API
    _cap = new VideoCapture(index, VideoCaptureAPIs.DSHOW);

    // Get resolution
    _cap.Set(VideoCaptureProperties.FrameWidth, width);
    _cap.Set(VideoCaptureProperties.FrameHeight, height);

    // Open the camera
    if(!_cap.IsOpened())
      throw new Exception($"Failed to open camera with index {index}");
    
  }

  public Mat ReadFrame()
  {
    var frame = new Mat();
    if(!_cap.Read(frame))
    {
      frame.Dispose();
      throw new Exception("Failed to read frame from camera");
    }
    return frame;
  }  

  public double Get(VideoCaptureProperties prop) => _cap.Get(prop);

  public void Dispose() => _cap?.Dispose();
}