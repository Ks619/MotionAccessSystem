using MotionAccessSystem.Capture;
using MotionAccessSystem.Vision;
using OpenCvSharp;
using System.Diagnostics;

namespace MotionAccessSystem
{
    class Program
    {
        public static void Main()
        {
            Console.WriteLine("Starting Motion Access System...");
            Console.WriteLine("Press 'q' in the video window to quit.\n");

            using var cam = new CameraStream(index: 0, width: 1280, height: 720);

            double actualW   = cam.Get(VideoCaptureProperties.FrameWidth);
            double actualH   = cam.Get(VideoCaptureProperties.FrameHeight);
            double actualFps = cam.Get(VideoCaptureProperties.Fps);

            Console.WriteLine($"Camera resolution: {actualW}x{actualH}, FPS: {actualFps}");

            const string WIN = "Motion Access System";
            const int MOTION_HOLD_FRAMES = 20;
            const int FACE_HOLD_FRAMES = 30;
            const int FACE_STICKY_MIN = 20;
            int motionHold = 0;
            int faceHold = 0;
            List<Rect> cachedFaces = new List<Rect>();

            Cv2.NamedWindow(WIN, WindowFlags.Normal);

            var stopwatch = new Stopwatch();
            stopwatch.Start();

            using var motionDetecor = new MotionDetector();
            string cascadePath = Path.Combine("cascades", "haarcascade_frontalface_default.xml");
            using var faceDetector = new FaceDetector(cascadePath, minSize: 80);

            int frames = 0;
            double lastSec = 0.0;
            string fpsText = "FPS ~ --";
            
            while (true)
            {
                using var frame = cam.ReadFrame();
                var display = frame;

                var (triggered, mask) = motionDetecor.Detect(frame);
                mask.Dispose();
      
                
                if (triggered)
                  motionHold = MOTION_HOLD_FRAMES;
                
                bool motionActive = triggered || motionHold > 0;
                if (motionHold > 0)
                    motionHold--;

                int faceCount = 0;
                if (motionActive)
                {
                  var faces = faceDetector.Detect(display);
                  faceCount = faces.Count;
                  if (faceCount > 0)
                  {
                    cachedFaces = faces;
                    faceHold = FACE_HOLD_FRAMES;
                  }

                  else
                  {
                    if (cachedFaces.Count > 0)
                    {
                      if (faceHold > FACE_STICKY_MIN)
                        faceHold--;
                      else
                        faceHold = FACE_STICKY_MIN;
                    }
                    else if (faceHold > 0)
                      faceHold--;
                  }

                  if (cachedFaces.Count > 0 && faceHold > 0)
                  {
                    faceCount = cachedFaces.Count;

                    foreach (var face in cachedFaces)
                      Cv2.Rectangle(display, face, Scalar.LimeGreen, 2);
                  }

                  else
                    faceCount = 0;
                }
                          
                else
                {
                  if (faceHold > 0)
                    faceHold--;

                  if (cachedFaces.Count > 0 && faceHold > 0)
                  {
                    faceCount = cachedFaces.Count;
                    foreach (var face in cachedFaces)
                      Cv2.Rectangle(display, face, Scalar.LimeGreen, 2);
                  }

                  else
                    faceCount = 0;
                }


                var motionText = motionActive ? "Motion Detected!" : "No Motion";

                Cv2.PutText(display, motionText, new Point(10, 90), HersheyFonts.HersheySimplex, 0.7,motionActive? Scalar.Red : Scalar.Green, 2);

                Cv2.PutText(display, $"Faces: {faceCount}", new Point(10, 120), HersheyFonts.HersheySimplex, 0.7, Scalar.Cyan, 2);

                frames++;
                var totalSec = stopwatch.Elapsed.TotalSeconds;

                if (totalSec - lastSec >= 2.0)
                {
                    var fps = frames / (totalSec - lastSec);
                    fpsText = $"FPS ~ {fps:F1}";
                    frames = 0;
                    lastSec = totalSec;
                }
                Cv2.PutText(display, fpsText, new Point(10, 60),
                        HersheyFonts.HersheySimplex, 0.7, Scalar.White, 2);

                Cv2.PutText(display, $"{WIN} - (press 'q' to quit)",
                    new Point(10, 30), HersheyFonts.HersheySimplex, 0.7, Scalar.White, 2);

                Cv2.ImShow(WIN, display);

                var key = Cv2.WaitKey(1) & 0xFF;
                if (key == 'q')
                {
                    Console.WriteLine("Exiting...");
                    break;
                }
            }
        }
    }
}
