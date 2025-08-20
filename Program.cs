using MotionAccessSystem.Capture;
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

            Cv2.NamedWindow(WIN, WindowFlags.Normal);

            var stopwatch = new Stopwatch();
            stopwatch.Start();

            int frames = 0;
            double lastSec = 0.0;
            string fpsText = "FPS ~ --";
            
            while (true)
            {
                using var frame = cam.ReadFrame();
                var display = frame;

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
