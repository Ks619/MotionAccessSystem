using MotionAccessSystem.Capture;
using MotionAccessSystem.Vision;          // MotionDetector + YoloFaceDetector
using OpenCvSharp;
using System.Diagnostics;
using System.Collections.Generic;         // List<Rect>

namespace MotionAccessSystem
{
    class Program
    {
        public static void Main()
        {
            Console.WriteLine("Starting Motion Access System...");
            Console.WriteLine("Press 'q' in the video window to quit.\n");

            using var cam = new CameraStream(index: 0, width: 1280, height: 720);   // Open camera

            // Read back real properties (driver may override requested size/FPS)
            double actualW   = cam.Get(VideoCaptureProperties.FrameWidth);
            double actualH   = cam.Get(VideoCaptureProperties.FrameHeight);
            double actualFps = cam.Get(VideoCaptureProperties.Fps);
            Console.WriteLine($"Camera resolution: {actualW}x{actualH}, FPS: {actualFps}");

            const string WIN = "Motion Access System";
            const int MOTION_HOLD_FRAMES = 15; // keep “motion active” a few frames after a trigger
            const int FACE_HOLD_FRAMES   = 10; // keep last face boxes to reduce flicker

            int motionHold = 0, faceHold = 0;  // “hold” state counters
            List<Rect> cachedFaces = new();    // last good detections (drawn while hold > 0)

            Cv2.NamedWindow(WIN, WindowFlags.Normal);    // Create resizable UI window

            var stopwatch = new Stopwatch();   // FPS overlay timer
            stopwatch.Start();

            using var motionDetecor = new MotionDetector();  // background subtraction (disposable)

            // YOLO-Face: make sure the ONNX exists at models\yolov11n-face.onnx
            // (If you prefer a fixed path, use AppContext.BaseDirectory + "models"...)
            string yoloPath = Path.Combine("models", "yolov11n-face.onnx");
            using var faceDetector = new YoloFaceDetector(
                modelPath: yoloPath,
                inputSize: 640,     // 320/416/512/640; tradeoff between speed/accuracy
                confThresh: 0.25f,  // minimum confidence
                iouThresh: 0.45f    // NMS overlap threshold
            );

            int frames = 0;                // rolling frame counter for FPS overlay
            double lastSec = 0.0;
            string fpsText = "FPS ~ --";

            while (true)
            {
                using var frame = cam.ReadFrame();  // grab a frame (Mat is IDisposable)
                var display = frame;                // alias we draw overlays on

                // Motion detection (returns bool + mask)
                var (triggered, mask) = motionDetecor.Detect(frame);
                mask.Dispose();                     // always release the mask Mat

                if (triggered) motionHold = MOTION_HOLD_FRAMES;     // refresh motion hold on trigger
                bool motionActive = triggered || motionHold > 0;     // gate face detection by motion
                if (motionHold > 0) motionHold--;                    // decay when idle

                int faceCount = 0;

                if (motionActive)
                {
                    var faces = faceDetector.Detect(display);        // run YOLO-Face only on motion
                    faceCount = faces.Count;

                    if (faceCount > 0) { cachedFaces = faces; faceHold = FACE_HOLD_FRAMES; }
                    else if (faceHold > 0) faceHold--;               // keep boxes briefly after miss

                    if (cachedFaces.Count > 0 && faceHold > 0)       // draw cached boxes during hold
                    {
                        faceCount = cachedFaces.Count;
                        foreach (var face in cachedFaces)
                            Cv2.Rectangle(display, face, Scalar.LimeGreen, 2);
                    }
                    else faceCount = 0;
                }
                else
                {
                    // No motion: just decay face hold and optionally keep drawing cache
                    if (faceHold > 0) faceHold--;
                    if (cachedFaces.Count > 0 && faceHold > 0)
                    {
                        faceCount = cachedFaces.Count;
                        foreach (var face in cachedFaces)
                            Cv2.Rectangle(display, face, Scalar.LimeGreen, 2);
                    }
                    else faceCount = 0;
                }

                // Text overlays: motion status + face count
                var motionText = motionActive ? "Motion Detected!" : "No Motion";
                Cv2.PutText(display, motionText, new Point(10, 90),
                    HersheyFonts.HersheySimplex, 0.7, motionActive ? Scalar.Red : Scalar.Green, 2);

                Cv2.PutText(display, $"Faces: {faceCount}", new Point(10, 120),
                    HersheyFonts.HersheySimplex, 0.7, Scalar.Cyan, 2);

                // FPS overlay (updated every ~2s for stable numbers)
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

                // Title + quit hint, render, and key handling
                Cv2.PutText(display, $"{WIN} - (press 'q' to quit)",
                    new Point(10, 30), HersheyFonts.HersheySimplex, 0.7, Scalar.White, 2);

                Cv2.ImShow(WIN, display);
                var key = Cv2.WaitKey(1) & 0xFF;
                if (key == 'q') { Console.WriteLine("Exiting..."); break; }
            }
        }
    }
}
