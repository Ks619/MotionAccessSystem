using MotionAccessSystem.Capture;
using MotionAccessSystem.Vision;
using MotionAccessSystem.Logging;
using OpenCvSharp;
using System.Diagnostics;


namespace MotionAccessSystem
{
    class Program
    {
        private static bool IsExitKey(int key)
        {
            int k = key & 0xFF;
            // ESC
            if (k == 27)
                return true;  

            if (k == 'q' || k == 'Q')
                return true;
            
            // Hebrew layout for physical 'Q'
            if (k == '/')
                return true;    

            return false;
        }

        public static void Main()
        {
            Console.WriteLine("Starting Motion Access System...");
            Console.WriteLine("Press 'q' (or ESC) in the video window to quit.\n");

            using var cam = new CameraStream(index: 0, width: 1280, height: 720);

            // first frame to lock real dimensions (driver might report 0x0)
            using var first = cam.ReadFrame();
            int actualW = first.Width;
            int actualH = first.Height;

            double actualFps = cam.Get(VideoCaptureProperties.Fps);
            Console.WriteLine($"Camera resolution: {actualW}x{actualH}, Driver FPS: {actualFps}");

            // recorder & logger setup (Israel TZ)
            int outW = actualW;
            int outH = actualH;
            const double TARGET_FPS_FOR_FILE = 20.0;

            var projectRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));
            var recordingsDir = Path.Combine(projectRoot, "recordings");

            const string ISRAEL_TZ_ID = "Israel Standard Time"; // falls back to Asia/Jerusalem inside

            using var recorder = new FaceRecorder(
                outputDir: recordingsDir,
                targetFps: TARGET_FPS_FOR_FILE,
                width: outW,
                height: outH,
                fourcc: "XVID",
                holdSeconds: 3.0,     
                usePreRoll: false,
                preRollSeconds: 0.0,
                timeZoneId: ISRAEL_TZ_ID
            );

            var logger = new EventLogger(recordingsDir);
            recorder.OnClipClosed = logger.Log;
            

            const string WIN = "Motion Access System";
            const int MOTION_HOLD_FRAMES = 15; 
            const int FACE_HOLD_FRAMES = 10; 

            int motionHold = 0, faceHold = 0;
            var cachedFaces = new List<Rect>();

            Cv2.NamedWindow(WIN, WindowFlags.Normal);

            var stopwatch = new Stopwatch();
            stopwatch.Start();

            using var motionDetecor = new MotionDetector();

            string yoloPath = Path.Combine("models", "yolov11n-face.onnx");
            using var faceDetector = new YoloFaceDetector(
                modelPath: yoloPath,
                inputSize: 640,
                confThresh: 0.25f,
                iouThresh: 0.45f
            );

            int frames = 0;
            double lastSec = 0.0;
            string fpsText = "FPS ~ --";

            while (true)
            {
                using var frame = cam.ReadFrame();
                var display = frame; // draw overlays directly on this Mat

                // motion detection 
                var (triggered, mask) = motionDetecor.Detect(frame);
                mask.Dispose();
                if (triggered)
                    motionHold = MOTION_HOLD_FRAMES;

                bool motionActive = triggered || motionHold > 0;
                if (motionHold > 0)
                    motionHold--;

                // face detection (every frame) 
                var faces = faceDetector.Detect(display);
                int faceCount = faces.Count;

                // update face cache
                if (faceCount > 0)
                {
                    cachedFaces = faces; 
                    faceHold = FACE_HOLD_FRAMES;
                }
                else if (faceHold > 0)
                    faceHold--;

                // draw the green face rectangles 
                if (cachedFaces.Count > 0 && faceHold > 0)
                {
                    foreach (var face in cachedFaces)
                        Cv2.Rectangle(display, face, Scalar.LimeGreen, 2);
                }

                // call update to decide if start recording
                bool faceDetectedNow = faceCount > 0;
                recorder.Update(display, faceDetectedNow);

                // REC indicator (top-right)
                if (recorder.IsRecording)
                {
                    int x = display.Width - 120;
                    int y = 30;
                    Cv2.Circle(display, new Point(x, y - 6), 8, Scalar.Red, -1);
                    Cv2.PutText(display, "REC", new Point(x + 18, y),
                        HersheyFonts.HersheySimplex, 0.7, Scalar.Red, 2);
                }

                // text overlays: motion, faces count, FPS, title
                var motionText = motionActive ? "Motion Detected!" : "No Motion";
                Cv2.PutText(display, motionText, new Point(10, 90),
                    HersheyFonts.HersheySimplex, 0.7, motionActive ? Scalar.Red : Scalar.Green, 2);

                Cv2.PutText(display, $"Faces: {faceCount}", new Point(10, 120),
                    HersheyFonts.HersheySimplex, 0.7, Scalar.Cyan, 2);

                // FPS overlay (updated every ~2s)
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

                Cv2.PutText(display, $"{WIN} - press Q/ESC (Hebrew layout: '/') to quit",
                    new Point(10, 30), HersheyFonts.HersheySimplex, 0.6, Scalar.White, 2);

                Cv2.ImShow(WIN, display);

                int key = Cv2.WaitKey(1);
                try
                {
                    if (Cv2.GetWindowProperty(WIN, WindowPropertyFlags.Visible) < 1)
                    {
                        Console.WriteLine("Window closed. Exiting...");
                        break;
                    }
                }
                catch
                {
                    Console.WriteLine("Window no longer exists. Exiting...");
                    break;
                }

                if (IsExitKey(key))
                {
                    Console.WriteLine("Exit key pressed. Exiting...");
                    break;
                }
            }
        }
    }
}
