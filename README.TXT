Motion Access System
====================

Overview
--------
This project is a real-time motion and face detection system built in C# (.NET 8.0) 
with OpenCvSharp and ONNX Runtime. It uses the computer's camera stream to detect 
motion and then runs a YOLO-Face model (onnx) only when motion is triggered.

Key Features
------------
1. CameraStream.cs
   - Opens and manages the camera stream (VideoCapture).
   - Ensures correct resolution and FPS.
   - Provides frames safely with error handling.

2. MotionDetector.cs
   - Detects motion using BackgroundSubtractorMOG2.
   - Filters out noise with minArea and streak/cooldown logic.
   - Returns both a trigger flag and a motion mask.

3. YoloFaceDetector.cs
   - Runs YOLO-Face ONNX model for face detection.
   - Configurable input size, confidence threshold, and IoU threshold.
   - Performs Non-Maximum Suppression (NMS) to reduce duplicates.

4. Program.cs
   - Main entry point.
   - Initializes camera and detectors.
   - Waits for motion before running face detection (to save CPU/GPU).
   - Draws rectangles and labels for detected faces on screen.
   - Allows quitting with 'q' or ESC.

5. Models and Cascades
   - models/yolov11n-face.onnx : The YOLO-Face model used for detection.
   - cascades/haarcascade_frontalface_default.xml : Legacy Haar face detector 
     (kept for reference).

6. Project File (.csproj)
   - References OpenCvSharp4 and OnnxRuntime.
   - Configures copying the ONNX model to the output directory.

Requirements
------------
- .NET 8.0 SDK
- OpenCvSharp4
- Microsoft.ML.OnnxRuntime
- Windows with camera access

How to Build & Run
------------------
1. Clone the repository:
   git clone https://github.com/Ks619/MotionAccessSystem.git

2. Build the project:
   dotnet build -c Release

3. Run the program:
   dotnet run -c Release

   - Press 'q' or ESC in the video window to quit.

Notes
-----
- The system runs YOLO-Face detection only when motion is detected 
  to improve performance.
- Parameters such as motion area, hold frames, and YOLO thresholds 
  can be adjusted in Program.cs.
- Output is displayed in a live OpenCV window.

Author
------
Developed by Ks619
