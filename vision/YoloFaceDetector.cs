using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace MotionAccessSystem.Vision
{
    public sealed class YoloFaceDetector : IDisposable
    {
        private readonly InferenceSession _session;   // ONNX Runtime session (holds the model)
        private readonly int _input;                  // square input side (e.g., 640)
        private readonly string _inputName;           // ONNX input tensor name (e.g., "images")
        private readonly string _outputName;          // ONNX output tensor name (e.g., "output0")
        private readonly float _confThresh;           // confidence threshold before NMS - Non-Max Suppression
        private readonly float _iouThresh;            // IoU - Intersection-over-Union threshold used in NMS
        private bool _disposed;

        public YoloFaceDetector(
            string modelPath,
            int inputSize = 640,
            string inputName = "images",
            string outputName = "output0",
            float confThresh = 0.25f,
            float iouThresh = 0.45f)
        {
            if (!File.Exists(modelPath))
                throw new FileNotFoundException("YOLO-Face ONNX not found", modelPath); 

            _session = new InferenceSession(modelPath); // load model into ORT - ONNX Runtime
            _input = inputSize;
            _inputName = inputName;
            _outputName = outputName;
            _confThresh = confThresh;
            _iouThresh = iouThresh;
        }

        public List<Rect> Detect(Mat frameBgr)
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(YoloFaceDetector)); 

            int origW = frameBgr.Width;
            int origH = frameBgr.Height;

            // convert frame to square
            float r = Math.Min(_input / (float)origW, _input / (float)origH); // scale ratio
            int newW = (int)Math.Round(origW * r);
            int newH = (int)Math.Round(origH * r);
            int dw = (_input - newW) / 2; // horizontal padding
            int dh = (_input - newH) / 2; // vertical padding

            using var resized = new Mat();
            Cv2.Resize(frameBgr, resized, new Size(newW, newH), 0, 0, InterpolationFlags.Area);

            using var padded = new Mat();
            Cv2.CopyMakeBorder(                                // add gray padding = 114,114,114
                resized, padded, dh, _input - newH - dh, dw, _input - newW - dw,
                BorderTypes.Constant, new Scalar(114,114,114));

            // build input tensor NCHW float32 in RGB, normalized to [0..1] 
            var input = new DenseTensor<float>(new[] { 1, 3, _input, _input });
            for (int y = 0; y < _input; y++)
                for (int x = 0; x < _input; x++)
                {
                    var bgr = padded.At<Vec3b>(y, x);
                    // [Batch, Channels, Height, Width]
                    input[0, 0, y, x] = bgr.Item2 / 255f; // R
                    input[0, 1, y, x] = bgr.Item1 / 255f; // G
                    input[0, 2, y, x] = bgr.Item0 / 255f; // B
                }

            // run inference
            using var results = _session.Run(new[]
            {
                NamedOnnxValue.CreateFromTensor(_inputName, input) // must match model's input name
            });

            var t = results.First(r => r.Name == _outputName).AsTensor<float>(); // get YOLO output
            var dims = t.Dimensions.ToArray();

            // accept both shapes [1,no,N] or [N,no]
            // batch 1, components per detection (no), N detections
            float[] data = t.ToArray();

            int no, N;
            bool layoutNoFirst;
            if (dims.Length == 3)         // [1, no, N]
            {
                no = dims[1];
                N  = dims[2];
                layoutNoFirst = true;     // "no" dimension comes before "N"
            }
            else if (dims.Length == 2)    // [N, no]
            {
                N  = dims[0];
                no = dims[1];
                layoutNoFirst = false;
            }
            else
            {
                // fallback: best-effort guess, prevents hard crash on unusual shapes 
                no = 6;
                N  = data.Length / no;
                layoutNoFirst = false;
            }

            var dets = new List<(float score, Rect rect)>(N); // raw detections before NMS - Non-Max Suppression

            // Helper to read a single value given anchor index and component index
            float Read(int a, int c) =>
                layoutNoFirst ? data[c * N + a] : data[a * no + c];

            for (int a = 0; a < N; a++)
            {
                float cx  = Read(a, 0);   // center-x 
                float cy  = Read(a, 1);   // center-y
                float w   = Read(a, 2);   // width
                float h   = Read(a, 3);   // height
                float obj = Read(a, 4);   // objectness (faceness)

                // Model returns multiple classes after obj (person/car/dog/face/...)
                // use the max class probability since each detection corresponds to its most likely class.
                float clsMax = 1f;
                if (no > 5)
                {
                    clsMax = float.MinValue;
                    for (int c = 5; c < no; c++)
                        clsMax = Math.Max(clsMax, Read(a, c));
                }
                float score = (no > 5) ? obj * clsMax : obj;
                if (score < _confThresh) continue; // confidence gate

                // convert from YOLO center-based box (cx,cy,w,h) to top-left corner format
                float x = cx - w / 2f;
                float y = cy - h / 2f;

                // map box from converted image back to original image by removing padding (dw,dh) and dividing by resize ratio r
                x = (x - dw) / r;
                y = (y - dh) / r;
                w =  w / r;
                h =  h / r;

                // set valid image region 
                int xi = (int)Math.Max(0, Math.Floor(x));
                int yi = (int)Math.Max(0, Math.Floor(y));
                int wi = (int)Math.Max(0, Math.Floor(w));
                int hi = (int)Math.Max(0, Math.Floor(h));

                if (wi >= 4 && hi >= 4 && xi < origW && yi < origH) // discard tiny or out-of-bounds detections
                {
                    // Ensure box width/height do not exceed image bounds by clipping to (origW, origH)
                    wi = Math.Min(wi, origW - xi);
                    hi = Math.Min(hi, origH - yi);
                    if (wi > 0 && hi > 0)
                        // Add detection to list
                        dets.Add((score, new Rect(xi, yi, wi, hi)));
                }
            }

            // Non-Max Suppression to remove overlapping boxes
            var kept = Nms(dets, _iouThresh);
            return kept.Select(k => k.rect).ToList();
        }

        public void Dispose()
        {
            if (_disposed)
                return;

            _session.Dispose();           
            _disposed = true;
            GC.SuppressFinalize(this);    
        }

        // -------- Helpers --------

        // Greedy NMS: keep highest score, drop boxes with IoU >= threshold
        private static List<(float score, Rect rect)> Nms(List<(float score, Rect rect)> dets, float iouThresh)
        {
            var sorted = dets.OrderByDescending(d => d.score).ToList();
            var keep = new List<(float score, Rect rect)>();
            while (sorted.Count > 0)
            {
                var best = sorted[0];
                keep.Add(best);
                sorted.RemoveAt(0);
                sorted = sorted.Where(d => IoU(best.rect, d.rect) < iouThresh).ToList();
            }
            return keep;
        }

        // Intersection-over-Union between two rectangles (0..1)
        private static float IoU(Rect a, Rect b)
        {
            // compute top-left (x1,y1) and bottom-right (x2,y2) of the overlap area between rectangles a and b
            int x1 = Math.Max(a.X, b.X);
            int y1 = Math.Max(a.Y, b.Y);
            int x2 = Math.Min(a.Right, b.Right);
            int y2 = Math.Min(a.Bottom, b.Bottom);

            int w = Math.Max(0, x2 - x1);
            int h = Math.Max(0, y2 - y1);
            int inter = w * h;
            // compute IoU: ratio of intersection area to union area of rectangles a and b
            int union = a.Width * a.Height + b.Width * b.Height - inter;

            if (union <= 0)
                return 0f;
            // return IoU value, intersection area divided by union area
            return (float)inter / union;
        }
    }
}
