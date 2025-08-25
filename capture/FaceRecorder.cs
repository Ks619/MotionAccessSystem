using OpenCvSharp;

namespace MotionAccessSystem.Capture
{
    /// <summary>
    /// Records clips only while faces exist in the CURRENT frame.
    /// - FPS with real-time pacing (prevents fast-forward playback).
    /// - Continues recording for 'holdSeconds' after last face disappears.
    /// - Raises OnClipClosed with metadata (start/end: UTC + Israel local, duration, frames, fps, path).
    /// - Timestamps (file names & logs) are generated in the configured time zone (Israel by default).
    /// </summary>
    public sealed class FaceRecorder : IDisposable
    {
        public sealed record ClipLog(
            DateTime StartUtc,
            DateTime EndUtc,
            DateTime StartLocal,   // Israel local time
            DateTime EndLocal,     // Israel local time
            double DurationSec,
            int Frames,
            double Fps,
            string FilePath,
            string TimeZoneId
        );

        public Action<ClipLog>? OnClipClosed { get; set; }

        private readonly string _outputDir;
        private readonly int _width;
        private readonly int _height;
        private readonly int _fourcc;
        private readonly TimeSpan _hold;

        // pacing
        private readonly double _targetFps;
        private readonly TimeSpan _frameInterval;
        private DateTime _nextWriteDueUtc = DateTime.MinValue;

        // writer state
        private VideoWriter? _writer;
        private bool _isRecording = false;
        private DateTime _lastFaceSeenUtc = DateTime.MinValue;
        private string _currentFilePath = string.Empty;

        // current clip stats
        private DateTime _currentStartUtc = DateTime.MinValue;
        private int _writtenFrames = 0;

        // pre-roll
        private readonly bool _usePreRoll;
        private readonly Queue<Mat> _preRoll;
        private readonly int _preRollCapacity;

        // expose recording status for UI
        public bool IsRecording => _isRecording;
        public double TargetFps => _targetFps;

        // time zone handling (Israel by default)
        private readonly TimeZoneInfo _tz;
        private readonly string _tzId;

        public FaceRecorder(
            string outputDir,
            double targetFps,
            int width,
            int height,
            string fourcc = "XVID",
            double holdSeconds = 0.0,
            bool usePreRoll = false,
            double preRollSeconds = 0.0,
            string? timeZoneId = null)
        {
            _outputDir = outputDir;
            Directory.CreateDirectory(_outputDir);

            _targetFps = ClampFps(targetFps);
            _frameInterval = TimeSpan.FromSeconds(1.0 / _targetFps);

            _width = width;
            _height = height;
            _fourcc = FourCC.FromString(fourcc);
            _hold = TimeSpan.FromSeconds(holdSeconds);

            _usePreRoll = usePreRoll;
            _preRollCapacity = Math.Max(0, (int)Math.Round(_targetFps * preRollSeconds));
            _preRoll = new Queue<Mat>(_preRollCapacity);

            _tz = ResolveTimeZone(timeZoneId);
            _tzId = _tz.Id;
        }

        public void Update(Mat frame, bool faceDetected)
        {
            var nowUtc = DateTime.UtcNow;

            if (_usePreRoll) EnqueuePreRoll(frame);

            if (faceDetected)
            {
                _lastFaceSeenUtc = nowUtc;
                if (!_isRecording)
                    StartRecording(nowUtc);
            }

            if (_isRecording)
            {
                if (_nextWriteDueUtc == DateTime.MinValue)
                    _nextWriteDueUtc = nowUtc;

                while (nowUtc >= _nextWriteDueUtc)
                {
                    _writer!.Write(frame);
                    _writtenFrames++;
                    _nextWriteDueUtc += _frameInterval;
                }

                if (!faceDetected && (nowUtc - _lastFaceSeenUtc) > _hold)
                    StopRecording();
            }
        }

        private void StartRecording(DateTime nowUtc)
        {
            // timestamp for file name in Israel local time
            var nowLocal = TimeZoneInfo.ConvertTimeFromUtc(nowUtc, _tz);
            string ts = nowLocal.ToString("yyyyMMdd_HHmmss");

            _currentFilePath = Path.Combine(_outputDir, $"face_{ts}.avi"); 

            _writer = new VideoWriter(
                _currentFilePath,
                _fourcc,
                _targetFps,
                new Size(_width, _height)
            );

            if (!_writer.IsOpened())
            {
                Console.WriteLine($"[WARN] Failed to start recording '{_currentFilePath}'");
                _writer?.Dispose();
                _writer = null;
                return;
            }

            _isRecording = true;
            _currentStartUtc = nowUtc;
            _writtenFrames = 0;
            _nextWriteDueUtc = nowUtc;

            Console.WriteLine($"[REC] Start → {_currentFilePath} (fps={_targetFps:F1}, tz={_tzId})");

            if (_usePreRoll && _preRoll.Count > 0)
            {
                foreach (var m in _preRoll)
                {
                    _writer!.Write(m);
                    _writtenFrames++;
                }
            }
        }

        private void StopRecording()
        {
            if (!_isRecording)
                return;

            _isRecording = false;

            try
            {
                _writer?.Release();
            }
            catch
            { /* ignore */ }
            _writer?.Dispose();
            _writer = null;

            _nextWriteDueUtc = DateTime.MinValue;

            var endUtc = DateTime.UtcNow;
            var duration = (endUtc - _currentStartUtc).TotalSeconds;

            var startLocal = TimeZoneInfo.ConvertTimeFromUtc(_currentStartUtc, _tz);
            var endLocal   = TimeZoneInfo.ConvertTimeFromUtc(endUtc, _tz);

            Console.WriteLine($"[REC] Stop  → {_currentFilePath} (frames={_writtenFrames}, dur={duration:F2}s)");

            try
            {
                OnClipClosed?.Invoke(new ClipLog(
                    StartUtc: _currentStartUtc,
                    EndUtc: endUtc,
                    StartLocal: startLocal,
                    EndLocal: endLocal,
                    DurationSec: duration,
                    Frames: _writtenFrames,
                    Fps: _targetFps,
                    FilePath: _currentFilePath,
                    TimeZoneId: _tzId
                ));
            }
            catch { /* don't fail app due to logging */ }

            _currentFilePath = string.Empty;
            _currentStartUtc = DateTime.MinValue;
            _writtenFrames = 0;

            ClearPreRoll();
        }

        private void EnqueuePreRoll(Mat frame)
        {
            if (_preRollCapacity <= 0)
                return;
                
            _preRoll.Enqueue(frame.Clone());
            while (_preRoll.Count > _preRollCapacity)
                _preRoll.Dequeue().Dispose();
        }

        private void ClearPreRoll()
        {
            while (_preRoll.Count > 0)
                _preRoll.Dequeue().Dispose();
        }

        private static double ClampFps(double fps)
        {
            if (double.IsFinite(fps) && fps >= 5.0 && fps <= 60.0)
                return fps;
            return 20.0; // default target fps
        }

        private static TimeZoneInfo ResolveTimeZone(string? preferredId)
        {
            // 1) explicit id if given
            if (!string.IsNullOrWhiteSpace(preferredId))
            {
                try
                {
                    return TimeZoneInfo.FindSystemTimeZoneById(preferredId);
                }
                catch { }
            }
            // 2) Windows id
            try
            {
                return TimeZoneInfo.FindSystemTimeZoneById("Israel Standard Time");
            }
            catch { }
            // 3) IANA id
            try
            {
                return TimeZoneInfo.FindSystemTimeZoneById("Asia/Jerusalem");
            }
            catch { }
            // 4) fallback
            return TimeZoneInfo.Local;
        }

        public void Dispose()
        {
            StopRecording();
            _writer?.Dispose();
            ClearPreRoll();
        }
    }
}
