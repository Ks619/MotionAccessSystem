using System;
using System.Globalization;
using System.IO;
using System.Text.Json;
using MotionAccessSystem.Capture;

namespace MotionAccessSystem.Logging
{
    public sealed class EventLogger
    {
        private readonly string _csvPath;
        private readonly string _jsonlPath;

        public EventLogger(string outputDir, string csvFileName = "logs.csv", string jsonlFileName = "logs.jsonl")
        {
            Directory.CreateDirectory(outputDir);
            _csvPath = Path.Combine(outputDir, csvFileName);
            _jsonlPath = Path.Combine(outputDir, jsonlFileName);
            EnsureCsvHeader();
        }

        public void Log(FaceRecorder.ClipLog log)
        {
            try
            {
                AppendCsv(log);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[LOG][CSV] {ex.Message}");
            }
            try
            {
                AppendJsonl(log);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[LOG][JSONL] {ex.Message}");
            }
        }

        private void EnsureCsvHeader()
        {
            if (!File.Exists(_csvPath))
            {
                File.WriteAllText(_csvPath,
                    "start_local_iso,end_local_iso,start_utc_iso,end_utc_iso,duration_sec,frames,fps,file_path,timezone" +
                    Environment.NewLine);
            }
        }

        private void AppendCsv(FaceRecorder.ClipLog log)
        {
            var line = string.Join(",",
                Escape(log.StartLocal.ToString("o")),  // Israel local ISO 8601
                Escape(log.EndLocal.ToString("o")),
                Escape(log.StartUtc.ToString("o")),
                Escape(log.EndUtc.ToString("o")),
                Escape(log.DurationSec.ToString("0.###", CultureInfo.InvariantCulture)),
                Escape(log.Frames.ToString(CultureInfo.InvariantCulture)),
                Escape(log.Fps.ToString("0.###", CultureInfo.InvariantCulture)),
                Escape(log.FilePath),
                Escape(log.TimeZoneId)
            ) + Environment.NewLine;

            File.AppendAllText(_csvPath, line);
        }

        private void AppendJsonl(FaceRecorder.ClipLog log)
        {
            var json = JsonSerializer.Serialize(log, new JsonSerializerOptions { WriteIndented = false });
            File.AppendAllText(_jsonlPath, json + Environment.NewLine);
        }

        private static string Escape(string s)
        {
            if (s.Contains('"') || s.Contains(','))
            {
                s = s.Replace("\"", "\"\"");
                return $"\"{s}\"";
            }
            return s;
        }
    }
}
