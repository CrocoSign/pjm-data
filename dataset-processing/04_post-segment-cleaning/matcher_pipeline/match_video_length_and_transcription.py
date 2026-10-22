import subprocess
import webvtt
import os
import logging
from pathlib import Path
from typing import Union

class VideoTranscriptionMatcher:
    def __init__(self, video_dir: Union[str, Path], transcription_dir: Union[str, Path], tolerance: float = 0.5):
        if isinstance(transcription_dir, Path):
            transcription_dir = str(transcription_dir)
        if isinstance(video_dir, Path):
            video_dir = str(video_dir)
        self.video_dir = video_dir
        self.transcription_dir = transcription_dir
        self.tolerance = tolerance
        self.match_count = 0
        self.cut_count = 0
        self.error_count = 0
        self.total_files = 0

    def check_video_length(self, video_path: str) -> float:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            raise RuntimeError(f"ffprobe error: {result.stderr}")

        duration_str = result.stdout.strip()
        if not duration_str:
            raise ValueError(f"No duration found for: {video_path}")

        return float(duration_str)

    def check_transcription_length(self, transcription_path: str) -> float:
        if not os.path.exists(transcription_path):
            raise FileNotFoundError(f"Transcription not found: {transcription_path}")

        max_end_time = 0.0
        for caption in webvtt.read(transcription_path):
            if caption.end_in_seconds:
                max_end_time = max(max_end_time, caption.end_in_seconds)

        return max_end_time

    def find_last_matching_segment(
        self, transcription_path: str, max_time: float
    ) -> float:
        if not os.path.exists(transcription_path):
            raise FileNotFoundError(f"Transcription not found: {transcription_path}")

        last_valid_end = 0.0
        for caption in webvtt.read(transcription_path):
            if caption.end_in_seconds and caption.end_in_seconds <= max_time:
                last_valid_end = caption.end_in_seconds
            else:
                break

        return last_valid_end

    def cut_video(
        self, input_video: str, output_video: str, start_time: float, end_time: float
    ) -> None:
        duration = end_time - start_time
        result = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_video,
                "-ss",
                str(start_time),
                "-t",
                str(duration),
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-preset",
                "fast",
                output_video,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            logging.error(f"FFmpeg error: {result.stderr}")
            raise RuntimeError(f"Video cutting failed: {result.stderr}")

    def cut_transcription(
        self, input_vtt: str, output_vtt: str, start_time: float, end_time: float
    ) -> None:
        def seconds_to_timestamp(seconds: float) -> str:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

        vtt = webvtt.read(input_vtt)
        new_vtt = webvtt.WebVTT()

        for caption in vtt:
            if (
                caption.start_in_seconds >= start_time
                and caption.end_in_seconds <= end_time
            ):
                new_start = seconds_to_timestamp(caption.start_in_seconds - start_time)
                new_end = seconds_to_timestamp(caption.end_in_seconds - start_time)
                new_caption = webvtt.Caption(
                    start=new_start, end=new_end, text=caption.text
                )
                new_vtt.captions.append(new_caption)

        new_vtt.save(output_vtt)

    def process_file(self, video_file: str) -> None:
        transcription_file = os.path.splitext(video_file)[0] + ".vtt"
        video_path = os.path.join(self.video_dir, video_file)
        transcription_path = os.path.join(self.transcription_dir, transcription_file)

        if not os.path.exists(transcription_path):
            logging.warning(f" Transcription not found for {video_file}, skipping.")
            self.error_count += 1
            return

        try:
            video_length = self.check_video_length(video_path)
            transcription_length = self.check_transcription_length(transcription_path)

            last_segment_end = self.find_last_matching_segment(
                transcription_path, video_length + self.tolerance
            )

            if abs(video_length - last_segment_end) < self.tolerance:
                logging.info(f" Video {video_file} and transcription are synchronized.")
                logging.info(
                    f"  Video: {video_length:.2f}s | Last segment: {last_segment_end:.2f}s"
                )
                self.match_count += 1
                return

            logging.warning(
                f"Video {video_file} and transcription need synchronization."
            )
            logging.warning(
                f"  Video: {video_length:.2f}s | Transcription: {transcription_length:.2f}s"
            )

            cut_point = min(video_length, transcription_length)
            last_segment_end = self.find_last_matching_segment(
                transcription_path, cut_point
            )

            if last_segment_end == 0.0:
                logging.error(
                    f"  No valid segments found within {cut_point:.2f}s, skipping."
                )
                self.error_count += 1
                return

            logging.info(f"  Cutting both to {last_segment_end:.2f}s...")

            temp_video = video_path + ".temp.mp4"
            temp_vtt = transcription_path + ".temp.vtt"

            self.cut_video(video_path, temp_video, 0.0, last_segment_end)
            os.replace(temp_video, video_path)
            logging.info(
                f"  Video cut from {video_length:.2f}s to {last_segment_end:.2f}s"
            )

            self.cut_transcription(transcription_path, temp_vtt, 0.0, last_segment_end)
            os.replace(temp_vtt, transcription_path)
            logging.info(
                f"Transcription cut from {transcription_length:.2f}s to {last_segment_end:.2f}s"
            )

            self.cut_count += 1

        except Exception as e:
            logging.error(f"Error processing {video_file}: {e}")
            self.error_count += 1

    def process_all(self) -> None:
        for video_file in os.listdir(self.video_dir):
            self.process_file(video_file)
            self.total_files += 1

        self.print_summary()

    def print_summary(self) -> None:
        logging.info(
            f"\nSummary: {self.match_count}/{self.total_files} files matched, "
            f"{self.cut_count} pairs cut, {self.error_count} errors."
        )

def main():
    logging.getLogger()
    base_dir = Path(__file__).resolve().parent.parent.parent
    transcription_dir = base_dir / "data" / "vtts"
    video_dir = base_dir / "data" / "videos"
    transcription_dir = base_dir / "data" / "vtts"

    matcher = VideoTranscriptionMatcher(video_dir, transcription_dir, tolerance=1.0)
    matcher.process_all()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
