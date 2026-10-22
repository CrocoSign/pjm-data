import subprocess
from typing import List
from pathlib import Path
from tempfile import NamedTemporaryFile

import webvtt

from .interfaces import ByteGetter, CaptionsProcessor, SegmentTimestamp

def timestamp_to_milliseconds(caption_timestamp: webvtt.models.Timestamp) -> int:
    h,m,s,ms = caption_timestamp.to_tuple()
    
    total_ms = ((h * 60 + m) * 60 + s) * 1000 + ms

    return total_ms

def milliseconds_to_timestamp(total_milliseconds: int) -> webvtt.models.Timestamp:
    hours = total_milliseconds // 3_600_000
    rem = total_milliseconds % 3_600_000
    minutes = rem // 60_000
    rem = rem % 60_000
    seconds = rem // 1000
    milliseconds = rem % 1000
    
    return webvtt.models.Timestamp(hours, minutes, seconds, milliseconds)

class VTTCaptionsProcessor(CaptionsProcessor):
    def __init__(self, offset_milliseconds: int):
        self.offset_milliseconds = offset_milliseconds
    
    def get_segments(self, vtt_path: Path) -> List[SegmentTimestamp]:
        vtt_file = webvtt.read(str(vtt_path.absolute()))

        timestamps = []
        for caption in vtt_file.captions:
            start_time_ms = timestamp_to_milliseconds(caption.start_time)
            start_time_ms = start_time_ms - self.offset_milliseconds
            if start_time_ms < 0:
                start_time_ms = 0
                
            end_time_ms = timestamp_to_milliseconds(caption.end_time)
            
            duration_ms = end_time_ms - start_time_ms
            duration = milliseconds_to_timestamp(duration_ms)
            
            timestamps.append(
                SegmentTimestamp(
                    start=str(caption.start_time),
                    duration=str(duration),
                    caption=caption.text
                )
            )
            
        return timestamps
    

class FFMPEGByteGetter(ByteGetter):
    def __init__(self, video_path: Path):
        self.video_path = video_path
    
    def get_bytes(self, start: str, duration: str) -> bytes:
        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "warning",
            "-ss", start,
            "-i", str(self.video_path.absolute()),
            "-t", duration,
            "-c", "copy",
            "-avoid_negative_ts", "1",
            "-f", "mpegts",
            "-"
        ]
        
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
        
        return result.stdout

class FFMPEGSegmentProcessor:
    def __init__(self, video_path: Path, gop_value: int, b_frames: int):
        self.video_path = video_path
        self.gop_value = gop_value
        self.b_frames = b_frames
    
    def __enter__(self):
        self.tmp_file = NamedTemporaryFile(suffix='.mp4')
        file_path = Path(self.tmp_file.name)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(self.video_path.absolute()),
            "-loglevel", "warning",
            "-c:v", "h264_nvenc",
            "-rc:v", "vbr",
            "-cq:v", "21",
            "-g", f"{self.gop_value}", 
            "-keyint_min", f"{self.gop_value}", "-sc_threshold", "0",
            "-bf", f"{self.b_frames}",
            "-an",
            file_path
        ]
        
        subprocess.run(cmd, check=True)
        
        return FFMPEGByteGetter(file_path)
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tmp_file.close()

if __name__ == "__main__":
    VIDEO_URL = "https://www.korpuspjm.uw.edu.pl/assets/movies/K27AF01-15_263.mp4"
    OUTPUT_PATH = str(Path("./output.mp4").absolute())
    GOP_VALUE = 5
    B_FRAMES = 0
    video_file = NamedTemporaryFile()
    video_path = Path(video_file.name)
    
    subprocess.run([
        "wget", VIDEO_URL,
        "-O", str(video_path.absolute())
    ])
    
    with FFMPEGSegmentProcessor(video_path, GOP_VALUE, B_FRAMES) as byte_getter:
        print(byte_getter.video_path)
        b = byte_getter.get_bytes("00:00:00.000", "00:00:05.000")
        print(b)
        with open(OUTPUT_PATH, "wb") as f:
            f.write(b)
    
    