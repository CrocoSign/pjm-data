from typing import List, AnyStr
from pathlib import Path
from dataclasses import dataclass

class Downloader:
    def to_file(self, url: str, output_path: Path) -> Path:
        raise NotImplementedError

@dataclass
class SegmentTimestamp:
    start: str
    duration: str
    caption: str

class CaptionsProcessor:
    def get_segments(self, vtt_path: Path) -> List[SegmentTimestamp]:
        raise NotImplementedError
    
class ByteGetter:
    def get_bytes(self, start: str, duration: str) -> AnyStr:
        raise NotImplementedError
  
class SegmentWriter:
    def write(self, segment_name: str, video_bytes: bytes, caption: str):
        raise NotImplementedError
