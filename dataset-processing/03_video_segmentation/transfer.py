import time
import requests
from threading import Lock

from pathlib import Path
from webdataset import ShardWriter  # type: ignore
import huggingface_hub as hf

from .interfaces import Downloader, SegmentWriter

class FileDownloader(Downloader):
    def __init__(self, timeout: int = 20, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
    
    def to_file(self, url: str, dest_dir: Path) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        safe_name = url.split('/')[-1].split('?')[0] or "download"
        dest = dest_dir / safe_name
        base = dest.stem
        suffix = dest.suffix or ""
        
        counter = 1
        while dest.exists():
            dest = dest_dir / f"{base}_{counter}{suffix}"
            counter += 1
        
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(url, timeout=self.timeout, stream=True)
                resp.raise_for_status()
                
                with open(dest, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                return dest
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise Exception(str(e))
                time.sleep(2 ** attempt)
        
        raise Exception("Failed to download file")

class HuggingfaceException(Exception):
    """Error occurred in huggingface"""
    pass

class HuggingFacePoster:
    def __init__(self, repo_id: str):
        self.api = hf.HfApi()
        self.repo_id = repo_id

    def post(self, file_path: str) -> bool:
        path = Path(file_path)
        self.api.upload_file(
            path_or_fileobj=path.absolute(),
            path_in_repo=path.name,
            repo_id=self.repo_id,
            repo_type="dataset",
        )
        
        return True

class WebDatasetSegmentWriter(SegmentWriter):
    def __init__(self, writer: ShardWriter):
        self.writer = writer
        self._lock = Lock()
    
    def write(self, segment_name: str, video_bytes: bytes, caption: str):        
        with self._lock:
            self.writer.write({
                "__key__": segment_name,
                "mp4": video_bytes,
                "txt": caption.encode('utf-8')
            })
        
if __name__ == "__main__":
    HF_REPO = 'croco-corp/pjm-segments'
    URL = "https://www.korpuspjm.uw.edu.pl/assets/movies/K64BF01-11_583.mp4"
    downloader = FileDownloader()
    
    hf_poster = HuggingFacePoster(HF_REPO)

    import tempfile    
    shards_dir = tempfile.TemporaryDirectory()
    downloads_dir = tempfile.TemporaryDirectory()
    
    shard_writer = ShardWriter(f"{shards_dir.name}/test-%06d.tar", maxcount=1, post=hf_poster.post)
    segment_writer = WebDatasetSegmentWriter(shard_writer)
    
    file_path = downloader.to_file(URL, Path(downloads_dir.name))
    with open(file_path.absolute(), "rb") as f:
        bytes = f.read()
    segment_writer.write("test", bytes, "test caption")
    
    shard_writer.close()
    shards_dir.cleanup()
    downloads_dir.cleanup()