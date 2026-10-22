"""
Sign Language Dialog Video Processing Pipeline

A robust pipeline for processing sign language dialog videos:
- Downloads videos and VTT files from URLs
- Splits videos into left/right speakers
- Processes VTT files with speaker alignment
- Manages metadata for individual speakers
- Uploads processed data to Hugging Face datasets
- Provides error handling and logging
"""

from pathlib import Path
import subprocess
import tempfile
import logging
import csv
import pandas as pd
import requests
import webvtt
import re
import shutil
import time
from typing import Optional, Tuple, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import huggingface_hub as hf
from dataclasses import dataclass


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration for the video processing pipeline"""
    max_folder_size_gb: float = 2.0
    download_timeout: int = 300  # 5 minutes
    ffmpeg_timeout: int = 600   # 10 minutes
    max_workers: int = 4
    temp_dir: Optional[str] = None
    hf_repo_id: str = 'croco-corp/pjm-data'
    ignore_patterns: List[str] = None
    
    def __post_init__(self):
        if self.ignore_patterns is None:
            self.ignore_patterns = ["*.log", "*.tmp", "__pycache__/"]


@dataclass
class VideoRecord:
    """Represents a video record from the CSV"""
    video_id: str
    video_url: str
    subtitles_url: str
    video_type: str
    age: str
    sex: str
    task_id: str


@dataclass
class ProcessedRecord:
    """Represents a processed video record with speaker information"""
    video_id: str
    speaker_side: str  # 'left' or 'right'
    video_path: str
    subtitles_path: str
    video_type: str
    age: str
    sex: str
    task_id: str


class VideoProcessor:
    """Handles video splitting using FFmpeg"""
    
    @staticmethod
    def split_video(video_path: Path, destination_path: Path, timeout: Optional[int] = None) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Split video into left and right halves
        
        Args:
            video_path: Path to input video
            destination_path: Directory to save split videos
            timeout: FFmpeg timeout in seconds
            
        Returns:
            Tuple of (left_video_path, right_video_path) or (None, None) on error
        """
        try:
            root_name = video_path.stem
            destination_path.mkdir(parents=True, exist_ok=True)
            
            left_file = destination_path / f'{root_name}_left.mp4'
            right_file = destination_path / f'{root_name}_right.mp4'
            
            cmd = [
                "ffmpeg", "-y",  # -y to overwrite output files
                "-i", str(video_path.absolute()),
                "-filter_complex",
                "[0:v]split=2[vL][vR];"
                "[vL]crop=iw/2:ih:0:0[left];"
                "[vR]crop=iw/2:ih:iw/2:0[right]",
                "-map", "[left]", "-c:v", "h264_nvenc", "-rc:v", "vbr", "-cq:v", "23", "-an",
                str(left_file),
                "-map", "[right]", "-c:v", "h264_nvenc", "-rc:v", "vbr", "-cq:v", "23", "-an",
                str(right_file)
            ]
            
            logger.info(f"Splitting video: {video_path.name}")
            result = subprocess.run(cmd, check=True, timeout=timeout, capture_output=True, text=True)
            
            if left_file.exists() and right_file.exists():
                logger.info(f"Successfully split video: {video_path.name}")
                return left_file, right_file
            else:
                logger.error(f"Split video files not created for: {video_path.name}")
                return None, None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg error splitting {video_path.name}: {e.stderr}")
            return None, None
        except subprocess.TimeoutExpired:
            logger.error(f"FFmpeg timeout splitting {video_path.name}")
            return None, None
        except Exception as e:
            logger.error(f"Unexpected error splitting {video_path.name}: {str(e)}")
            return None, None


class VTTProcessor:
    """Handles VTT subtitle file processing"""
    
    ALIGN_PATTERN = re.compile(r'(?i)\balign\s*:\s*([^\s]+)')
    
    @staticmethod
    def split_vtt(vtt_path: Path, destination_path: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Split VTT file based on align metadata
        
        Args:
            vtt_path: Path to input VTT file
            destination_path: Directory to save split VTT files
            
        Returns:
            Tuple of (left_vtt_path, right_vtt_path) or (None, None) on error
        """
        try:
            root_name = vtt_path.stem
            destination_path.mkdir(parents=True, exist_ok=True)
            
            left_file = destination_path / f'{root_name}_left.vtt'
            right_file = destination_path / f'{root_name}_right.vtt'
            
            # Read and parse VTT content
            text = vtt_path.read_text(encoding="utf-8")
            aligns = VTTProcessor.ALIGN_PATTERN.findall(text)
            
            try:
                input_vtt = webvtt.from_string(text)
            except Exception as e:
                logger.error(f"Error parsing VTT file {vtt_path.name}: {str(e)}")
                return None, None
            
            left_vtt = webvtt.WebVTT()
            right_vtt = webvtt.WebVTT()
            
            # Check if we have alignment data for all captions
            if len(input_vtt.captions) != len(aligns):
                logger.warning(f"Mismatch between captions ({len(input_vtt.captions)}) and align data ({len(aligns)}) in {vtt_path.name}")
                return None, None
            
            # Split captions based on alignment
            for i, caption in enumerate(input_vtt.captions):
                try:
                    align = aligns[i].lower()
                    if align == 'left':
                        left_vtt.captions.append(caption)
                    elif align == 'right':
                        right_vtt.captions.append(caption)
                    else:
                        logger.warning(f"Unknown align value '{align}' in {vtt_path.name}")
                except IndexError:
                    logger.error(f"Index error processing caption {i} in {vtt_path.name}")
                    continue
            
            # Save split VTT files
            left_vtt.save(str(left_file))
            right_vtt.save(str(right_file))
            
            logger.info(f"Successfully split VTT: {vtt_path.name} (left: {len(left_vtt.captions)}, right: {len(right_vtt.captions)})")
            return left_file, right_file
            
        except Exception as e:
            logger.error(f"Error splitting VTT {vtt_path.name}: {str(e)}")
            return None, None


class MetadataProcessor:
    """Handles metadata processing and splitting"""
    
    @staticmethod
    def split_metadata(age: str, sex: str) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        """
        Split age and sex metadata for left and right speakers
        
        Args:
            age: Age string like "32-45, 32-45" or "60+, 60+"
            sex: Sex string like "MM" or "FF"
            
        Returns:
            Tuple of ((left_age, left_sex), (right_age, right_sex))
        """
        # Split age
        age_parts = [part.strip() for part in age.split(',')]
        if len(age_parts) >= 2:
            left_age, right_age = age_parts[0], age_parts[1]
        else:
            left_age = right_age = age_parts[0] if age_parts else age
        
        # Split sex
        if len(sex) >= 2:
            left_sex, right_sex = sex[0], sex[1]
        else:
            left_sex = right_sex = sex
        
        return (left_age, left_sex), (right_age, right_sex)


class FileDownloader:
    """Handles file downloading with retry logic"""
    
    @staticmethod
    def download_to_file(url: str, dest_dir: Path, timeout: int = 20, max_retries: int = 3) -> Optional[Path]:
        """
        Download file from URL with retry logic
        
        Args:
            url: URL to download from
            dest_dir: Destination directory
            timeout: Download timeout in seconds
            max_retries: Maximum number of retry attempts
            
        Returns:
            Path to downloaded file or None on failure
        """
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename
        safe_name = url.split('/')[-1].split('?')[0] or "download"
        dest = dest_dir / safe_name
        base = dest.stem
        suffix = dest.suffix or ""
        
        # Handle filename conflicts
        counter = 1
        while dest.exists():
            dest = dest_dir / f"{base}_{counter}{suffix}"
            counter += 1
        
        # Download with retries
        for attempt in range(max_retries):
            try:
                logger.debug(f"Downloading {url} (attempt {attempt + 1}/{max_retries})")
                resp = requests.get(url, timeout=timeout, stream=True)
                resp.raise_for_status()
                
                with open(dest, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                logger.info(f"Successfully downloaded: {dest.name}")
                return dest
                
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to download after {max_retries} attempts: {url}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None


class HuggingFaceUploader:
    """Handles uploading to Hugging Face repository"""
    
    def __init__(self, repo_id: str):
        self.api = hf.HfApi()
        self.repo_id = repo_id
    
    def upload_file(self, file_path: Path, repo_path: Path) -> bool:
        try:
            logger.info(f'Uploading file {file_path} to {self.repo_id}')
            self.api.upload_file(
                path_or_fileobj=str(file_path.absolute()),
                path_in_repo=str(repo_path),
                repo_id=self.repo_id,
                repo_type="dataset",
            )

            return True
        except Exception as e:
            logger.error(f"Failed to upload file {file_path}: {str(e)}")
            
            return False

    def upload_folder(self, folder_path: Path, ignore_patterns: List[str]) -> bool:
        """
        Upload folder to Hugging Face repository
        
        Args:
            folder_path: Path to folder to upload
            ignore_patterns: Patterns to ignore during upload
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Uploading folder {folder_path} to {self.repo_id}")
            self.api.upload_large_folder(
                repo_id=self.repo_id,
                folder_path=str(folder_path.absolute()),
                repo_type='dataset',
                ignore_patterns=ignore_patterns
            )
            logger.info(f"Successfully uploaded folder: {folder_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload folder {folder_path}: {str(e)}")
            return False


class FolderSizeMonitor:
    """Monitors folder size and triggers uploads when threshold is reached"""
    
    @staticmethod
    def get_folder_size_gb(folder_path: Path) -> float:
        """Get folder size in GB"""
        total_size = 0
        for file_path in folder_path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 ** 3)  # Convert to GB
    
    @staticmethod
    def should_upload(folder_path: Path, max_size_gb: float) -> bool:
        """Check if folder should be uploaded based on size"""
        return FolderSizeMonitor.get_folder_size_gb(folder_path) >= max_size_gb


class VideoProcessingPipeline:
    """Main pipeline for processing sign language dialog videos"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.video_processor = VideoProcessor()
        self.vtt_processor = VTTProcessor()
        self.metadata_processor = MetadataProcessor()
        self.downloader = FileDownloader()
        self.uploader = HuggingFaceUploader(config.hf_repo_id)
        self.folder_monitor = FolderSizeMonitor()
        
        # Error tracking
        self.failed_records: List[Dict] = []
        self.processed_count = 0
        self.success_count = 0
    
    def init_paths(self, temp_dir: tempfile.TemporaryDirectory) -> Tuple[Path, Path]:
        """Initialize download and send paths"""
        download_path = Path(temp_dir.name) / 'download'
        send_path = Path(temp_dir.name) / 'send'
        
        download_path.mkdir(exist_ok=True, parents=True)
        (send_path / 'videos').mkdir(exist_ok=True, parents=True)
        (send_path / 'translations').mkdir(exist_ok=True, parents=True)
        
        return download_path, send_path
    
    def process_single_record(self, record: VideoRecord, download_path: Path, send_path: Path) -> Tuple[bool, List[ProcessedRecord]]:
        """
        Process a single video record
        
        Returns:
            Tuple of (success, list_of_processed_records)
        """
        try:
            logger.info(f"Processing video ID: {record.video_id}")
            
            # Download video and subtitles
            video_file = self.downloader.download_to_file(
                record.video_url, 
                download_path, 
                timeout=self.config.download_timeout
            )
            
            if not video_file:
                raise Exception("Failed to download video")
            
            vtt_file = self.downloader.download_to_file(
                record.subtitles_url, 
                download_path, 
                timeout=self.config.download_timeout
            )
            
            if not vtt_file:
                raise Exception("Failed to download VTT file")
            
            # Split video
            left_video, right_video = self.video_processor.split_video(
                video_file, 
                send_path / 'videos',
                timeout=self.config.ffmpeg_timeout
            )
            
            if not left_video or not right_video:
                raise Exception("Failed to split video")
            
            # Split VTT
            left_vtt, right_vtt = self.vtt_processor.split_vtt(
                vtt_file,
                send_path / 'translations'
            )
            
            if not left_vtt or not right_vtt:
                raise Exception("Failed to split VTT file")
            
            # Process metadata
            (left_age, left_sex), (right_age, right_sex) = self.metadata_processor.split_metadata(
                record.age, record.sex
            )
            
            # Create processed records
            processed_records = [
                ProcessedRecord(
                    video_id=record.video_id,
                    speaker_side="left",
                    video_path=str(left_video.relative_to(send_path)),
                    subtitles_path=str(left_vtt.relative_to(send_path)),
                    video_type=record.video_type,
                    age=left_age,
                    sex=left_sex,
                    task_id=record.task_id,
                ),
                ProcessedRecord(
                    video_id=record.video_id,
                    speaker_side="right",
                    video_path=str(right_video.relative_to(send_path)),
                    subtitles_path=str(right_vtt.relative_to(send_path)),
                    video_type=record.video_type,
                    age=right_age,
                    sex=right_sex,
                    task_id=record.task_id,
                )
            ]
            
            # Clean up downloaded files
            video_file.unlink(missing_ok=True)
            vtt_file.unlink(missing_ok=True)
            
            logger.info(f"Successfully processed video ID: {record.video_id}")
            return True, processed_records
            
        except Exception as e:
            logger.error(f"Failed to process video ID {record.video_id}: {str(e)}")
            self.failed_records.append({
                'video_id': record.video_id,
                'error': str(e),
                'video_url': record.video_url,
                'subtitles_url': record.subtitles_url
            })
            return False, []
    
    def save_processed_fields(self, processed_records: List[ProcessedRecord], output_path: Path):
        """Save processed records to CSV"""
        file_exists = output_path.exists()
        
        with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'video_id', 'speaker_side', 'video_path', 'subtitles_path',
                'video_type', 'age', 'sex', 'task_id'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            for record in processed_records:
                writer.writerow({
                    'video_id': record.video_id,
                    'speaker_side': record.speaker_side,
                    'video_path': record.video_path,
                    'subtitles_path': record.subtitles_path,
                    'video_type': record.video_type,
                    'age': record.age,
                    'sex': record.sex,
                    'task_id': record.task_id
                })

    def save_error_log(self, output_path: Path):
        """Save error log to CSV"""
        if not self.failed_records:
            return
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['video_id', 'error', 'video_url', 'subtitles_url']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.failed_records)
    
    def process_csv(self, csv_path: Path, output_dir: Path):
        """
        Process CSV file containing video records
        
        Args:
            csv_path: Path to input CSV file
            output_dir: Directory to save outputs
        """
        logger.info(f"Starting pipeline with CSV: {csv_path}")
        
        # Read CSV
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Failed to read CSV file: {str(e)}")
            return
        
        # Validate required columns
        required_columns = ['video_id', 'video_url', 'subtitles_url', 'video_type', 'age', 'sex', 'task_id']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return
        
        # Convert to VideoRecord objects
        records = []
        for _, row in df.iterrows():
            records.append(VideoRecord(
                video_id=str(row['video_id']),
                video_url=str(row['video_url']),
                subtitles_url=str(row['subtitles_url']),
                video_type=str(row['video_type']),
                age=str(row['age']),
                sex=str(row['sex']),
                task_id=str(row['task_id'])
            ))
        
        logger.info(f"Found {len(records)} records to process")
        
        # Process records
        all_processed_records = []
        batch_number = 1

        output_dir.mkdir(parents=True, exist_ok=True)
        output_csv_path = output_dir / 'output.csv'
        
        with tempfile.TemporaryDirectory(dir=self.config.temp_dir) as temp_dir:
            download_path, send_path = self.init_paths(type('TempDir', (), {'name': temp_dir})())
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                future_to_record = {
                    executor.submit(self.process_single_record, record, download_path, send_path): record
                    for record in records
                }
                
                for future in as_completed(future_to_record):
                    record = future_to_record[future]
                    self.processed_count += 1
                    
                    try:
                        success, processed_records = future.result()
                        if success:
                            self.success_count += 1
                            all_processed_records.extend(processed_records)
                            
                            # Check if we should upload
                            if self.folder_monitor.should_upload(send_path, self.config.max_folder_size_gb):
                                logger.info(f"Folder size limit reached, uploading batch {batch_number}")
                                
                                # Save current batch CSV
                                self.save_processed_fields(all_processed_records, output_csv_path)
                                
                                # Upload to Hugging Face
                                if self.uploader.upload_folder(send_path, self.config.ignore_patterns):
                                    logger.info(f"Successfully uploaded batch {batch_number}")
                                    self.uploader.upload_file(output_csv_path, Path('data.csv'))
                                    
                                    # Clean local files
                                    shutil.rmtree(send_path / 'videos', ignore_errors=True)
                                    shutil.rmtree(send_path / 'translations', ignore_errors=True)
                                    (send_path / 'videos').mkdir(exist_ok=True)
                                    (send_path / 'translations').mkdir(exist_ok=True)
                                    
                                    all_processed_records = []  # Reset for next batch
                                    batch_number += 1
                                else:
                                    logger.error(f"Failed to upload batch {batch_number}")
                        
                        # Log progress
                        if self.processed_count % 10 == 0:
                            logger.info(f"Progress: {self.processed_count}/{len(records)} processed, {self.success_count} successful")
                    
                    except Exception as e:
                        logger.error(f"Unexpected error processing record {record.video_id}: {str(e)}")
            
            # Upload remaining files
            if all_processed_records:
                logger.info(f"Uploading final batch {batch_number}")
                self.save_processed_fields(all_processed_records, output_csv_path)
                
                if self.uploader.upload_folder(send_path, self.config.ignore_patterns):
                    self.uploader.upload_file(output_csv_path, Path('data.csv'))
                else:
                    logger.error(f"Failed to upload final batch {batch_number}")
        
        # Save error log
        if self.failed_records:
            error_log_path = output_dir / 'failed_records.csv'
            self.save_error_log(error_log_path)
            logger.info(f"Saved error log to: {error_log_path}")
        
        # Final summary
        logger.info(f"Pipeline completed: {self.success_count}/{len(records)} successful, {len(self.failed_records)} failed")


def main():
    """Main function to run the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process sign language dialog videos')
    parser.add_argument('csv_path', help='Path to input CSV file')
    parser.add_argument('--output-dir', default='./output', help='Output directory')
    parser.add_argument('--max-size-gb', type=float, default=2.0, help='Maximum folder size in GB before upload')
    parser.add_argument('--max-workers', type=int, default=4, help='Maximum number of worker threads')
    parser.add_argument('--hf-repo-id', default='croco-corp/pjm-data', help='Hugging Face repository ID')
    parser.add_argument('--temp-dir', help='Temporary directory for processing')
    
    args = parser.parse_args()
    
    # Create configuration
    config = ProcessingConfig(
        max_folder_size_gb=args.max_size_gb,
        max_workers=args.max_workers,
        hf_repo_id=args.hf_repo_id,
        temp_dir=args.temp_dir
    )
    
    # Create and run pipeline
    pipeline = VideoProcessingPipeline(config)
    pipeline.process_csv(Path(args.csv_path), Path(args.output_dir))


if __name__ == "__main__":
    main()
