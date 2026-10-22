import csv
import tempfile
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Literal, Optional, Dict, Type
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from . import interfaces
from .processing import FFMPEGSegmentProcessor

logger = logging.getLogger(__name__)

@dataclass
class Record:
    video_id: int
    video_url: str
    captions_url: str
    speaker_size: str

@dataclass
class PipelineConfig:
    max_workers: int = 4
    output_dir: str = "output"
    gop_value: int = 2
    b_frames: int = 0 

@dataclass
class FailedStatus:
    failed_segments: List[Dict[str,str]]
    failed_records: List[Dict[str,str]]

class SegmentationPipeline:
    def __init__(self, 
                    downloader: interfaces.Downloader,
                    captions_processor: interfaces.CaptionsProcessor,
                    segment_processor: Type[FFMPEGSegmentProcessor],
                    segment_writer: interfaces.SegmentWriter,
                    config: Optional[PipelineConfig] = None
                ):
        self.downloader = downloader
        self.captions_processor = captions_processor
        self.segment_processor = segment_processor
        self.segment_writer = segment_writer
        self.config = config or PipelineConfig()
    
    def _resolve_file_suffix(self, speaker_side: str) -> Literal["R", "L", "M"]:
        match speaker_side:
            case "left":
                return "L"
            case "right":
                return "R"
            case _:
                return "M"
    
    def process_single_record(self, record: Record, download_dir: Path) -> List[Dict[str, str]]:
        video_path = self.downloader.to_file(record.video_url, download_dir)
        captions_path = self.downloader.to_file(record.captions_url, download_dir)
        
        segments = self.captions_processor.get_segments(captions_path)
        if len(segments) == 0:
            raise Exception("no segments in vtt file")

        failed_segments = []
        with self.segment_processor(video_path, self.config.gop_value, self.config.b_frames) as byte_getter:
            for i, segment_timestamp in enumerate(segments):
                try:
                    video_bytes = byte_getter.get_bytes(segment_timestamp.start, segment_timestamp.duration)
                   
                    file_suffix = self._resolve_file_suffix(record.speaker_size)
                    segment_name = f'{record.video_id:03d}_{i:03d}_{file_suffix}'
                    self.segment_writer.write(segment_name, video_bytes, segment_timestamp.caption)
                except Exception as e:
                    failed_segments.append(
                        {
                            "video_id": str(record.video_id),
                            "video_url": record.video_url,
                            "captions_url": record.captions_url,
                            "start": segment_timestamp.start,
                            "duration": segment_timestamp.duration,
                            "caption": segment_timestamp.caption,
                            "error_msg": str(e)
                        }
                    )

        video_path.unlink(missing_ok=True)
        captions_path.unlink(missing_ok=True)
        
        return failed_segments
    
    def _write_csv(self, output_file: Path, rows: List[Dict], fieldnames: List[str]):
        with open(output_file.absolute(), 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    def get_records(self, csv_path: Path) -> List[Record]:
        df = pd.read_csv(csv_path.absolute())

        required_columns = ['video_id', 'video_url', 'subtitles_url', 'speaker_side']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise Exception(f"required columns not present: {missing_columns}")
        
        records = [
            Record(row['video_id'], row['video_url'], row['subtitles_url'], row['speaker_side'])
            for _, row in df.iterrows()
        ]
        
        return records

    
    def process_records(self, records: List[Record], download_dir: Path) -> FailedStatus:
        failed_records = []
        failed_segments = []
        success_count = 0
        pbar = tqdm(total=len(records), desc="Processing records", unit="record")
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_record = {
                executor.submit(self.process_single_record, record, Path(download_dir)): record
                for record in records
            }
                
            for future in as_completed(future_to_record):
                record = future_to_record[future]
                try:
                    process_stats = future.result()
                    if len(process_stats) > 0:
                        failed_segments.extend(process_stats)
                    success_count += 1
                    logger.debug(f"record {record.video_id} successfully processed")
                except Exception as e:
                    logger.exception(f"Unexpected error processing record: %s", record.video_id)
                    failed_records.append({
                        'video_id': record.video_id,
                        'video_url': record.video_url,
                        'captions_url': record.captions_url,
                        'error_msg': str(e)
                    })
                finally:
                    pbar.update(1)
                    pbar.set_postfix({"success": success_count, "failed records": len(failed_records), "failed segments": len(failed_segments)})
        pbar.close()
        return FailedStatus(failed_segments, failed_records)

    def handle_failed_status(self, status: FailedStatus):
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        failed_records_fieldnames = ['video_id', 'video_url', 'captions_url', 'error_msg']
        self._write_csv(Path(self.config.output_dir) / "failed_records.csv", status.failed_records, failed_records_fieldnames)
        
        failed_segments_fieldnames = ["video_id", "video_url", "captions_url", "start", "duration", "caption", "error_msg"]
        self._write_csv(Path(self.config.output_dir) / "failed_segments.csv", status.failed_segments, failed_segments_fieldnames)
    
    def __call__(self, csv_path: Path):
        logger.info("Processing start")
        records = self.get_records(csv_path)
        logger.info(f"Got {len(records)} records")
        
        with tempfile.TemporaryDirectory() as temp_dir:            
           failed_status = self.process_records(records, Path(temp_dir))
        try:
            self.handle_failed_status(failed_status)
        except Exception:
            logger.exception("handle failed status error")
        
        logger.info(f"Processing complete. Failed records: {len(failed_status.failed_records)}, Failed segments: {len(failed_status.failed_segments)}")
