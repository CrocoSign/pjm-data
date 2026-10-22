from typing import Callable
from datasets import Dataset
from abc import abstractmethod, ABC
from dataclasses import dataclass
import tempfile
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from tqdm import tqdm
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class BoundingBoxInfo:
  xywh_cumulative: np.ndarray
  xyxy_cumulative: np.ndarray
  img_shape: tuple[int,int]

@dataclass
class Padding:
  top: int = 0
  bottom: int = 0
  left: int = 0
  right: int = 0

@dataclass
class CropParams:
  x_start: int
  y_start: int
  x_end: int
  y_end: int
  padding: Padding

@dataclass
class BoundingBoxSummary:
  x_middle: np.float32
  y_middle: np.float32
  box_width: np.int32
  box_height: np.int32
  video_height: np.int32
  video_width: np.int32

class BoundingBoxExtractor(ABC):
  """Bounding box extractor"""
  @abstractmethod
  def extract(self, video_path: Path) -> BoundingBoxInfo:
    pass

class Collector(ABC):
  """Crop params collector"""
  @abstractmethod
  def collect(self, crop_params: CropParams, video_id: str):
    pass

class ExtractionPipeline:
  def __init__(
      self,
      bbox_extractor: BoundingBoxExtractor,
      bbox_summariser_fn: Callable[[BoundingBoxInfo],BoundingBoxSummary],
      bbox_summary_to_crop_params_fn: Callable[[BoundingBoxSummary], CropParams],
      collector: Collector,
      max_workers: int = 8
  ):
    self.bbox_extractor = bbox_extractor
    self.bbox_summariser_fn = bbox_summariser_fn
    self.bbox_summary_to_crop_params_fn = bbox_summary_to_crop_params_fn
    self.collector = collector
    self.max_workers = max_workers

  def record_to_crop_params(self, record: dict) -> CropParams:
    with tempfile.NamedTemporaryFile(suffix='.ts') as video_file:
      video_file.write(record['mp4'])
      bbox_info = self.bbox_extractor.extract(Path(video_file.name))
      bbox_summary = self.bbox_summariser_fn(bbox_info)
      crop_params = self.bbox_summary_to_crop_params_fn(bbox_summary)
      return crop_params

  def process_dataset(self, dataset):
    failed_videos = []
    success_count = 0
    pbar = tqdm(desc="Processing records", unit="record")
    it = iter(dataset)
    pending = set()
    future_to_record = {}

    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        for _ in range(self.max_workers):
            try:
                record = next(it)
            except StopIteration:
                break
            f = executor.submit(self.record_to_crop_params, record)
            pending.add(f)
            future_to_record[f] = record

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                record = future_to_record.pop(future)
                try:
                    params = future.result()
                    self.collector.collect(params, record['__key__'])
                    success_count += 1
                except Exception as e:
                    key = record['__key__']
                    logger.exception(f'failed to extract crop params for {key}')
                    failed_videos.append({"video_name": key, "exception": str(e)})
                finally:
                    pbar.update(1)
                    pbar.set_postfix({"success": success_count, "failed videos": len(failed_videos)})

                try:
                    next_record = next(it)
                    f = executor.submit(self.record_to_crop_params, next_record)
                    pending.add(f)
                    future_to_record[f] = next_record
                except StopIteration:
                    pass

    return failed_videos

  def __call__(self, dataset: Dataset) -> list:
    return self.process_dataset(dataset)
