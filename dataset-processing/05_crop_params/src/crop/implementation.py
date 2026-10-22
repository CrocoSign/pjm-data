from .pipeline import BoundingBoxExtractor, BoundingBoxInfo, BoundingBoxSummary, CropParams, Padding, Collector
from dataclasses import asdict
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from lmdb import Environment
import msgpack
from math import floor, ceil
import torch

class YOLOBoundingBoxExtractor(BoundingBoxExtractor):
  PEOPLE_YOLO_CLASS = 0

  def __init__(self, model_name: str, device: str):
    self.device = device
    self.model_name = model_name

  def extract(self, video_path: Path) -> BoundingBoxInfo:
    frame_stream = YOLO(self.model_name)(
      str(video_path.absolute()),
      stream=True,
      classes=self.PEOPLE_YOLO_CLASS,
      device=self.device,
      verbose=False
    )

    xywh_cumulative = []
    xyxy_cumulative = []
    for result in frame_stream:
      for bbox_xyxy in result.boxes.xyxy:
        xyxy_cumulative.append(bbox_xyxy)

      for bbox_xywh in result.boxes.xywh:
        xywh_cumulative.append(bbox_xywh)
    
    if len(xywh_cumulative) == 0 or len(xyxy_cumulative) == 0:
      raise Exception("bounding box not found") 

    xywh_cumulative = torch.stack(xywh_cumulative)
    xyxy_cumulative = torch.stack(xyxy_cumulative)
    img_shape = result.orig_shape

    return BoundingBoxInfo(
      xywh_cumulative=xywh_cumulative.cpu().numpy(),
      xyxy_cumulative=xyxy_cumulative.cpu().numpy(),
      img_shape=img_shape
    )


def bbox_summariser_from_xyxy_value(bbox_info: BoundingBoxInfo) -> BoundingBoxSummary:
  # [box_start, box_end)
  box_start = np.percentile(bbox_info.xyxy_cumulative[:, 0], 1)
  box_start = floor(box_start)
  box_end = np.percentile(bbox_info.xyxy_cumulative[:, 2], 99)
  box_end = ceil(box_end)
  box_width = box_end - box_start

  x_center = (box_end + box_start)/2

  video_height = bbox_info.img_shape[0]
  video_width = bbox_info.img_shape[1]
  return BoundingBoxSummary(
      x_middle=x_center,
      y_middle=video_height/2,
      box_width=box_width,
      box_height=video_height,
      video_height=video_height,
      video_width=video_width
  )

def split_padding_val(overall_padding: int) -> tuple[int, int]:
  padding_val = int(overall_padding/2)
  if overall_padding%2 == 0:
    return (padding_val, padding_val)
  else:
    return (padding_val, padding_val+1)

def bbox_summary_to_crop_params(bbox_summary: BoundingBoxSummary):
  box_height = bbox_summary.box_height
  box_width = bbox_summary.box_width
  video_height = bbox_summary.video_height
  video_width = bbox_summary.video_width

  size = max(box_width, box_height)

  horizontal_padding = 0
  vertical_padding = 0

  x_start = floor(bbox_summary.x_middle - (size/2))
  x_end = x_start + size
  if x_start < 0:
    horizontal_padding += abs(x_start)
    x_start = 0
  if x_end > video_width:
    horizontal_padding += x_end - video_width
    x_end = video_width

  y_start = floor(bbox_summary.y_middle - (size/2))
  y_end = y_start + size
  if y_start < 0:
    vertical_padding += abs(y_start)
    y_start = 0
  if y_end > video_height:
    vertical_padding += y_end - video_height
    y_end = video_height

  padding = Padding()
  if horizontal_padding != 0 or vertical_padding != 0:
    (left, right) = split_padding_val(horizontal_padding)
    (top, bottom) = split_padding_val(vertical_padding)
    padding = Padding(
        top=top,
        bottom=bottom,
        left=left,
        right=right
    )

  return CropParams(
      x_start=x_start,
      y_start=y_start,
      x_end=x_end,
      y_end=y_end,
      padding=padding
  )

class LMDBCollector(Collector):
  def __init__(self, lmdb_environment: Environment):
    self.environment = lmdb_environment

  def collect(self, crop_params: CropParams, video_id: str):
    key = video_id.encode()
    value = msgpack.packb(asdict(crop_params))
    with self.environment.begin(write=True) as transaction:
      transaction.put(key, value)