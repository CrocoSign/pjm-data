import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from crop.pipeline import ExtractionPipeline
from crop.implementation import YOLOBoundingBoxExtractor, LMDBCollector, bbox_summariser_from_xyxy_value, bbox_summary_to_crop_params
import datasets
from pathlib import Path
from csv import DictWriter
import lmdb
import argparse

def main():
    FOUR_GB = 4* 1024**3
    FAILED_FILE_NAME = 'failed.csv'
    LMDB_FOLDER_NAME = 'crop_params.lmdb'
    parser = argparse.ArgumentParser(description='Extract crop parameters from video dataset')
    parser.add_argument('--output', type=Path, default='output',
                        help='Path to output folder for failed videos and lmdb (default: output)')
    parser.add_argument('--max-workers', type=int, default=8,
                        help='Maximum number of worker threads (default: 8)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run YOLO model on (default: cuda:0)')
    parser.add_argument('--yolo-model', type=str, default='yolo11n.pt',
                        help='YOLO model file (default: yolo11n.pt)')
    parser.add_argument('--db-name', type=str, default='output/crop_params.lmdb',
                        help='LMDB database name (default: crop_params.lmdb)')
    parser.add_argument('--db-size', type=int, default=FOUR_GB,
                        help='LMDB database size in bytes (default: 4GB)')
    parser.add_argument('--dataset-path', type=str, default='croco-corp/pjm-segments',
                        help='HuggingFace dataset path (default: croco-corp/pjm-segments)')
    
    args = parser.parse_args()

    output_folder: Path = args.output
    output_folder.mkdir(exist_ok=True, parents=True)
    lmdb_folder_path = output_folder / LMDB_FOLDER_NAME
    failed_videos_file = output_folder / FAILED_FILE_NAME
    lmdb_environment = lmdb.open(
        str(lmdb_folder_path.absolute()),
        map_size=FOUR_GB,
        max_dbs=0,
        max_readers=64,
        writemap=False,
        sync=True,
        metasync=True
    )
    collector = LMDBCollector(lmdb_environment)
    extractor = YOLOBoundingBoxExtractor(args.yolo_model, args.device)

    pipe = ExtractionPipeline(
        bbox_extractor=extractor,
        bbox_summariser_fn=bbox_summariser_from_xyxy_value,
        bbox_summary_to_crop_params_fn=bbox_summary_to_crop_params,
        collector=collector,
        max_workers=args.max_workers
    )

    dataset = datasets.load_dataset(args.dataset_path, split='train')
    
    failed_videos = pipe(dataset)
    with open(str(failed_videos_file.absolute()), 'w') as f:
        writer = DictWriter(f, ["video_name", "exception"])
        writer.writeheader()
        writer.writerows(failed_videos)

if __name__ == "__main__":
    main()
