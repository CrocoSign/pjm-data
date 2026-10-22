import tempfile
import argparse
from pathlib import Path
import logging

from webdataset import ShardWriter # type: ignore
from .pipeline import SegmentationPipeline, PipelineConfig
from .processing import FFMPEGSegmentProcessor, VTTCaptionsProcessor
from .transfer import WebDatasetSegmentWriter, FileDownloader, HuggingFacePoster

def main():
    parser = argparse.ArgumentParser(description='CrocoSign segmentation pipeline')
    
    parser.add_argument('--hf-repo', default='croco-corp/pjm-segments', 
                       help='HuggingFace repository name (default: croco-corp/pjm-segments)')
    parser.add_argument('--offset-millis', type=int, default=100,
                       help='Offset in milliseconds (default: 100)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum number of workers (default: 4)')
    parser.add_argument('--output-dir', default='logs/',
                       help='Output directory (default: logs/)')
    parser.add_argument('--files-per-shard', type=int, default=500,
                        help="Files per huggingface repository shard (default: 500)")
    parser.add_argument('--csv-path', required=True,
                       help='Path to CSV file (required)')
    parser.add_argument('--gop', type=int, default=2,
                        help='Number of frames between two keyframes (default: 2)')
    parser.add_argument('--bf', type=int, default=0,
                        help='Number of b-frames during video re-encoding (default: 0)')
    parser.add_argument('--loglevel', default="INFO",
                     help="Log level (default: INFO)")
    parser.add_argument('--start_shard', type=int, default=0,
                        help='Shard file number to start from (default: 0)')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging._nameToLevel[args.loglevel])
    logger = logging.getLogger(__name__)
    
    downloader = FileDownloader()
    
    hf_poster = HuggingFacePoster(args.hf_repo)
    shards_dir = tempfile.TemporaryDirectory()
    logger.debug(f"shards dir at: {str(shards_dir)}")
    shard_writer = ShardWriter(f"{shards_dir.name}/train-%06d.tar", maxcount=args.files_per_shard, post=hf_poster.post, start_shard=args.start_shard)
    
    segment_writer = WebDatasetSegmentWriter(shard_writer)
    
    captions_processor = VTTCaptionsProcessor(args.offset_millis)
    
    config = PipelineConfig(
        max_workers=args.max_workers,
        output_dir=args.output_dir,
        gop_value=args.gop,
        b_frames=args.bf
    )
    
    pipeline = SegmentationPipeline(
        downloader,
        captions_processor,
        FFMPEGSegmentProcessor,
        segment_writer,
        config
    )
    
    try:
        pipeline(Path(args.csv_path))
    except Exception:
        logger.exception("pipeline failed.")
    
    shard_writer.close()

if __name__ == "__main__":
    main()