# What is it?
Part of the automatic PJM translator project. A Python-based data pipeline that extracts optimal crop parameters from sign language videos using computer vision. The pipeline processes video datasets to detect human subjects using YOLO object detection, calculates bounding box statistics across frames, and generates crop parameters that can be used to frame signers consistently. Results are stored in an LMDB database for efficient access.

# Technical stack
- YOLO (YOLOv11) for real-time object detection and bounding box extraction
- LMDB for high-performance key-value storage of crop parameters
- HuggingFace Datasets for video dataset loading
- Multithreading for parallel video processing
