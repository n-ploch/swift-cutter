"""
Encoding Pipeline - Pre-processing stage for video and frame encoding.

Separates expensive encoding operations from the main creative pipeline,
enabling reuse of encodings across multiple story iterations.
"""

import os
import json
import time
import logging
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class EncodingConfig:
    """
    Configuration for encoding pipeline.

    This config is completely separate from PipelineConfig to enforce
    separation of concerns between encoding and creative stages.
    """

    # Required inputs
    video_paths: List[str]  # List of directories or video files (.mp4, .mov, .avi, .mkv)
    output_dir: str  # Where to create /encodings subfolder

    # Video Encoder settings (Stage 1)
    video_encoder_enabled: bool = True
    video_encoder_fps: float = 0.1
    video_encoder_model: str = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
    video_encoder_system_prompt: Optional[str] = None

    # Frame Encoder settings (Stage 2)
    frame_encoder_enabled: bool = True
    frame_encoder_model: str = "siglip"  # or "clip"
    frame_encoder_fps: int = 1
    frame_encoder_record_frames: bool = False

    # Logging
    logging_level: str = "INFO"

    @classmethod
    def from_json(cls, path: str) -> 'EncodingConfig':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


class EncodingPipeline:
    """
    Pre-processing pipeline for video encoding stages.

    Outputs encodings to /encodings folder within output_dir:
    - /encodings/video_encoding.json
    - /encodings/frame_encodings.npy
    - /encodings/frame_encodings_metadata.json
    - /encodings/encoding_manifest.json (metadata about this encoding run)
    """

    def __init__(self, config: EncodingConfig):
        self.config = config
        self.encodings_dir = None
        self.results = {}
        self.start_time = None

        # Configure logging
        self._setup_logging()

        # Discover video files from provided paths
        self.video_files = self._discover_video_files()

    def _setup_logging(self):
        """Configure logging for encoding pipeline."""
        log_level = getattr(logging, self.config.logging_level.upper(), logging.INFO)

        logging.basicConfig(
            level=log_level,
            format='%(message)s',
            force=True
        )

        self.logger = logging.getLogger('encoding_pipeline')
        self.logger.setLevel(log_level)

    def _discover_video_files(self) -> List[str]:
        """
        Discover video files from the provided paths.

        Handles both:
        - Directory paths: Scans for .mp4, .mov, .avi, .mkv files
        - File paths: Uses them directly

        Returns:
            List of absolute paths to video files

        Raises:
            ValueError: If no video files are found or paths don't exist
        """
        video_files = []
        supported_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.MP4', '.MOV', '.AVI', '.MKV'}

        for path in self.config.video_paths:
            abs_path = os.path.abspath(path)

            if os.path.isfile(abs_path):
                # Direct file path
                if any(abs_path.endswith(ext) for ext in supported_extensions):
                    video_files.append(abs_path)
                else:
                    self.logger.warning(f"Skipping non-video file: {abs_path}")

            elif os.path.isdir(abs_path):
                # Directory path - scan for video files
                found_in_dir = []
                for filename in sorted(os.listdir(abs_path)):
                    if any(filename.endswith(ext) for ext in supported_extensions):
                        video_path = os.path.join(abs_path, filename)
                        found_in_dir.append(video_path)
                        video_files.append(video_path)

                self.logger.info(f"Found {len(found_in_dir)} video(s) in: {abs_path}")

            else:
                raise ValueError(f"Path does not exist: {path}")

        if not video_files:
            raise ValueError(
                f"No video files found in provided paths: {self.config.video_paths}\n"
                f"Supported formats: {', '.join(sorted(supported_extensions))}"
            )

        self.logger.info(f"Total videos discovered: {len(video_files)}")
        return video_files

    def run(self) -> Dict[str, Any]:
        """
        Execute encoding pipeline.

        Returns:
            Dict with paths to generated encodings and metadata
        """
        self.start_time = time.time()

        # 1. Create encodings directory
        self.encodings_dir = self._create_encodings_directory()
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"Encoding Pipeline Started")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Encodings Directory: {self.encodings_dir}\n")

        try:
            # 2. Run encoding stages
            if self.config.video_encoder_enabled:
                self._run_video_encoder()

            if self.config.frame_encoder_enabled:
                self._run_frame_encoder()

            # 3. Save encoding manifest
            self._save_encoding_manifest(status='success')

            elapsed = time.time() - self.start_time
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"✓ Encoding Complete! ({elapsed:.1f}s)")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"Encodings saved to: {self.encodings_dir}\n")

            return self.results

        except Exception as e:
            self._save_encoding_manifest(status='error', error=str(e))
            self.logger.error(f"\n{'='*80}")
            self.logger.error(f"✗ Encoding Failed: {e}")
            self.logger.error(f"{'='*80}\n")
            raise

    def _create_encodings_directory(self) -> str:
        """Create /encodings subdirectory within output_dir."""
        encodings_path = os.path.join(self.config.output_dir, "encodings")
        os.makedirs(encodings_path, exist_ok=True)

        # Also save the config used for this encoding
        config_path = os.path.join(encodings_path, "encoding_config.json")
        self.config.to_json(config_path)

        return encodings_path

    def _run_video_encoder(self):
        """Stage 1: Generate video descriptions with Qwen2-VL."""
        self.logger.info(f"[1/2] Video Encoder (Qwen2-VL)")
        self.logger.info("-" * 80)

        from video_encoder import VideoProcessor

        processor = VideoProcessor(
            model_path=self.config.video_encoder_model,
            logging_level=self.config.logging_level
        )
        output_path = os.path.join(self.encodings_dir, "video_encoding.json")

        # Prepare system prompt
        system_prompt = self.config.video_encoder_system_prompt
        if system_prompt is None:
            system_prompt = "You are an expert cinematic analyst. Describe the video content in detail."

        results = processor.process_video_qwen_sequence(
            video_list=self.video_files,
            output_json=output_path,
            fps=self.config.video_encoder_fps,
            system_prompt=system_prompt
        )

        self.results['video_encoding'] = {
            'path': output_path,
            'data': results
        }
        self.logger.debug(f"✓ Saved: {output_path}\n")

    def _run_frame_encoder(self):
        """Stage 2: Extract frame embeddings with CLIP/SigLIP."""
        self.logger.info(f"[2/2] Frame Encoder ({self.config.frame_encoder_model.upper()})")
        self.logger.info("-" * 80)

        from frame_encoder import FrameIndexer

        indexer = FrameIndexer(
            model_type=self.config.frame_encoder_model,
            logging_level=self.config.logging_level
        )
        output_npy = os.path.join(self.encodings_dir, "frame_encodings.npy")

        result = indexer.process_video(
            video_paths=self.video_files,
            output_filename=output_npy,
            record_frames=self.config.frame_encoder_record_frames
        )

        self.results['frame_encoding'] = result
        self.logger.debug(f"✓ Saved: {output_npy}\n")

    def _save_encoding_manifest(self, status: str, error: Optional[str] = None):
        """Save metadata about this encoding run."""
        manifest = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - self.start_time if self.start_time else 0,
            'video_paths': self.video_files,
            'video_count': len(self.video_files),
            'encoders': {
                'video_encoder': {
                    'enabled': self.config.video_encoder_enabled,
                    'model': self.config.video_encoder_model,
                    'fps': self.config.video_encoder_fps,
                },
                'frame_encoder': {
                    'enabled': self.config.frame_encoder_enabled,
                    'model': self.config.frame_encoder_model,
                    'fps': self.config.frame_encoder_fps,
                }
            },
            'outputs': {
                'video_encoding': self.results.get('video_encoding', {}).get('path'),
                'frame_encodings_npy': self.results.get('frame_encoding', {}).get('paths', {}).get('npy') if self.results.get('frame_encoding') else None,
                'frame_encodings_json': self.results.get('frame_encoding', {}).get('paths', {}).get('json') if self.results.get('frame_encoding') else None,
            },
            'status': status,
            'error': error
        }

        manifest_path = os.path.join(self.encodings_dir, "encoding_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
