"""
Unified AutoEdit Pipeline Controller

Orchestrates the complete autoedit workflow from video encoding to OTIO timeline generation.
Organizes outputs in per-run folders and enables easy component reuse.
"""

import os
import json
import time
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from config import PipelineConfig


class AutoEditPipeline:
    """
    Unified orchestrator for the complete autoedit pipeline.

    Pipeline stages:
    1. Video Encoder (Qwen2-VL) → video_encoding.json
    2. Frame Encoder (CLIP/SigLIP) → frame_encodings.npy + metadata
    3. Story Pipeline (2 agents) → story.json + storyboard.json
    4. Timeline Generator → timeline_json.json + metadata
    5. OTIO Parser → timeline_otio.otio
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.run_dir = None
        self.results = {}
        self.start_time = None

        # Configure logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging for the pipeline."""
        log_level = getattr(logging, self.config.runtime.logging_level.upper(), logging.INFO)

        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(message)s',  # Simple format for clean output
            force=True  # Override any existing configuration
        )

        # Create logger for this module
        self.logger = logging.getLogger('autoedit_pipeline')
        self.logger.setLevel(log_level)

    def _validate_encodings(self):
        """Validate that required encodings exist before starting pipeline."""
        self.encoding_paths = {}

        if not os.path.exists(self.config.encoding_manifest_path):
            raise FileNotFoundError(f"Encoding manifest not found: {self.config.encoding_manifest_path}")
        if not os.path.exists(self.config.video_encoding_path):
            raise FileNotFoundError(f"Video encoding not found: {self.config.video_encoding_path}")
        if not os.path.exists(self.config.frame_encodings_path):
            raise FileNotFoundError(f"Frame encodings not found: {self.config.frame_encodings_path}")
        if self.config.frame_encodings_metadata_path and not os.path.exists(self.config.frame_encodings_metadata_path):
            raise FileNotFoundError(f"Frame encodings metadata not found: {self.config.frame_encodings_metadata_path}")

        self.encoding_paths['encoding_manifest'] = self.config.encoding_manifest_path
        self.encoding_paths['video_encoding'] = self.config.video_encoding_path
        self.encoding_paths['frame_encodings'] = self.config.frame_encodings_path
        self.encoding_paths['frame_metadata'] = self.config.frame_encodings_metadata_path
        self.logger.debug("Using explicit encoding paths")

    def _load_encodings(self):
        """Load encodings from provided paths."""
        import numpy as np

        self.logger.info(f"Loading Pre-computed Encodings")
        self.logger.info("-" * 80)

        # Load video encoding
        with open(self.encoding_paths['video_encoding'], 'r') as f:
            video_data = json.load(f)

        self.results['video_encoding'] = {
            'path': self.encoding_paths['video_encoding'],
            'data': video_data
        }
        self.logger.debug(f"✓ Loaded video encoding from: {self.encoding_paths['video_encoding']}")

        # Load frame encodings
        embeddings = np.load(self.encoding_paths['frame_encodings'])

        with open(self.encoding_paths['encoding_manifest'], 'r') as f:
            encoding_manifest = json.load(f)
        
        frame_encoder_model = encoding_manifest['encoders']['frame_encoder']['model']

        # Load metadata if available
        metadata = []
        if self.encoding_paths['frame_metadata'] and os.path.exists(self.encoding_paths['frame_metadata']):
            with open(self.encoding_paths['frame_metadata'], 'r') as f:
                metadata = json.load(f).get('metadata', [])

        self.results['frame_encoding'] = {
            'frame_encoder_model': frame_encoder_model,
            'embeddings': embeddings,
            'metadata': metadata,
            'paths': {
                'npy': self.encoding_paths['frame_encodings'],
                'json': self.encoding_paths['frame_metadata']
            }
        }

    def run(self) -> Dict[str, Any]:
        """Execute the complete pipeline."""
        self.start_time = time.time()

        # 1. Create run directory
        self.run_dir = self._create_run_directory()
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"AutoEdit Pipeline Started")
        self.logger.info(f"{'='*80}")
        self.logger.info(f"Run Directory: {self.run_dir}\n")

        try:
            # 2. Validate encodings exist
            self._validate_encodings()

            # 3. Load encodings
            self._load_encodings()

            # 4. Execute creative stages
            if self.config.story.enabled:
                self._run_story_pipeline()

            if self.config.timeline.enabled:
                self._run_timeline_generator()

            if self.config.otio.enabled:
                self._run_otio_parser()

            # 5. Save run manifest
            self._save_run_manifest(status='success')

            elapsed = time.time() - self.start_time
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"✓ Pipeline Complete! ({elapsed:.1f}s)")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"Results: {self.run_dir}\n")

            return self.results

        except Exception as e:
            self._save_run_manifest(status='error', error=str(e))
            self.logger.error(f"\n{'='*80}")
            self.logger.error(f"✗ Pipeline Failed: {e}")
            self.logger.error(f"{'='*80}\n")
            raise

    def _create_run_directory(self) -> str:
        """Create per-run directory with date-based or custom naming."""
        if self.config.runtime.run_id:
            run_name = self.config.runtime.run_id
        else:
            run_name = datetime.now().strftime("%Y-%m-%d")

        run_path = os.path.join(self.config.runtime.output_base_dir, run_name)

        # Handle multiple runs per day
        if os.path.exists(run_path) and not self.config.runtime.run_id:
            counter = 1
            while os.path.exists(f"{run_path}_{counter}"):
                counter += 1
            run_path = f"{run_path}_{counter}"

        os.makedirs(run_path, exist_ok=True)
        return run_path

    def _run_story_pipeline(self):
        """Stage 1: Generate storyboard with 2-agent system."""
        self.logger.info(f"[1/3] Story Pipeline (Screenwriter + Editor)")
        self.logger.info("-" * 80)

        from story_pipeline import StoryPipeline

        video_summaries_path = self.results['video_encoding']['path']

        # Pass nested configs directly - no manual field mapping needed!
        pipeline = StoryPipeline(
            story_config=self.config.story,
            runtime_config=self.config.runtime
        )
        storyboard = pipeline.generate_storyboard(
            user_prompt=self.config.user_prompt,
            video_summaries_path=video_summaries_path,
            output_dir=self.run_dir
        )

        self.results['story'] = {
            'storyboard': storyboard,
            'storyboard_path': os.path.join(self.run_dir, "storyboard.json"),
            'story_path': os.path.join(self.run_dir, "story.json")
        }
        self.logger.debug(f"✓ Saved: story.json + storyboard.json\n")

    def _run_timeline_generator(self):
        """Stage 2: Match storyboard to video frames."""
        self.logger.info(f"[2/3] Timeline Generator")
        self.logger.info("-" * 80)

        from timeline_generator import TimelineGenerator

        encoding_model = self.results['frame_encoding']['frame_encoder_model']

        generator = TimelineGenerator(
            model_type=encoding_model,
            debugging=self.config.timeline.debugging,
            logging_level=self.config.runtime.logging_level
        )

        storyboard_path = self.results['story']['storyboard_path']
        embeddings_path = self.results['frame_encoding']['paths']['npy']

        timeline = generator.generate_timeline(
            storyboard_path=storyboard_path,
            video_embeddings_path=embeddings_path,
            moments_per_scene=self.config.timeline.moments_per_scene,
            output_dir=self.run_dir
        )

        self.results['timeline'] = {
            'timeline': timeline,
            'timeline_path': os.path.join(self.run_dir, "timeline_json.json"),
            'metadata_path': os.path.join(self.run_dir, "timeline_metadata.json") if self.config.timeline.debugging else None
        }
        self.logger.debug(f"✓ Saved: timeline_json.json\n")

    def _run_otio_parser(self):
        """Stage 3: Export to OpenTimelineIO format."""
        self.logger.info(f"[3/3] OTIO Parser")
        self.logger.info("-" * 80)

        from otio_parser import OtioTimeline

        parser = OtioTimeline(logging_level=self.config.runtime.logging_level)
        timeline_path = self.results['timeline']['timeline_path']
        metadata_path = self.results.get('frame_encoding', {}).get('paths', {}).get('json')
        output_path = os.path.join(self.run_dir, "timeline_otio.otio")

        otio_timeline = parser.generate_otio_timeline(
            timeline_path=timeline_path,
            metadata_path=metadata_path,
            output_path=output_path
        )

        self.results['otio'] = {
            'path': output_path,
            'timeline': otio_timeline
        }
        self.logger.debug(f"✓ Saved: {output_path}\n")

    def _save_run_manifest(self, status: str, error: Optional[str] = None):
        """Save run metadata and configuration."""
        manifest = {
            'run_id': os.path.basename(self.run_dir),
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': time.time() - self.start_time if self.start_time else 0,
            'config': {
                'video_paths': self.config.video_paths,
                'user_prompt': self.config.user_prompt,
                'story_provider': self.config.story.screenwriter.provider,
                'story_model': self.config.story.screenwriter.model
            },
            'outputs': {
                'video_encoding': self.results.get('video_encoding', {}).get('path'),
                'frame_encodings': self.results.get('frame_encoding', {}).get('paths', {}).get('npy') if self.results.get('frame_encoding') and self.results.get('frame_encoding').get('paths') else None,
                'storyboard': self.results.get('story', {}).get('storyboard_path'),
                'timeline': self.results.get('timeline', {}).get('timeline_path'),
                'otio': self.results.get('otio', {}).get('path')
            },
            'status': status,
            'error': error
        }

        manifest_path = os.path.join(self.run_dir, "run_manifest.json")
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
