#!/usr/bin/env python3
"""
AutoEdit CLI - Command-line interface for the unified autoedit pipeline.
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on system environment variables

from autoedit_pipeline import AutoEditPipeline, PipelineConfig
from encoding_pipeline import EncodingPipeline, EncodingConfig

logger = logging.getLogger('autoedit_cli')


def cmd_run(args):
    """Execute the full pipeline."""

    # Load from config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)

        # Override with CLI arguments if provided
        if args.videos:
            config_dict['video_paths'] = args.videos
        if args.prompt:
            config_dict['user_prompt'] = args.prompt
        if args.run_id:
            config_dict['run_id'] = args.run_id
        if args.output_dir != './data/runs':
            config_dict['output_base_dir'] = args.output_dir
        if args.encodings_dir:
            config_dict['encodings_dir'] = args.encodings_dir
        if args.log_level != 'INFO':
            config_dict['logging_level'] = args.log_level

        if any(k in config_dict for k in ("runtime", "story", "timeline", "otio")):
            config = PipelineConfig.from_nested_dict(config_dict)
        else:
            config = PipelineConfig.from_flat_dict(config_dict)
    else:
        # Build config from CLI arguments
        if not args.videos or not args.prompt:
            logger.error("Error: --videos and --prompt are required when not using --config")
            return 1

        config = PipelineConfig(
            video_paths=args.videos,
            user_prompt=args.prompt,
            output_base_dir=args.output_dir,
            run_id=args.run_id,
            # Encoding discovery
            encodings_dir=args.encodings_dir,
            video_encoding_path=args.video_encoding,
            frame_encodings_path=args.frame_encodings,
            frame_encodings_metadata_path=args.frame_metadata,
            # Stage controls
            story_enabled=not args.skip_story,
            timeline_enabled=not args.skip_timeline,
            otio_enabled=not args.skip_otio,
            # Model settings
            story_provider=args.story_provider,
            story_model=args.story_model,
            # Logging
            logging_level=args.log_level
        )

    pipeline = AutoEditPipeline(config)
    try:
        results = pipeline.run()
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


def cmd_list_runs(args):
    """List all pipeline runs."""
    runs_dir = Path(args.output_dir)

    if not runs_dir.exists():
        print(f"No runs directory found: {runs_dir}")
        return 1

    runs = []
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if run_dir.is_dir():
            manifest_path = run_dir / "run_manifest.json"
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                    runs.append({
                        'id': run_dir.name,
                        'timestamp': manifest.get('timestamp', ''),
                        'status': manifest.get('status', 'unknown'),
                        'prompt': manifest.get('config', {}).get('user_prompt', '')[:60]
                    })

    if not runs:
        print("No runs found")
        return 0

    print(f"\nFound {len(runs)} run(s):\n")
    print(f"{'ID':<20} {'Timestamp':<20} {'Status':<10} {'Prompt'}")
    print("-" * 100)
    for run in runs:
        print(f"{run['id']:<20} {run['timestamp'][:19]:<20} {run['status']:<10} {run['prompt']}...")
    print()
    return 0


def cmd_show_run(args):
    """Show details of a specific run."""
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "run_manifest.json"

    if not manifest_path.exists():
        print(f"No manifest found: {manifest_path}")
        return 1

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"\n{'='*80}")
    print(f"Run: {manifest['run_id']}")
    print(f"{'='*80}")
    print(f"Status: {manifest['status']}")
    print(f"Timestamp: {manifest['timestamp']}")
    print(f"Duration: {manifest.get('duration_seconds', 0):.1f}s")
    if manifest.get('error'):
        print(f"Error: {manifest['error']}")

    print(f"\nConfiguration:")
    print(f"  Prompt: {manifest['config']['user_prompt']}")
    print(f"  Videos: {len(manifest['config']['video_paths'])} file(s)")

    print(f"\nOutputs:")
    for stage, path in manifest['outputs'].items():
        exists = "✓" if path and Path(path).exists() else "✗"
        print(f"  {exists} {stage}: {path or 'N/A'}")
    print()
    return 0


def cmd_run_video_encoding(args):
    """Run video encoding only."""
    # Load from config file if provided
    if args.config:
        config = EncodingConfig.from_json(args.config)

        # Override with CLI arguments if provided
        if args.videos:
            config.video_paths = args.videos
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.fps:
            config.video_encoder_fps = args.fps
        if args.log_level != 'INFO':
            config.logging_level = args.log_level
    else:
        # Build config from CLI arguments
        if not args.videos or not args.output_dir:
            logger.error("Error: --videos and --output-dir are required when not using --config")
            return 1

        config = EncodingConfig(
            video_paths=args.videos,
            output_dir=args.output_dir,
            video_encoder_enabled=True,
            frame_encoder_enabled=False,
            video_encoder_fps=args.fps,
            logging_level=args.log_level
        )

    pipeline = EncodingPipeline(config)
    try:
        results = pipeline.run()
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


def cmd_run_frame_encoding(args):
    """Run frame encoding only."""
    # Load from config file if provided
    if args.config:
        config = EncodingConfig.from_json(args.config)

        # Override with CLI arguments if provided
        if args.videos:
            config.video_paths = args.videos
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.model:
            config.frame_encoder_model = args.model
        if args.fps:
            config.frame_encoder_fps = args.fps
        if args.log_level != 'INFO':
            config.logging_level = args.log_level
    else:
        # Build config from CLI arguments
        if not args.videos or not args.output_dir:
            logger.error("Error: --videos and --output-dir are required when not using --config")
            return 1

        config = EncodingConfig(
            video_paths=args.videos,
            output_dir=args.output_dir,
            video_encoder_enabled=False,
            frame_encoder_enabled=True,
            frame_encoder_model=args.model,
            frame_encoder_fps=args.fps,
            logging_level=args.log_level
        )

    pipeline = EncodingPipeline(config)
    try:
        results = pipeline.run()
        return 0
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="AutoEdit - Unified AI-powered video editing pipeline"
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # RUN command
    run_parser = subparsers.add_parser('run', help='Run the creative pipeline (story, timeline, OTIO)')
    run_parser.add_argument('--config', help='JSON config file (if provided, other args are optional overrides)')
    run_parser.add_argument('--videos', nargs='+', help='Video file paths')
    run_parser.add_argument('--prompt', help='Creative vision prompt')
    run_parser.add_argument('--output-dir', default='./data/runs', help='Output directory')
    run_parser.add_argument('--run-id', help='Custom run ID (default: YYYY-MM-DD)')

    # Encoding discovery
    run_parser.add_argument('--encodings-dir', help='Directory containing /encodings subfolder')
    run_parser.add_argument('--video-encoding', help='Path to existing video_encoding.json')
    run_parser.add_argument('--frame-encodings', help='Path to existing frame_encodings.npy')
    run_parser.add_argument('--frame-metadata', help='Path to existing frame_encodings_metadata.json')

    # Stage controls
    run_parser.add_argument('--skip-story', action='store_true', help='Skip story generation stage')
    run_parser.add_argument('--skip-timeline', action='store_true', help='Skip timeline generation stage')
    run_parser.add_argument('--skip-otio', action='store_true', help='Skip OTIO export stage')

    # Model settings
    run_parser.add_argument('--story-provider', default='gemini', choices=['gemini', 'local'], help='LLM provider')
    run_parser.add_argument('--story-model', default='gemini-2.5-flash', help='LLM model name')
    run_parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging verbosity level')
    run_parser.set_defaults(func=cmd_run)

    # RUN-VIDEO-ENCODING command
    video_enc_parser = subparsers.add_parser('run-video-encoding', help='Run video encoding only')
    video_enc_parser.add_argument('--config', help='JSON config file')
    video_enc_parser.add_argument('--videos', nargs='+', help='Video file paths')
    video_enc_parser.add_argument('--output-dir', help='Output directory')
    video_enc_parser.add_argument('--fps', type=float, default=0.1, help='Video encoding FPS')
    video_enc_parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging verbosity level')
    video_enc_parser.set_defaults(func=cmd_run_video_encoding)

    # RUN-FRAME-ENCODING command
    frame_enc_parser = subparsers.add_parser('run-frame-encoding', help='Run frame encoding only')
    frame_enc_parser.add_argument('--config', help='JSON config file')
    frame_enc_parser.add_argument('--videos', nargs='+', help='Video file paths')
    frame_enc_parser.add_argument('--output-dir', help='Output directory')
    frame_enc_parser.add_argument('--model', default='siglip', choices=['siglip', 'clip'], help='Frame encoder model')
    frame_enc_parser.add_argument('--fps', type=int, default=1, help='Frame encoding FPS')
    frame_enc_parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging verbosity level')
    frame_enc_parser.set_defaults(func=cmd_run_frame_encoding)

    # LIST-RUNS command
    list_parser = subparsers.add_parser('list-runs', help='List all runs')
    list_parser.add_argument('--output-dir', default='./data/runs', help='Runs directory')
    list_parser.set_defaults(func=cmd_list_runs)

    # SHOW-RUN command
    show_parser = subparsers.add_parser('show-run', help='Show run details')
    show_parser.add_argument('run_dir', help='Path to run directory')
    show_parser.set_defaults(func=cmd_show_run)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
