# SwiftCutter - AI-Powered Video Editing Pipeline

SwiftCutter is an intelligent video editing pipeline that uses AI models to automatically create compelling stories from raw footage. It combines vision-language models, frame embeddings, and creative agents to generate ready-to-edit timelines in OpenTimelineIO (OTIO) format.

## Highlights

- **AI-Powered Storytelling**: Uses advanced AI models to understand video content and generate creative narratives
- **Dual-Agent Architecture**: Screenwriter and Editor agents collaborate to create balanced, engaging stories
- **Industry Standard Output**: Generates OTIO timelines compatible with DaVinci Resolve, Premiere Pro, and Final Cut Pro
- **Fast Iteration**: Separate encoding and creative pipelines for rapid experimentation
- **Observability**: Built-in Langfuse integration for monitoring and debugging
- **Local Encoding**: All video and frame encoding happens locally
- **Local Timeline Generation**: All timeline generation happens locally
- **Optimized for Apple Silicon**: Uses Apple's Metal Performance Shaders for fast inference

## Architecture

The pipeline is split into two stages for optimal performance:

### 1. Encoding Pipeline (Run Once)
Computationally expensive preprocessing that needs to be done only once per video set:
- **Video Encoding**: Qwen2-VL analyzes video content and generates semantic descriptions
- **Frame Encoding**: CLIP/SigLIP creates frame embeddings for visual search

### 2. Creative Pipeline (Iterate Quickly)
Fast creative stages that can be run multiple times with different prompts:
- **Story Generation**: 2-agent system (Screenwriter + Editor) creates narrative structure
- **Timeline Matching**: Maps story beats to actual video frames using embeddings
- **OTIO Export**: Generates industry-standard timeline files for editors

This separation allows you to:
- Generate encodings once (expensive, 5-20 minutes)
- Iterate on creative direction multiple times (fast, 1-2 minutes)
- Try different prompts without re-encoding videos

## Installation

### Prerequisites

- **Python 3.8+**
- **FFmpeg** (required for video processing)
  - Download and installation guide: https://www.ffmpeg.org/download.html
  - Platform-specific installation:
    - **macOS**: `brew install ffmpeg`
    - **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
    - **Windows**: Download from the link above and add to PATH
- **Sufficient disk space** for video files and encodings (several GB recommended)

git clone <repository-url>
cd SwiftCutter
```

2. Create and activate a virtual environment:
source .autoedit/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:

Create a `.env` file in the project root:
```bash
# For Gemini API (recommended)
GOOGLE_API_KEY=your_gemini_api_key_here

# Optional: Langfuse for observability
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Quick Start

### 1. Create Configs

See `sample_encoding_config.json` for encoding config and `sample_pipeline_config.json` for pipeline config.

### 2. Generate Encodings (One Time)

Edit `sample_encoding_config.json` with your video paths and run:

```bash
python -m src.autoedit_cli run-video-encoding --config sample_encoding_config.json
python -m src.autoedit_cli run-frame-encoding --config sample_encoding_config.json
```

### 3. Run Creative Pipeline (Fast Iteration)

```bash
python -m src.autoedit_cli run \
  --config sample_pipeline_config.json
```

### 4. Try Different Creative Directions

Reuse the same encodings with different prompts:

```bash
# Peaceful nature montage
python -m src.autoedit_cli run \
  --videos ./data/*.mp4 \
  --prompt "Create a peaceful nature montage with serene moments" \
  --encodings-dir ./data/runs/my-project \
  --run-id my-project-peaceful

# Documentary style
python -m src.autoedit_cli run \
  --videos ./data/*.mp4 \
  --prompt "Create a cinematic documentary about the journey" \
  --encodings-dir ./data/runs/my-project \
  --run-id my-project-documentary
```

## CLI Reference

### Running the CLI

The CLI can be run as a Python module in two ways:

```bash
# Full module path
python -m src.autoedit_cli --help

# Shorter version
python -m src --help
```

Both commands are equivalent. Use whichever you prefer.

### Commands Overview

```bash
python -m src --help
```

Available commands:
- `run` - Run the creative pipeline (story, timeline, OTIO)
- `run-video-encoding` - Run video encoding only
- `run-frame-encoding` - Run frame encoding only
- `list-runs` - List all pipeline runs
- `show-run` - Show details of a specific run

### Encoding Commands

#### Video Encoding

Generate semantic video descriptions using Qwen2-VL:

```bash
python -m src.autoedit_cli run-video-encoding \
  --videos video1.mp4 video2.mp4 \
  --output-dir ./data/runs/project-name \
  --fps 0.1 \
  --log-level INFO
```

Options:
- `--config` - JSON config file (optional)
- `--videos` - Video file paths (required)
- `--output-dir` - Output directory (required)
- `--fps` - Sampling rate in frames per second (default: 0.1)
- `--log-level` - Logging verbosity: DEBUG, INFO, WARNING, ERROR

#### Frame Encoding

Generate frame embeddings using CLIP or SigLIP:

```bash
python -m src.autoedit_cli run-frame-encoding \
  --videos video1.mp4 video2.mp4 \
  --output-dir ./data/runs/project-name \
  --model siglip \
  --fps 1 \
  --log-level INFO
```

Options:
- `--config` - JSON config file (optional)
- `--videos` - Video file paths (required)
- `--output-dir` - Output directory (required)
- `--model` - Encoder model: siglip or clip (default: siglip)
- `--fps` - Sampling rate in frames per second (default: 1)
- `--log-level` - Logging verbosity

### Creative Pipeline Command

Run the complete creative pipeline (requires pre-computed encodings):

```bash
python -m src.autoedit_cli run \
  --videos video1.mp4 video2.mp4 \
  --prompt "Your creative vision here" \
  --encodings-dir ./data/runs/project-name \
  --run-id custom-run-name \
  --log-level INFO
```

Options:
- `--config` - JSON config file (optional)
- `--videos` - Video file paths (required without config)
- `--prompt` - Creative vision prompt (required without config)
- `--output-dir` - Output base directory (default: ./data/runs)
- `--run-id` - Custom run identifier (default: YYYY-MM-DD format)
- `--encodings-dir` - Directory containing /encodings subfolder
- `--video-encoding` - Explicit path to video_encoding.json
- `--frame-encodings` - Explicit path to frame_encodings.npy
- `--frame-metadata` - Explicit path to frame_encodings_metadata.json
- `--skip-story` - Skip story generation stage
- `--skip-timeline` - Skip timeline generation stage
- `--skip-otio` - Skip OTIO export stage
- `--story-provider` - LLM provider: gemini or local (default: gemini)
- `--story-model` - LLM model name (default: gemini-2.5-flash)
- `--log-level` - Logging verbosity

### Management Commands

#### List Runs

View all pipeline runs:

```bash
python -m src.autoedit_cli list-runs
python -m src.autoedit_cli list-runs --output-dir ./data/runs
```

#### Show Run Details

View detailed information about a specific run:

```bash
python -m src.autoedit_cli show-run ./data/runs/my-project-epic
```

## Configuration Files

### Encoding Config (`sample_encoding_config.json`)

```json
{
  "video_paths": [
    "./data/video1.mp4",
    "./data/video2.mp4"
  ],
  "output_dir": "./data/runs/my-project",
  "video_encoder_enabled": true,
  "video_encoder_fps": 0.1,
  "video_encoder_model": "mlx-community/Qwen2-VL-7B-Instruct-4bit",
  "frame_encoder_enabled": true,
  "frame_encoder_model": "siglip",
  "frame_encoder_fps": 1,
  "frame_encoder_record_frames": false,
  "logging_level": "INFO"
}
```

### Pipeline Config (`sample_pipeline_config.json`)

```json
{
  "video_paths": [
    "./data/video1.mp4",
    "./data/video2.mp4"
  ],
  "user_prompt": "Create a compelling story from the footage",
  "output_base_dir": "./data/runs",
  "run_id": "my-project-v1",
  "encodings_dir": "./data/runs/my-project",
  "story_enabled": true,
  "story_provider": "gemini",
  "story_model": "gemini-2.5-flash",
  "story_screenwriter_temp": 0.6,
  "story_editor_temp": 0.2,
  "story_queries_per_scene": 4,
  "timeline_enabled": true,
  "timeline_moments_per_scene": 2,
  "timeline_debugging": false,
  "otio_enabled": true,
  "logging_level": "INFO"
}
```

## Output Structure

### After Encoding

```
data/runs/my-project/
└── encodings/
    ├── video_encoding.json              # Semantic video descriptions
    ├── frame_encodings.npy              # Frame embedding vectors
    ├── frame_encodings_metadata.json    # Frame timestamps and paths
    ├── encoding_config.json             # Config used for encoding
    └── encoding_manifest.json           # Encoding run metadata
```

### After Creative Pipeline Run

```
data/runs/my-project-epic/
├── story.json                    # Generated storyline
├── storyboard.json              # Scene breakdown with search queries
├── timeline_json.json           # Matched video frames to story beats
├── timeline_otio.otio           # OpenTimelineIO export (import to editors)
├── video_encoding.json          # Copied from encodings
├── frame_encodings.npy          # Copied from encodings
├── frame_encodings_metadata.json
└── run_manifest.json            # Run metadata and config
```

## Typical Workflows

### Workflow 1: Single Project, Multiple Iterations

```bash
# 1. Generate encodings once (expensive, ~10 minutes)
python -m src.autoedit_cli run-video-encoding \
  --videos ./footage/*.mp4 \
  --output-dir ./data/runs/vacation-trip

python -m src.autoedit_cli run-frame-encoding \
  --videos ./footage/*.mp4 \
  --output-dir ./data/runs/vacation-trip

# 2. Try different creative directions (fast, ~1-2 minutes each)
python -m src.autoedit_cli run \
  --videos ./footage/*.mp4 \
  --prompt "Epic adventure highlighting exciting moments" \
  --encodings-dir ./data/runs/vacation-trip \
  --run-id vacation-epic

python -m src.autoedit_cli run \
  --videos ./footage/*.mp4 \
  --prompt "Peaceful journey focusing on beautiful landscapes" \
  --encodings-dir ./data/runs/vacation-trip \
  --run-id vacation-peaceful

python -m src.autoedit_cli run \
  --videos ./footage/*.mp4 \
  --prompt "Documentary style with narration focus" \
  --encodings-dir ./data/runs/vacation-trip \
  --run-id vacation-documentary

# 3. Review results
python -m src.autoedit_cli list-runs
python -m src.autoedit_cli show-run ./data/runs/vacation-epic
```

### Workflow 2: Config-Based Workflow

```bash
# 1. Generate sample configs
python -m src.autoedit_cli create-sample-encoding-config
python -m src.autoedit_cli create-sample-pipeline-config

# 2. Edit configs with your settings
# Edit sample_encoding_config.json
# Edit sample_pipeline_config.json

# 3. Run encodings
python -m src.autoedit_cli run-video-encoding --config sample_encoding_config.json
python -m src.autoedit_cli run-frame-encoding --config sample_encoding_config.json

# 4. Run creative pipeline
python -m src.autoedit_cli run --config sample_pipeline_config.json
```

## Logging Levels

Control output verbosity with `--log-level`:

### INFO (default)
Shows major pipeline stages and completion:
```bash
python -m src.autoedit_cli run --config config.json --log-level INFO
```
Output:
- Stage headers ([1/3], [2/3], [3/3])
- Major completions
- Errors

### DEBUG
Shows detailed information:
```bash
python -m src.autoedit_cli run --config config.json --log-level DEBUG
```
Output:
- All INFO messages
- File paths and operations
- Scene enumeration
- Timing metrics
- Model initialization details

### WARNING
Shows only warnings and errors:
```bash
python -m src.autoedit_cli run --config config.json --log-level WARNING
```
Output:
- Warning messages
- Errors

## Advanced Usage

### Using Explicit Encoding Paths

Instead of `--encodings-dir`, you can specify exact paths:

```bash
python -m src.autoedit_cli run \
  --videos ./footage/*.mp4 \
  --prompt "Your creative vision" \
  --video-encoding ./custom/path/video_encoding.json \
  --frame-encodings ./custom/path/frame_encodings.npy \
  --frame-metadata ./custom/path/frame_encodings_metadata.json \
  --run-id custom-paths-run
```


## Troubleshooting

### Error: Video encodings not found

If you see:
```
ERROR: Video encodings not found!
```

Solution:
1. Generate encodings first using `run-video-encoding` and `run-frame-encoding`
2. Verify the encodings directory path
3. Use explicit paths with `--video-encoding` and `--frame-encodings`

### Error: Module not found

If you see `ModuleNotFoundError`:
```bash
# Make sure virtual environment is activated
source .autoedit/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### FFmpeg not found

Install FFmpeg:
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **Windows**: Download from https://ffmpeg.org/download.html

### Out of memory during encoding

For large videos:
1. Reduce frame sampling rate: `--fps 0.5` (for video encoding)
2. Reduce frame sampling rate: `--fps 0.5` (for frame encoding)
3. Process videos in smaller batches

## Performance Tips

1. **Use SigLIP over CLIP**: SigLIP is faster and more accurate
2. **Adjust sampling rates**: Lower FPS = faster encoding but less detail
3. **Reuse encodings**: Generate once, iterate on creative direction multiple times
4. **Use DEBUG logging sparingly**: INFO level is faster for production runs
5. **Process similar videos together**: Better for narrative coherence

## Importing to Video Editors

The pipeline generates `.otio` files compatible with:
- **DaVinci Resolve** (native support via Import>Timeline)
- **Adobe Premiere Pro**
- **Final Cut Pro**
- **Kdenlive**

To use in your editor:
1. Review how to import OTIO timelines in your editor
2. Import the `timeline_otio.otio` file from your run directory
3. Edit and refine as needed

## Contributing

Contributions are welcome! Please submit issues and pull requests to the repository.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Qwen2-VL** for video understanding
- **SigLIP/CLIP** for frame embeddings
- **Gemini** for creative story generation (as of now)
- **OpenTimelineIO** for timeline interchange format
