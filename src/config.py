"""
Unified configuration for the AI video autoedit pipeline.

Nested structure separates runtime config from stage-specific and agent-specific configs.
Inspired by LangGraph's configuration pattern.
"""
from dataclasses import dataclass, field
from typing import List, Optional


# ============================================================================
# Runtime Configuration (Cross-cutting concerns)
# ============================================================================

@dataclass
class RuntimeConfig:
    """Runtime configuration for the entire pipeline."""
    # Output management
    output_base_dir: str = "./data/runs"
    run_id: Optional[str] = None  # Auto-generated if None

    # Observability
    enable_langfuse: bool = True
    langfuse_session_id: Optional[str] = None

    # Logging
    logging_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR


# ============================================================================
# Story Stage Configuration
# ============================================================================

@dataclass
class ScreenwriterConfig:
    """Configuration for the Screenwriter agent."""
    provider: str = "gemini"  # "gemini" or "local" (Ollama)
    model: str = "gemini-2.5-flash"
    temperature: float = 0.6  # Creative storytelling


@dataclass
class EditorConfig:
    """Configuration for the Editor agent."""
    provider: str = "gemini"  # "gemini" or "local" (Ollama)
    model: str = "gemini-2.5-flash"
    temperature: float = 0.2  # Precise CLIP queries
    queries_per_scene: int = 4  # 3-5 range


@dataclass
class StoryConfig:
    """Configuration for the story pipeline stage."""
    enabled: bool = True

    # Agent configs (nested)
    screenwriter: ScreenwriterConfig = field(default_factory=ScreenwriterConfig)
    editor: EditorConfig = field(default_factory=EditorConfig)

    # Stage-specific settings
    save_intermediate: bool = False  # Save storyline JSON


# ============================================================================
# Timeline Stage Configuration
# ============================================================================

@dataclass
class TimelineConfig:
    """Configuration for the timeline generator stage."""
    enabled: bool = True
    moments_per_scene: int = 2
    debugging: bool = False  # Enable metadata collection


# ============================================================================
# OTIO Stage Configuration
# ============================================================================

@dataclass
class OtioConfig:
    """Configuration for the OTIO parser stage."""
    enabled: bool = True


# ============================================================================
# Top-Level Pipeline Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """
    Complete autoedit pipeline configuration.

    Nested structure:
    - runtime: Cross-cutting concerns (logging, paths, observability)
    - story: Story stage with nested agent configs
    - timeline: Timeline generator stage
    - otio: OTIO parser stage

    Example:
        config = PipelineConfig(
            video_paths=["./videos", "./footage"],  # Directories containing .mp4/.mov files
            user_prompt="Create a nature documentary",
            encoding_manifest_path="./data/encodings/encoding_manifest.json",
            video_encoding_path="./data/encodings/video_encodings.npy",
            frame_encodings_path="./data/encodings/frame_encodings.npy",
            frame_encodings_metadata_path="./data/encodings/frame_encodings_metadata.json",
            runtime=RuntimeConfig(enable_langfuse=True, logging_level="DEBUG"),
            story=StoryConfig(
                screenwriter=ScreenwriterConfig(temperature=0.7),
                editor=EditorConfig(queries_per_scene=5)
            )
        )
    """
    # Required inputs
    video_paths: List[str]  # List of directory paths containing video files (.mp4, .mov)
    user_prompt: str

    # Encoding discovery (required - encodings must be pre-computed)
    encoding_manifest_path: str = None
    video_encoding_path: str = None
    frame_encodings_path: str = None
    frame_encodings_metadata_path: str = None

    # Nested stage configurations
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    story: StoryConfig = field(default_factory=StoryConfig)
    timeline: TimelineConfig = field(default_factory=TimelineConfig)
    otio: OtioConfig = field(default_factory=OtioConfig)

    @classmethod
    def from_flat_dict(cls, data: dict) -> "PipelineConfig":
        """
        Convert flat JSON (old format) to nested config.

        Provides backward compatibility for existing JSON config files.

        Args:
            data: Flat dictionary with prefixed keys (story_provider, story_model, etc.)

        Returns:
            PipelineConfig with nested structure

        Example:
            {
                "video_paths": ["./data/videos"],
                "user_prompt": "Create a video",
                "story_provider": "gemini",
                "story_model": "gemini-2.5-flash",
                "story_screenwriter_temp": 0.6,
                "story_editor_temp": 0.2,
                "story_queries_per_scene": 4,
                "enable_langfuse": true,
                "logging_level": "INFO"
            }
        """
        return cls(
            video_paths=data["video_paths"],
            user_prompt=data["user_prompt"],
            encoding_manifest_path=data.get("encoding_manifest_path"),
            video_encoding_path=data.get("video_encoding_path"),
            frame_encodings_path=data.get("frame_encodings_path"),
            frame_encodings_metadata_path=data.get("frame_encodings_metadata_path"),
            runtime=RuntimeConfig(
                output_base_dir=data.get("output_base_dir", "./data/runs"),
                run_id=data.get("run_id"),
                enable_langfuse=data.get("enable_langfuse", True),
                langfuse_session_id=data.get("langfuse_session_id"),
                logging_level=data.get("logging_level", "INFO")
            ),
            story=StoryConfig(
                enabled=data.get("story_enabled", True),
                screenwriter=ScreenwriterConfig(
                    provider=data.get("story_provider", "gemini"),
                    model=data.get("story_model", "gemini-2.5-flash"),
                    temperature=data.get("story_screenwriter_temp", 0.6)
                ),
                editor=EditorConfig(
                    provider=data.get("story_provider", "gemini"),
                    model=data.get("story_model", "gemini-2.5-flash"),
                    temperature=data.get("story_editor_temp", 0.2),
                    queries_per_scene=data.get("story_queries_per_scene", 4)
                ),
                save_intermediate=data.get("save_intermediate", False)
            ),
            timeline=TimelineConfig(
                enabled=data.get("timeline_enabled", True),
                moments_per_scene=data.get("timeline_moments_per_scene", 2),
                debugging=data.get("timeline_debugging", False)
            ),
            otio=OtioConfig(
                enabled=data.get("otio_enabled", True)
            )
        )

    @classmethod
    def from_nested_dict(cls, data: dict) -> "PipelineConfig":
        """
        Convert nested JSON config into PipelineConfig.

        Accepts dicts for runtime/story/timeline/otio and their nested agent configs.
        """
        runtime_data = data.get("runtime", {}) or {}
        story_data = data.get("story", {}) or {}
        timeline_data = data.get("timeline", {}) or {}
        otio_data = data.get("otio", {}) or {}

        screenwriter_data = story_data.get("screenwriter", {}) or {}
        editor_data = story_data.get("editor", {}) or {}

        return cls(
            video_paths=data.get("video_paths", []),
            user_prompt=data.get("user_prompt", ""),
            encoding_manifest_path=data.get("encoding_manifest_path"),
            video_encoding_path=data.get("video_encoding_path"),
            frame_encodings_path=data.get("frame_encodings_path"),
            frame_encodings_metadata_path=data.get("frame_encodings_metadata_path"),
            runtime=RuntimeConfig(**runtime_data),
            story=StoryConfig(
                enabled=story_data.get("enabled", True),
                screenwriter=ScreenwriterConfig(**screenwriter_data),
                editor=EditorConfig(**editor_data),
                save_intermediate=story_data.get("save_intermediate", False)
            ),
            timeline=TimelineConfig(**timeline_data),
            otio=OtioConfig(**otio_data)
        )
