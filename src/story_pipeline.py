"""
Story Pipeline - Orchestrates the two-agent visual translator workflow.

This module coordinates Agent 1 (Screenwriter) and Agent 2 (Editor) to convert
user prompts and video footage into CLIP-optimized storyboards with full Langfuse observability.
"""

import os
import json
import uuid
import logging
from typing import List, Optional
from langfuse.langchain import CallbackHandler
from langfuse import get_client, propagate_attributes

from config import StoryConfig, RuntimeConfig
from models import VideoSummary, Storyline, Storyboard
from screenwriter_agent import ScreenwriterAgent
from editor_agent import EditorAgent
from tracing import trace

logger = logging.getLogger('story_pipeline')


class StoryPipeline:
    """
    Orchestrates the two-agent workflow for visual translation.

    Pipeline flow:
    1. Load video summaries
    2. Agent 1 (Screenwriter): User prompt + summaries → Storyline
    3. Agent 2 (Editor): Storyline + summaries → Storyboard
    4. Return Storyboard (compatible with timeline_generator.py)

    With full Langfuse tracing for observability.
    """

    def __init__(self, story_config: StoryConfig, runtime_config: RuntimeConfig):
        """
        Initialize the story pipeline with both agents.

        Args:
            story_config: Story stage configuration (agents, save_intermediate)
            runtime_config: Runtime configuration (logging, langfuse, paths)
        """
        self.story_config = story_config
        self.runtime_config = runtime_config

        # Configure logging
        log_level = getattr(logging, runtime_config.logging_level.upper(), logging.INFO)
        logger.setLevel(log_level)

        if runtime_config.enable_langfuse:
            self.langfuse = get_client()
            self.langfuse_handler = CallbackHandler()

        # Initialize Agent 1: Screenwriter with agent-specific config
        self.screenwriter = ScreenwriterAgent(
            config=story_config.screenwriter,
            runtime=runtime_config
        )

        # Initialize Agent 2: Editor with agent-specific config
        self.editor = EditorAgent(
            config=story_config.editor,
            runtime=runtime_config
        )

    @trace(as_type="chain", name="pipeline_generate_storyboard")
    def generate_storyboard(
        self,
        user_prompt: str,
        video_summaries_path: str,
        session_id: Optional[str] = None,
        save_intermediate: Optional[bool] = None,
        output_dir: Optional[str] = None
    ) -> Storyboard:
        """
        Generate a complete storyboard from user prompt and video footage.

        Args:
            user_prompt: User's creative vision for the video
            video_summaries_path: Path to JSON file with video summaries
            session_id: Optional Langfuse session ID for grouping related runs
            save_intermediate: Override config to save/skip intermediate storyline
            output_dir: Override config output directory

        Returns:
            Storyboard object with CLIP-optimized queries, ready for timeline_generator.py

        Raises:
            FileNotFoundError: If video_summaries_path doesn't exist
            ValidationError: If LLM outputs don't match expected schemas
            Exception: If agent execution fails
        """
        # Use config defaults if not overridden
        save_intermediate = save_intermediate if save_intermediate is not None else self.story_config.save_intermediate
        output_dir = output_dir or self.runtime_config.output_base_dir

        # Generate session_id if not provided (for Langfuse grouping)
        if session_id is None and self.runtime_config.enable_langfuse:
            session_id = f"pipeline_{uuid.uuid4().hex[:8]}"

        # Load video summaries
        logger.info(f"[StoryPipeline] Loading video summaries from: {video_summaries_path}")
        video_summaries = self._load_video_summaries(video_summaries_path)
        logger.info(f"[StoryPipeline] Loaded {len(video_summaries)} video summaries")

        if self.runtime_config.enable_langfuse:
            logger.debug(f"[StoryPipeline] Langfuse session: {session_id}")
        else:
            logger.debug(f"[StoryPipeline] Langfuse tracing disabled")

        try:
            # AGENT 1: Generate storyline
            logger.info("\n[StoryPipeline] === AGENT 1: Screenwriter ===")
            logger.info(f"[StoryPipeline] Generating narrative storyline...")

            storyline = self._run_screenwriter(
                user_prompt=user_prompt,
                video_summaries=video_summaries,
                session_id=session_id
            )

            logger.info(f"[StoryPipeline] ✓ Generated storyline with {len(storyline.scenes)} scenes")
            for i, scene in enumerate(storyline.scenes, 1):
                logger.debug(f"[StoryPipeline]   {i}. {scene.narrative_id} ({scene.estimated_duration}s)")

            # ALWAYS save storyline when output_dir is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                storyline_path = os.path.join(output_dir, "story.json")
                self._save_storyline(storyline, storyline_path)
                logger.debug(f"[StoryPipeline] Saved storyline to: {storyline_path}")

            # AGENT 2: Generate storyboard with queries
            logger.info("\n[StoryPipeline] === AGENT 2: Editor ===")
            logger.info(f"[StoryPipeline] Generating CLIP-optimized queries...")

            storyboard = self._run_editor(
                storyline=storyline,
                video_summaries=video_summaries,
                session_id=session_id
            )

            total_queries = sum(len(scene.visual_queries) for scene in storyboard.scenes)
            logger.info(f"[StoryPipeline] ✓ Generated storyboard with {len(storyboard.scenes)} scenes, {total_queries} total queries")
            for i, scene in enumerate(storyboard.scenes, 1):
                logger.debug(f"[StoryPipeline]   {i}. {scene.narrative_id}: {len(scene.visual_queries)} queries")

            # ALWAYS save storyboard when output_dir is provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                storyboard_path = os.path.join(output_dir, "storyboard.json")
                with open(storyboard_path, 'w') as f:
                    json.dump(storyboard.model_dump(), f, indent=2)
                logger.debug(f"[StoryPipeline] Saved storyboard to: {storyboard_path}")

            logger.info("\n[StoryPipeline] === Pipeline Complete ===")
            return storyboard

        except Exception as e:
            logger.error(f"\n[StoryPipeline] ERROR: Pipeline failed: {e}")
            raise e

    @trace(as_type="span", name="pipeline_run_screenwriter")
    def _run_screenwriter(
        self,
        user_prompt: str,
        video_summaries: List[VideoSummary],
        session_id: Optional[str]
    ) -> Storyline:
        """Execute Agent 1 with Langfuse callback handler."""
        if self.runtime_config.enable_langfuse and session_id:
            # Create callback handler with session_id for grouping
            langfuse_handler = CallbackHandler()
        else:
            langfuse_handler = None

        storyline = self.screenwriter.generate_storyline(
            user_prompt=user_prompt,
            video_summaries=video_summaries,
            langfuse_handler=langfuse_handler
        )

        return storyline

    @trace(as_type="span", name="pipeline_run_editor")
    def _run_editor(
        self,
        storyline: Storyline,
        video_summaries: List[VideoSummary],
        session_id: Optional[str]
    ) -> Storyboard:
        """Execute Agent 2 with Langfuse callback handler."""
        if self.runtime_config.enable_langfuse and session_id:
            # Create callback handler with session_id for grouping
            langfuse_handler = CallbackHandler()
        else:
            langfuse_handler = None

        storyboard = self.editor.generate_storyboard(
            storyline=storyline,
            video_summaries=video_summaries,
            langfuse_handler=langfuse_handler
        )

        return storyboard

    def _load_video_summaries(self, path: str) -> List[VideoSummary]:
        """
        Load and parse video summaries from JSON file.

        Args:
            path: Path to JSON file (e.g., qwen_analysis_sequence.json)

        Returns:
            List of VideoSummary objects

        Raises:
            FileNotFoundError: If path doesn't exist
            JSONDecodeError: If file is not valid JSON
            ValidationError: If JSON doesn't match VideoSummary schema
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video summaries file not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        # Parse into Pydantic models for validation
        return [VideoSummary(**item) for item in data]

    def _save_storyline(self, storyline: Storyline, path: str) -> None:
        """Save intermediate storyline to JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with open(path, 'w') as f:
            json.dump(storyline.model_dump(), f, indent=2)


# Convenience function for simple usage
def generate_storyboard(
    user_prompt: str,
    video_summaries_path: str,
    provider: str = "gemini",
    model_name: str = "gemini-2.5-flash",
    **kwargs
) -> Storyboard:
    """
    Convenience function to generate a storyboard without explicit config.

    Args:
        user_prompt: User's creative vision
        video_summaries_path: Path to video summaries JSON
        provider: LLM provider ("gemini" or "local")
        model_name: Model identifier
        **kwargs: Additional arguments passed to generate_storyboard()

    Returns:
        Storyboard object

    Example:
        storyboard = generate_storyboard(
            user_prompt="Create a hiking video in the Alps",
            video_summaries_path="data/qwen_analysis_sequence.json"
        )
    """
    config = PipelineConfig(provider=provider, model_name=model_name)
    pipeline = StoryPipeline(config=config)
    return pipeline.generate_storyboard(user_prompt, video_summaries_path, **kwargs)


if __name__ == "__main__":
    # Example usage
    pipeline = StoryPipeline()

    storyboard = pipeline.generate_storyboard(
        user_prompt=(
            "Create a video that shows the amazing nature we saw in our trip through the Alps. "
            "We were hiking all the time. "
            "Start with some big landscape views of the Alps. Continue with our hike down a river. "
            "Next, we travelled by car to a big lake. "
            "Finally we hiked the shore around the turquoise lake. Show the group of friends hiking."
        ),
        video_summaries_path="data/qwen_analysis_sequence.json",
        save_intermediate=True
    )

    # Save final storyboard
    output_path = "data/generated_storyboard.json"
    with open(output_path, 'w') as f:
        json.dump(storyboard.model_dump(), f, indent=2)

    logger.debug(f"\n[StoryPipeline] Saved final storyboard to: {output_path}")
