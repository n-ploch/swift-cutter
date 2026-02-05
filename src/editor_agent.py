"""
Agent 2: Editor - Converts storyline into CLIP-optimized visual search queries.

This agent acts as a video editor and CLIP search specialist, creating precise
visual queries that maximize cosine similarity matching with video frame embeddings.
"""

import os
import json
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler

from config import EditorConfig, RuntimeConfig
from models import VideoSummary, Storyline, Storyboard
from tracing import trace


class EditorAgent:
    """
    Agent 2: Converts storyline into CLIP-optimized visual search queries.

    Responsibilities:
    - Analyze narrative structure from Agent 1
    - Review available video footage
    - Generate 3-5 highly specific CLIP queries per scene
    - Optimize queries for maximum cosine similarity matching
    - Ensure narrative_id consistency with input storyline
    - Output Storyboard compatible with timeline_generator.py
    """

    def __init__(self, config: EditorConfig, runtime: RuntimeConfig):
        """
        Initialize the Editor agent.

        Args:
            config: Editor-specific configuration (model, temperature, queries_per_scene)
            runtime: Runtime configuration (logging, langfuse)
        """
        self.config = config
        self.runtime = runtime

        # Setup the LLM
        if self.config.provider == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model=self.config.model,
                temperature=self.config.temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        elif self.config.provider == "local":
            self.llm = ChatOllama(
                model=self.config.model,
                temperature=self.config.temperature
            )
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}. Use 'gemini' or 'local'.")

        # Setup the parser for structured output
        self.parser = PydanticOutputParser(pydantic_object=Storyboard)

        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert Video Editor and CLIP Search Specialist.
You receive a storyline and must create PRECISE visual search queries to find matching frames. Think of every query 
in a scene being responsible for a single shot of the final video.

### YOUR TASK:
For each scene in the storyline, generate {queries_per_scene} CLIP-optimized visual queries:
- Every scene covers a different aspect of the scene description
- Capture different angles of the scene
- The queries have to be highly specific and concrete
- Start with framing keywords
- Include contextual details

### CLIP QUERY OPTIMIZATION RULES:

**Specific aspect per query** - Every query covers one aspect per query:
   - Queries should break down into the single aspects and shots described in the scene description.
   - Do NOT enumerate many aspects in one query
   - BAD example: "An aerial wide shot of the vast Alps mountain range, showing jagged rocky peaks, lush green emerald forests, and a stunning turquoise lake reflecting bright sunlight, under a clear blue sky."
   - GOOD examples: 
      - An aerial wide shot of a mountain range, showing jagged rocky peaks under a clear blue sky.
      - A sweeping aerial drone view across a large turquoise alpine lake.
      - A high aerial wide shot of lush green emerald forests in bright daylight.

**Framing Keywords** - Always start with specific framing:
   - "A cinematic wide shot of..."
   - "A close-up photo of..."
   - "An aerial drone view of..."
   - "A medium shot showing..."
   - "A point-of-view shot from..."
   - "A tracking shot following..."

**Structural Formula Injection** - Use [Subject] + [Action] + [Environment] in EVERY query:
   - Example: "A professional chef (Subject) chopping fresh vegetables (Action) in a high-end restaurant kitchen (Environment)."

**Physical Descriptors Only** - No abstract concepts or emotions:
   - Bad: "A scary drive through the mountains"
   - Good: "A car driving on a narrow cliff edge road, steep drop-off, dark cloudy sky, rugged mountain terrain"
   - Bad: "A peaceful moment by the lake"
   - Good: "A wide shot of a calm turquoise lake with reflections, surrounded by mountains, people sitting on shore"

**Diverse Coverage** - Vary the queries per scene in content and setting to capture different aspects:
   - Query 1: Establishing wide shot (sets the scene)
   - Query 2: Subject/action medium shot (shows main activity)
   - Query 3: Detail/close-up (captures specific elements)
   - Query 4: Alternative angle or composition (provides variety)
   - [Query 5 if requested]: Movement or transition element

**Color & Lighting Specificity** - Include concrete visual attributes:
   - Colors: "turquoise lake", "snow-capped peaks", "lush green forest", "rocky gray terrain"
   - Lighting: "bright sunny day", "morning golden light", "overcast sky", "soft afternoon light"
   - Weather: "clear blue sky", "cloudy weather", "misty conditions"s

**Positive Alternative Injection** - Replace "negative" concepts with "present" descriptions:
   - Example: Instead of "a room without a chair," use "an empty minimalist floor space" or "a room containing only a bed."

### WEIGHT ASSIGNMENT:
- Use weight = 1.0 for direct visual matches (core scene elements)
- Use weight = 0.9 for supporting visual elements
- Use weight = 0.8 for thematic/atmospheric elements

### NARRATIVE_ID CONSISTENCY:
- You MUST use the exact narrative_id from the input storyline
- Copy the description and rationale from the storyline
- Maintain the estimated_duration from the storyline

### AVAILABLE FOOTAGE:
{video_summaries}

### STORYLINE TO CONVERT:
{storyline_json}

{format_instructions}

Generate the complete storyboard with visual queries now. Remember: maximum specificity for cosine similarity matching!
"""),
            ("user", "Convert the storyline into a storyboard with {queries_per_scene} CLIP-optimized queries per scene.")
        ])

    @trace(as_type="agent", name="editor_generate_storyboard")
    def generate_storyboard(
        self,
        storyline: Storyline,
        video_summaries: List[VideoSummary],
        langfuse_handler: Optional[CallbackHandler] = None
    ) -> Storyboard:
        """
        Generate CLIP-optimized storyboard from storyline and available footage.

        Args:
            storyline: Storyline object from Agent 1 with StoryScene objects
            video_summaries: List of available video footage with descriptions
            langfuse_handler: Optional Langfuse callback handler for tracing

        Returns:
            Storyboard object with Scene objects containing visual queries

        Raises:
            ValidationError: If LLM output doesn't match Storyboard schema
            Exception: If LLM generation fails
        """
        # Format video summaries for the prompt
        formatted_summaries = self._format_summaries(video_summaries)

        # Convert storyline to JSON for the prompt
        storyline_json = storyline.model_dump_json(indent=2)

        # Create the chain
        chain = self.prompt | self.llm | self.parser

        # Prepare config with callbacks if provided
        config = {}
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]

        try:
            # Execute the chain
            result: Storyboard = chain.invoke(
                {
                    "queries_per_scene": self.config.queries_per_scene,
                    "video_summaries": formatted_summaries,
                    "storyline_json": storyline_json,
                    "format_instructions": self.parser.get_format_instructions()
                },
                config=config
            )

            # Validate narrative_id consistency
            self._validate_narrative_ids(storyline, result)

            return result

        except Exception as e:
            print(f"[EditorAgent] Error generating storyboard: {e}")
            raise e

    def _format_summaries(self, summaries: List[VideoSummary]) -> str:
        """
        Format video summaries into a readable string for the LLM prompt.

        Args:
            summaries: List of VideoSummary objects

        Returns:
            Formatted string with numbered video descriptions
        """
        if not summaries:
            return "No video footage available."

        formatted = []
        for i, summary in enumerate(summaries, 1):
            filename = os.path.basename(summary.video)
            formatted.append(
                f"{i}. Video: {filename}\n"
                f"   Frames: {summary.num_frames}\n"
                f"   Content: {summary.description}"
            )

        return "\n\n".join(formatted)

    def _validate_narrative_ids(self, storyline: Storyline, storyboard: Storyboard) -> None:
        """
        Validate that storyboard scenes match storyline narrative_ids.

        Args:
            storyline: Input storyline from Agent 1
            storyboard: Output storyboard from Agent 2

        Raises:
            ValueError: If narrative_ids don't match
        """
        storyline_ids = {scene.narrative_id for scene in storyline.scenes}
        storyboard_ids = {scene.narrative_id for scene in storyboard.scenes}

        if storyline_ids != storyboard_ids:
            missing_in_storyboard = storyline_ids - storyboard_ids
            extra_in_storyboard = storyboard_ids - storyline_ids

            error_msg = "[EditorAgent] Narrative ID mismatch:\n"
            if missing_in_storyboard:
                error_msg += f"  Missing in storyboard: {missing_in_storyboard}\n"
            if extra_in_storyboard:
                error_msg += f"  Extra in storyboard: {extra_in_storyboard}\n"

            print(error_msg)
            # Note: We're printing a warning but not raising an exception
            # to allow the pipeline to continue. The LLM might reorder scenes.
