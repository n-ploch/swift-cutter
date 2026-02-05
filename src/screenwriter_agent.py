"""
Agent 1: Screenwriter - Creates narrative structure from user prompt and available footage.

This agent acts as a screenwriter and cinematographer, crafting a coherent narrative
based on the user's creative vision and the available video footage.
"""

import os
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langfuse.langchain import CallbackHandler

from config import ScreenwriterConfig, RuntimeConfig
from models import VideoSummary, Storyline
from tracing import trace


class ScreenwriterAgent:
    """
    Agent 1: Creates narrative structure based on user prompt and available footage.

    Responsibilities:
    - Analyze user's creative vision
    - Review available video footage summaries
    - Craft chronological narrative with cinematographic detail
    - Estimate scene durations based on narrative pacing
    - Output structured Storyline for Agent 2 to convert into queries
    """

    def __init__(self, config: ScreenwriterConfig, runtime: RuntimeConfig):
        """
        Initialize the Screenwriter agent.

        Args:
            config: Screenwriter-specific configuration (model, temperature)
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
        self.parser = PydanticOutputParser(pydantic_object=Storyline)

        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert Screenwriter and Cinematographer working with existing footage.
Your role is to craft a compelling narrative structure based on:
1. The user's creative vision
2. The available footage (video summaries provided)

### YOUR TASK:
Create a storyline that:
- Tells a coherent, engaging story with clear beginning, middle, and end
- Leverages the footage you have available
- Breaks the narrative into distinct, chronological scenes
- Uses cinematic language (framing, composition, pacing)
- Estimates realistic scene durations

### CRITICAL RULES:

1. **Work with Available Footage**: Only create scenes that can be matched to the video summaries provided.
   Review the footage carefully and design scenes around what exists.

2. **Cinematographic Detail**: Describe VISUAL elements, not abstract emotions.
   - Bad: "The scene feels peaceful and joyful"
   - Good: "Wide shot of calm turquoise lake reflecting snow-capped mountains, soft morning light, hikers in colorful gear walking along the shore"

3. **Chronological Flow**: Order scenes to tell a story with narrative progression.
   Consider user input regarding the Consider setup → journey → climax → resolution.

4. **Specific Settings**: Extract the main location, time of day, and subjects from the user's prompt.
   Maintain consistency across scenes (e.g., if it's "Alps in summer", all scenes should reflect this).

5. **Scene Duration**: Estimate realistic duration (5-20 seconds per scene) based on:
   - Narrative importance (climax scenes may be longer)
   - Visual complexity (establishing shots may be shorter)
   - Available footage (don't request 30 seconds if footage is limited)

6. **Framing Language**: Use professional cinematography terms:
   - Wide shot / Establishing shot
   - Medium shot
   - Close-up
   - Aerial / Drone shot
   - Point-of-view shot
   - Tracking shot

### AVAILABLE FOOTAGE:
{video_summaries}

{format_instructions}

### USER'S CREATIVE VISION:
"""),
            ("user", "{user_prompt}")
        ])

    @trace(as_type="agent", name="screenwriter_generate_storyline")
    def generate_storyline(
        self,
        user_prompt: str,
        video_summaries: List[VideoSummary],
        langfuse_handler: Optional[CallbackHandler] = None,
        session_id: Optional[str] = None
    ) -> Storyline:
        """
        Generate a narrative storyline from user prompt and available footage.

        Args:
            user_prompt: User's creative vision for the video
            video_summaries: List of available video footage with descriptions
            langfuse_handler: Optional Langfuse callback handler for tracing

        Returns:
            Storyline object with ordered StoryScene objects

        Raises:
            ValidationError: If LLM output doesn't match Storyline schema
            Exception: If LLM generation fails
        """
        # Format video summaries for the prompt
        formatted_summaries = self._format_summaries(video_summaries)

        # Create the chain
        chain = self.prompt | self.llm | self.parser

        # Prepare config with callbacks if provided
        config = {}
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]

        try:
            # Execute the chain
            result: Storyline = chain.invoke(
                {
                    "user_prompt": user_prompt,
                    "video_summaries": formatted_summaries,
                    "format_instructions": self.parser.get_format_instructions()
                },
                config=config
            )

            return result

        except Exception as e:
            print(f"[ScreenwriterAgent] Error generating storyline: {e}")
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
            # Extract just the filename for readability
            filename = os.path.basename(summary.video)
            formatted.append(
                f"{i}. Video: {filename}\n"
                f"   Frames analyzed: {summary.num_frames}\n"
                f"   Content: {summary.description}"
            )

        return "\n\n".join(formatted)
