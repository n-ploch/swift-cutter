import opentimelineio as otio
import json
import os
import logging
from typing import List, Dict

logger = logging.getLogger('otio_parser')

class OtioTimeline:
    """
    Handles the conversion of generated timeline segments into OpenTimelineIO format.
    """
    def __init__(self, logging_level="INFO"):
        self.timeline_data = []
        self.frame_metadata = []
        self.otio_timeline = None
        self.fps = 1.0

        # Configure logging
        log_level = getattr(logging, logging_level.upper(), logging.INFO)
        logger.setLevel(log_level)

    def load_timeline(self, timeline_path: str):
        """
        Loads the timeline object from a JSON file.
        Expects a list of dictionaries with start_second and end_second.
        """
        with open(timeline_path, 'r') as f:
            self.timeline_data = json.load(f)

    def load_metadata(self, metadata_path: str):
        """
        Loads the video metadata from a JSON file.
        Expects a list of dictionaries with filename and ts (timestamp).
        """
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            self.frame_metadata = data.get('metadata', [])

    def create_otio_timeline(self, timeline_name: str = "Semantic Search Timeline"):
        """
        Creates an OTIO timeline by matching segments to actual video files and timestamps.
        Handles segments that span across multiple video files by splitting them into multiple clips.
        """
        self.otio_timeline = otio.schema.Timeline(name=timeline_name)
        track = otio.schema.Track(name="Video Track", kind=otio.schema.TrackKind.Video)
        self.otio_timeline.tracks.append(track)

        if len(self.frame_metadata) == 0:
            raise ValueError("No video metadata loaded.")

        if len(self.timeline_data) == 0:
            raise ValueError("No timeline data loaded.")

        for segment in self.timeline_data:
            start_idx = segment['start_second']
            end_idx = segment['end_second']

            current_file = None
            clip_start_ts = None
            clip_end_ts = None

            # Iterate through indices to find continuous file blocks
            for i in range(start_idx, end_idx + 1):
                if i >= len(self.frame_metadata):
                    break

                entry = self.frame_metadata[i]
                filename = entry['filename']
                ts = entry['ts']

                if filename != current_file:
                    # If file changed, add the previous clip if it existed
                    if current_file is not None:
                        self._add_clip(track, current_file, clip_start_ts, clip_end_ts, self.fps)
                    
                    current_file = filename
                    clip_start_ts = ts
                
                clip_end_ts = ts

            # Finish the last clip of the segment
            if current_file is not None:
                self._add_clip(track, current_file, clip_start_ts, clip_end_ts, self.fps)

        return self.otio_timeline

    def _should_add_clip(self, filename: str, start_ts: float, end_ts: float, fps: float) -> bool:
        """
        Validates whether a clip should be added based on various criteria.
        Returns True if clip should be added, False otherwise.
        """
        # Calculate duration
        duration_seconds = end_ts - start_ts + (1.0 / fps)

        # Check 1: Skip clips with duration of 1.0
        if duration_seconds == 1.0:
            return False

        # Future checks can be added here
        # Check 2: Skip clips shorter than X seconds
        # if duration_seconds < min_duration:
        #     return False

        return True

    def _add_clip(self, track: otio.schema.Track, filename: str, start_ts: float, end_ts: float, fps: float):
        """
        Helper to create and add a clip to the track.
        """
        # Validate before creating the clip
        if not self._should_add_clip(filename, start_ts, end_ts, fps):
            return  # Skip this clip

        # Source range: start time and duration
        # We assume each index represents 1 second (1 frame at 1 fps)
        # So duration is end_ts - start_ts + 1 second
        duration_seconds = end_ts - start_ts + (1.0 / fps)
        
        start_time = otio.opentime.RationalTime(start_ts, fps)
        duration = otio.opentime.RationalTime(duration_seconds, fps)
        source_range = otio.opentime.TimeRange(start_time, duration)

        # Create External Reference
        media_reference = otio.schema.ExternalReference(
            target_url=filename,
            available_range=None # Could be populated with full video length if known
        )

        clip_name = os.path.basename(filename)
        clip = otio.schema.Clip(
            name=clip_name,
            media_reference=media_reference,
            source_range=source_range
        )

        track.append(clip)

    def save_otio(self, output_path: str):
        """
        Stores the created OTIO timeline as an .otio file.
        """
        if self.otio_timeline is None:
            raise ValueError("Timeline not created. Call create_otio_timeline() first.")

        otio.adapters.write_to_file(self.otio_timeline, output_path)
        logger.info(f"OTIO timeline saved to: {output_path}")

    def generate_otio_timeline(self, timeline_path: str, metadata_path: str, output_path: str = None, timeline_name: str = "Semantic Search Timeline"):
        """
        Complete OTIO generation pipeline - convenience method that combines all steps.

        Args:
            timeline_path: Path to timeline JSON file
            metadata_path: Path to frame metadata JSON file
            output_path: If provided, saves OTIO file to this path
            timeline_name: Name for the OTIO timeline

        Returns:
            OTIO Timeline object
        """
        self.load_timeline(timeline_path)
        self.load_metadata(metadata_path)
        otio_timeline = self.create_otio_timeline(timeline_name=timeline_name)

        if output_path:
            self.save_otio(output_path)

        return otio_timeline

if __name__ == "__main__":
    # Quick test/usage example
    parser = OtioTimeline()
    try:
        parser.load_timeline('data/agentic-2/agentic_timeline_multi_video_output_medium.json')
        parser.load_metadata('data/multi_video_output_medium.json')
        parser.create_otio_timeline()
        parser.save_otio('data/agentic-2/test_3_agentic_otio_multi_video_output_medium.otio')
    except Exception as e:
        logger.error(f"Could not run default export: {e}")
        logger.info("Ensure you have run timeline_generator.py first.")
