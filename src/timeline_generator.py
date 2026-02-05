from frame_encoder import VisionModelWrapper
import json
import logging
from typing import List, Dict
import numpy as np
from dataclasses import dataclass, asdict
from scipy.signal import peak_widths, find_peaks

logger = logging.getLogger('timeline_generator')


@dataclass
class Timeline:
    scene_id: int
    peak_second: int
    start_second: int
    end_second: int
    score: float
    duration_seconds: int

class TimelineGenerator:
    def __init__(self, model_type: str, debugging: bool = False, logging_level: str = "INFO"):
        self.encoder = VisionModelWrapper(model_type=model_type, logging_level=logging_level)
        self.metadata = {"analysis": {}}
        self.debugging = debugging
        self.timeline = []

        # Configure logging
        log_level = getattr(logging, logging_level.upper(), logging.INFO)
        logger.setLevel(log_level)

    def generate_timeline(self, storyboard_path: str, video_embeddings_path: str, moments_per_scene: int = 5, output_dir: str = None) -> List[Timeline]:
        """
        Generate timeline by matching storyboard to video embeddings.

        Args:
            storyboard_path: Path to storyboard JSON file
            video_embeddings_path: Path to frame embeddings .npy file
            moments_per_scene: Number of moments to find per scene
            output_dir: If provided, saves timeline_json.json and timeline_metadata.json

        Returns:
            List of Timeline objects
        """
        final_timeline = []
        storyboard = self.load_storyboard(storyboard_path)
        video_embeddings = self.load_video_embeddings(video_embeddings_path)

        self.metadata["storyboard_path"] = storyboard_path
        self.metadata["video_embeddings_path"] = video_embeddings_path
        
        for scene_idx, scene in enumerate(storyboard['scenes']):
            all_queries = [query['query_text'] for query in scene['visual_queries']]
            query_embeddings = self.encoder.encode_text(all_queries)
            
            # scores shape: (num_queries, num_seconds)
            scores = query_embeddings @ video_embeddings.T
            
            # Aggregate scores across all visual queries for this scene
            scene_scores = np.max(scores, axis=0)
            
            # Use find_top_k_scenes to get more candidates (e.g., 50) then merge
            candidates, smoothed_scores = self.find_top_k_scenes(scores, sigma=1.5, k=moments_per_scene*len(scene["visual_queries"]))
            top_moments = self.temporal_nms_merge(candidates, k=moments_per_scene)

            # Mask out selected moments in video embeddings to prevent reuse
            for moment in top_moments:
                start_idx = int(moment['start_second'])
                end_idx = int(moment['end_second'])
                # Set the rows for selected time range to -inf
                video_embeddings[start_idx:end_idx+1] = -np.inf

            if self.debugging:
                self.metadata["analysis"][scene_idx] = {
                    "narrative_id": scene['narrative_id'],
                    "visual_queries": [{**query, "scores": scores[i, :].tolist(), 'smoothed_scores': smoothed_scores[i, :].tolist() if smoothed_scores.shape[0] > 1 else "NA"} for i, query in enumerate(scene['visual_queries'])],
                    "scores": [float(s) for s in scene_scores],
                    "smoothed_scores": [float(s) for s in smoothed_scores] if smoothed_scores.ndim == 1 else [float(s) for s in scene_scores],
                    "candidates": candidates,
                    "top_moments": top_moments
                }
            
            logger.debug(f"\n--- Scene {scene_idx}: {scene['narrative_id']} ---")
            for moment in top_moments:
                final_timeline.append(Timeline(
                    scene_id=scene_idx,
                    peak_second=int(moment['peak_second']),
                    start_second=int(moment['start_second']),
                    end_second=int(moment['end_second']),
                    score=float(moment['score']),
                    duration_seconds=int(moment['duration_seconds']),
                ))
        
        self.timeline = final_timeline

        # Save to disk if output_dir provided
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            self.save_timeline(os.path.join(output_dir, "timeline_json.json"))
            if self.debugging:
                self.store_metadata(os.path.join(output_dir, "timeline_metadata.json"))

        return final_timeline

    def gaussian_smooth(self, scores: np.ndarray, sigma: float) -> np.ndarray:
        """Simple 1D Gaussian smoothing using numpy."""
        if len(scores) <= 1:
            return scores
            
        size = int(4 * sigma + 0.5)
        x = np.arange(-size, size + 1)
        kernel = np.exp(-0.5 * (x / sigma)**2)
        kernel /= kernel.sum()
        
        # mode='same' returns max(len(scores), len(kernel))
        # We must ensure it always returns len(scores) for short videos
        smoothed = np.convolve(scores, kernel, mode='same')
        
        if len(smoothed) > len(scores):
            # Extract the centered part of length len(scores)
            start = (len(smoothed) - len(scores)) // 2
            return smoothed[start : start + len(scores)]
            
        return smoothed

    def find_top_k_scenes(self, scores: np.ndarray, sigma: float, k: int, p2p_distance: int = 4, rel_height: float = 0.4) -> List[Dict]:
        """
        Applies Non-Maximum Suppression with adaptive window size.
        Supports both 1D arrays and 2D matrices (num_queries, num_timestamps).
        """
        # Ensure scores is 2D
        original_shape = scores.shape
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)

        num_queries, num_timestamps = scores.shape
        smoothed_scores_matrix = np.zeros_like(scores)
        candidates = []

        # Process each query row
        for query_idx in range(num_queries):
            query_scores = scores[query_idx, :]
            smoothed = self.gaussian_smooth(query_scores, sigma)
            smoothed_scores_matrix[query_idx, :] = smoothed

            peaks, properties = find_peaks(smoothed, distance=p2p_distance)

            if len(peaks) > 0:
                results = peak_widths(smoothed, peaks, rel_height=rel_height)
                widths, width_heights, left_ips, right_ips = results
                query_candidates = []
                for i in range(len(peaks)):
                    query_candidates.append({
                        'peak_second': int(peaks[i]),
                        'start_second': int(left_ips[i]),
                        'end_second': int(right_ips[i]),
                        'score': float(smoothed[peaks[i]])
                    })
                query_candidates.sort(key=lambda x: x['score'], reverse=True)
                num_candidates = int(k/num_queries) + 1
                for c in query_candidates[:num_candidates]:
                    candidates.append(c)

        # Sort and return Top K
        candidates.sort(key=lambda x: x['score'], reverse=True)

        # Return smoothed scores in original shape
        if len(original_shape) == 1:
            return candidates[:k], smoothed_scores_matrix.flatten()
        else:
            return candidates[:k], smoothed_scores_matrix

    def temporal_nms_merge(self, candidates: List[Dict], k: int) -> List[Dict]:
        """
        Modified Temporal NMS that merges overlapping candidate scenes.
        1. Pick the best candidate.
        2. Find all others that overlap with it.
        3. Merge them into a single segment covering the full reach.
        4. Repeat until K scenes are reached.
        """
        if not candidates:
            return []
            
        remaining = list(candidates)
        merged_scenes = []
        
        while remaining and len(merged_scenes) < k:
            # 1. Pick the best one (remaining is assumed sorted by score)
            best = remaining.pop(0)
            
            curr_start = best['start_second']
            curr_end = best['end_second']
            curr_score = best['score']
            curr_peak = best['peak_second']
            
            # 2. Find all that (partly) overlap with it
            # Continue checking for overlaps with the newly expanded window
            while True:
                to_merge_indices = []
                for i, cand in enumerate(remaining):
                    # Check for overlap (IoU > 0)
                    inter_s = max(curr_start, cand['start_second'])
                    inter_e = min(curr_end, cand['end_second'])

                    if inter_e > inter_s: # Overlap exists
                        to_merge_indices.append(i)

                if not to_merge_indices:
                    # No more overlaps found with the current window
                    break

                # 3. Merge them: set the reach of the combined scenes
                for i in sorted(to_merge_indices, reverse=True):
                    cand = remaining.pop(i)
                    curr_start = min(curr_start, cand['start_second'])
                    curr_end = max(curr_end, cand['end_second'])
                    # Keep max score
                    curr_score = max(curr_score, cand['score'])

                # Loop continues to check if the newly expanded window overlaps with any remaining candidates
            
            merged_scenes.append({
                'peak_second': curr_peak,
                'start_second': curr_start,
                'end_second': curr_end,
                'duration_seconds': curr_end - curr_start,
                'score': curr_score
            })
            
        return merged_scenes


    def load_storyboard(self, storyboard_path: str):
        with open(storyboard_path, 'r') as f:
            storyboard = json.load(f)
        return storyboard

    def load_video_embeddings(self, video_path: str):
        video_embeddings = np.load(video_path)
        return video_embeddings

    def store_metadata(self, metadata_path: str, indent: int = 2):
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=indent)
    
    def save_timeline(self, timeline_path: str):
        with open(timeline_path, 'w') as f:
            json.dump([asdict(t) for t in self.timeline], f, indent=2)

if __name__ == '__main__':
    generator = TimelineGenerator(model_type='siglip', debugging=True)
    generator.generate_timeline(storyboard_path='data/test_storyboard.json', video_embeddings_path='data/multi_video_output_medium.npy', moments_per_scene=2)
    generator.store_metadata('data/agentic-2/agentic_metadata_multi_video_output_medium.json')
    generator.save_timeline('data/agentic-2/agentic_timeline_multi_video_output_medium.json')
