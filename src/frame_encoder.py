import subprocess
import numpy as np
import torch
import os
import time
import logging
from PIL import Image
from transformers import AutoProcessor, AutoModel, CLIPProcessor, CLIPModel
import ffmpeg
import json
import threading
import queue

logger = logging.getLogger('frame_encoder')

# ==========================================
# 1. Flexible Model Wrapper
# ==========================================
class VisionModelWrapper:
    """
    Standardizes SigLIP and CLIP so the main loop doesn't care which one you use.
    """
    def __init__(self, model_type="siglip", model_name=None, logging_level="INFO"):
        # Configure logging
        log_level = getattr(logging, logging_level.upper(), logging.INFO)
        logger.setLevel(log_level)

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        self.model_type = model_type.lower()

        logger.debug(f"--- Loading {self.model_type.upper()} Model ---")
        logger.debug(f"    Device: {self.device}")

        if self.model_type == "siglip":
            # Default: Google's SigLIP (Better for search)
            name = model_name or "google/siglip-so400m-patch14-384"
            logger.debug(f"    Model: {name}")
            self.processor = AutoProcessor.from_pretrained(name, use_fast=True)
            self.model = AutoModel.from_pretrained(name).to(self.device).eval()
            self.target_size = 384 # SigLIP 384 expects 384x384

        elif self.model_type == "clip":
            # Default: OpenAI CLIP (Classic)
            name = model_name or "openai/clip-vit-base-patch32"
            logger.debug(f"    Model: {name}")
            self.processor = CLIPProcessor.from_pretrained(name)
            self.model = CLIPModel.from_pretrained(name).to(self.device).eval()
            self.target_size = 224 # Standard CLIP size

    def encode_batch(self, frames_batch_np):
        """
        Input: Numpy array of shape (Batch, H, W, 3)
        Output: Numpy array of normalized embeddings (Batch, Dim)
        """
        # Convert Numpy -> List of PIL Images
        # (Transformers lib expects PIL or direct tensors)
        images = [Image.fromarray(img) for img in frames_batch_np]

        # Preprocess
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        start_inference = time.time()
        with torch.no_grad():
            if self.model_type == "siglip":
                output = self.model.get_image_features(**inputs)
            else: # CLIP
                output = self.model.get_image_features(**inputs)

            # Extract tensor from model output (handle both old and new API)
            if isinstance(output, torch.Tensor):
                features = output
            elif hasattr(output, 'pooler_output') and output.pooler_output is not None:
                features = output.pooler_output
            elif hasattr(output, 'last_hidden_state'):
                features = output.last_hidden_state[:, 0]  # Use [CLS] token
            else:
                # Fallback for any other structure
                features = output[0] if isinstance(output, (tuple, list)) else output

            # CRITICAL: Normalize vectors so dot product works later
            features = features / features.norm(p=2, dim=-1, keepdim=True)

        inference_time = time.time() - start_inference
        return features.cpu().numpy(), inference_time

    def encode_text(self, texts):
        """
        Encode text queries to embeddings for semantic search.

        Args:
            texts: List of text strings to encode

        Returns:
            Numpy array of normalized text embeddings (N, Dim)
        """
        # Preprocess text inputs
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)

        start_inference = time.time()
        with torch.no_grad():
            if self.model_type == "siglip":
                output = self.model.get_text_features(**inputs)
            else:  # CLIP
                output = self.model.get_text_features(**inputs)

            # Extract tensor from model output (handle both old and new API)
            if isinstance(output, torch.Tensor):
                features = output
            elif hasattr(output, 'pooler_output') and output.pooler_output is not None:
                features = output.pooler_output
            elif hasattr(output, 'last_hidden_state'):
                features = output.last_hidden_state[:, 0]  # Use [CLS] token
            else:
                # Fallback for any other structure
                features = output[0] if isinstance(output, (tuple, list)) else output

            # CRITICAL: Normalize vectors for cosine similarity (dot product)
            features = features / features.norm(p=2, dim=-1, keepdim=True)

        inference_time = time.time() - start_inference
        return features.cpu().numpy()

# ==========================================
# 2. The FFmpeg Streamer (No Temp Files)
# ==========================================
def stream_frames(video_path, fps=1, width=224, height=224):
    """
    Yields frames using the ffmpeg-python library wrapper.
    Functionally identical to the subprocess version, but cleaner code.
    Tries to use hardware acceleration (VideoToolbox) on Mac Silicon.
    """
    frame_bytes = width * height * 3

    try:
        # 1. Try with hardware acceleration (VideoToolbox)
        process = (
            ffmpeg
            .input(video_path, hwaccel='videotoolbox')
            .filter('fps', fps=fps)
            .filter('scale', width, height)
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True, quiet=True)
        )
    except ffmpeg.Error as e:
        logger.error("FFmpeg Error:", e.stderr.decode('utf8'))
        return

    # The reading logic remains the EXACT same because run_async()
    # returns a standard subprocess.Popen object.
    frame_idx = 1
    while True:
        start_read = time.time()
        raw_data = process.stdout.read(frame_bytes)
        read_time = time.time() - start_read

        if len(raw_data) != frame_bytes:
            break

        timestamp = frame_idx / fps
        yield np.frombuffer(raw_data, dtype='uint8').reshape((height, width, 3)), timestamp, read_time
        frame_idx += 1

    process.wait()

# ==========================================
# 3. The Frame Indexer Class
# ==========================================
class FrameIndexer:
    """
    Processes videos and extracts frame embeddings.
    """
    def __init__(self, model_type="siglip", model_name=None, logging_level="INFO"):
        """Initialize with vision model."""
        self.encoder = VisionModelWrapper(model_type=model_type, model_name=model_name, logging_level=logging_level)

    def process_video(self, video_paths, output_filename=None, record_frames=False):
        """
        Processes one or more videos, extracts frame embeddings, and optionally saves them to a .npy file.
        Also saves a companion .json file with frame metadata (filename and timestamp).

        Args:
            video_paths: List or single video path
            output_filename: If provided, saves .npy and .json files
            record_frames: Whether to save frame images

        Returns:
            dict with keys: 'embeddings', 'metadata', 'paths'
        """
        if isinstance(video_paths, str):
            video_paths = [video_paths]

        # Validate all paths exist
        valid_paths = []
        for vp in video_paths:
            if not os.path.exists(vp):
                logger.warning(f"Warning: File {vp} not found. Skipping.")
            else:
                valid_paths.append(vp)

        if not valid_paths:
            logger.error("Error: No valid video files found.")
            return

        # 1. Setup Processing
        start_time = time.time()
        batch_size = 4
        current_batch = []
        all_embeddings = []
        metadata = []
        total_frame_count = 0

        # 2. Setup frame storage folder
        if record_frames:
            timestamp_folder = time.strftime("%Y-%m-%d_%H_%M_test")
            os.makedirs(timestamp_folder, exist_ok=True)
            logger.debug(f"--- Storing frames in: {timestamp_folder} ---")

        # 3. Producer (Background Thread)
        frame_queue = queue.Queue(maxsize=batch_size * 2)
        stop_event = threading.Event()

        def producer():
            for video_path in valid_paths:
                logger.debug(f"\n--- Extracting: {video_path} ---")
                abs_path = os.path.abspath(video_path)
                stream = stream_frames(video_path, fps=1, width=self.encoder.target_size, height=self.encoder.target_size)
                for frame, timestamp, resize_time in stream:
                    if stop_event.is_set():
                        break
                    frame_queue.put((frame, timestamp, abs_path, resize_time))
            frame_queue.put(None) # Sentinel

        threading.Thread(target=producer, daemon=True).start()

        # 4. Consumer (Main Thread - Neural Engine)
        total_inference_time = 0
        total_resize_time = 0
        while True:
            item = frame_queue.get()
            if item is None:
                break

            frame, timestamp, abs_path, resize_time = item
            total_resize_time += resize_time
            current_batch.append(frame)
            metadata.append({
                "filename": abs_path,
                "ts": timestamp
            })

            # Save frame to disk
            if record_frames:
                video_basename = os.path.splitext(os.path.basename(abs_path))[0]
                frame_filename = f"{video_basename}_{timestamp:.2f}.jpg"
                frame_path = os.path.join(timestamp_folder, frame_filename)
                Image.fromarray(frame).save(frame_path)

            # When batch is full, encode it
            if len(current_batch) >= batch_size:
                emb, inf_time = self.encoder.encode_batch(np.array(current_batch))
                total_inference_time += inf_time
                all_embeddings.append(emb)
                total_frame_count += len(current_batch)
                current_batch = []
                logger.debug(f"    Indexed {total_frame_count} frames... (Avg Inference: {total_inference_time/(total_frame_count/batch_size if total_frame_count > 0 else 1):.4f}s)")

        # Process leftovers
        if current_batch:
            emb, inf_time = self.encoder.encode_batch(np.array(current_batch))
            total_inference_time += inf_time
            all_embeddings.append(emb)
            total_frame_count += len(current_batch)

        logger.info(f"\n    Finished! Total Frames: {total_frame_count}")

        # 5. Process embeddings
        if not all_embeddings:
            logger.error("Error: No frames were extracted. Check video files.")
            return {
                'embeddings': None,
                'metadata': metadata,
                'paths': None
            }

        # Stack list of arrays into one big (N, Dim) matrix
        final_matrix = np.vstack(all_embeddings)
        elapsed = time.time() - start_time

        # 6. Save to disk if output_filename provided
        paths = None
        if output_filename:
            np.save(output_filename, final_matrix)

            # Save Metadata JSON
            json_filename = os.path.splitext(output_filename)[0] + ".json"
            with open(json_filename, 'w') as f:
                json.dump({"metadata": metadata}, f, indent=2)

            paths = {'npy': output_filename, 'json': json_filename}

            logger.debug(f"\n--- Saved Index to '{output_filename}' ---")
            logger.debug(f"--- Saved Metadata to '{json_filename}' ---")
            logger.debug(f"    Matrix Shape: {final_matrix.shape}")
            logger.debug(f"    Total Time: {elapsed:.2f}s ({total_frame_count/elapsed:.1f} fps)")
            logger.debug(f"    Avg Resize/Dec Time: {total_resize_time/total_frame_count:.4f}s per frame")
            logger.debug(f"    Avg Model Inference: {total_inference_time/(total_frame_count/batch_size) if total_frame_count > 0 else 0:.4f}s per batch")

        # 7. ALWAYS return results for pipeline chaining
        return {
            'embeddings': final_matrix,
            'metadata': metadata,
            'paths': paths
        }

# ==========================================
# Backward Compatibility Function
# ==========================================
def process_video(video_paths, output_filename, model_type="siglip", record_frames=False):
    """
    Legacy function for backward compatibility.
    Maintains the original API while using the new FrameIndexer class internally.
    """
    indexer = FrameIndexer(model_type=model_type)
    indexer.process_video(video_paths, output_filename, record_frames)

# ==========================================
# Usage
# ==========================================
if __name__ == "__main__":
    # Example 1: Use a list of video paths
    video_files = [
        "./data/DJI_20250803143631_0217_D_short.MP4",
        "./data/DJI_20250803144140_0220_D.MP4",
        "./data/DJI_20250801134716_0192_D.MP4",
        "./data/DJI_20250802150946_0204_D.MP4",
        "./data/DJI_20250802144037_0199_D.MP4",
        "./data/DJI_20250803143754_0218_D.MP4",
        "./data/DJI_20250514105245_0135_D.MP4"
    ]

    output_npy = "./data/multi_video_output_medium.npy"

    # Use the better SigLIP model (Default)
    process_video(video_files, output_npy, model_type="siglip", record_frames=True)

    logger.info(f"\nResults saved to {output_npy} and associated .json file.")
