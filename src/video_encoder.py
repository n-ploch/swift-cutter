import numpy as np
import os
import time
import json
import logging
import threading
import queue
import ffmpeg
from PIL import Image
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

# Suppress Hugging Face Tokenizers fork warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger('video_encoder')

# ==========================================
# 1. Qwen Vision Model Wrapper (MLX Optimized)
# ==========================================
class QwenVisionWrapper:
    """
    Handles Qwen2-VL 7B inference using MLX for maximum speed on Mac M1 Pro.
    """
    def __init__(self, model_path="mlx-community/Qwen2-VL-7B-Instruct-4bit", logging_level="INFO"):
        # Configure logging
        log_level = getattr(logging, logging_level.upper(), logging.INFO)
        logger.setLevel(log_level)

        logger.debug(f"--- Loading Qwen2-VL (MLX) ---")
        # Load model and processor (4-bit is recommended for 16GB-32GB RAM)
        self.model, self.processor = load(model_path)
        self.config = self.model.config

    def describe_batch(self, frames_batch_np, system_prompt="You are a video description assistant. Describe what is happening in this frame in 50 words or less.", user_prompt="Describe the image."):
        """
        Input: List of Numpy frames (H, W, 3)
        Output: List of text descriptions
        """
        results = []
        start_inference = time.time()

        for frame_np in frames_batch_np:
            pil_img = Image.fromarray(frame_np)

            # Prepare the prompt for Qwen's specific template
            user_messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image"}
                    ],
                }
            ]

            # Generate response
            user_prompt_data = apply_chat_template(self.processor, self.config, user_messages, num_images=1)
            output = generate(
                model=self.model,
                processor=self.processor,
                prompt=user_prompt_data,
                image=[pil_img],
                max_tokens=70,
                verbose=False,
            )
            results.append(output)
        inference_time = time.time() - start_inference
        return results, inference_time

    def describe_image_sequence(self, frames_batch_list, fps, system_prompt="You are a video description assistant. Describe what is happening in the video in 100 words or less."):
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"}] * len(frames_batch_list) +
                    [{"type": "text",
                    "text": "Watch the image sequence and describe the video in a narrative style.",
                    }]
            }
        ]


        start_inference = time.time()
        user_prompt_data = apply_chat_template(self.processor, self.config, messages, num_images=len(frames_batch_list))
        output = generate(
            model=self.model,
            processor=self.processor,
            prompt=user_prompt_data,
            image=frames_batch_list,
            max_tokens=150,
            verbose=False,
        )
        inference_time = time.time() - start_inference
        return output, inference_time



# ==========================================
# 2. FFmpeg Streamer (Same as your original)
# ==========================================
def get_video_as_frame_sequence(video_path, fps=0.5, width=448, num_frames=None):
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
    orig_w = int(video_stream['width'])
    orig_h = int(video_stream['height'])

    # 2. Calculate proportional height (must be even for most codecs)
    aspect_ratio = orig_h / orig_w
    target_height = int(width * aspect_ratio)
    if target_height % 2 != 0: target_height += 1 # Ensure even number

    # 3. Calculate exactly how many bytes to read per frame
    frame_bytes = width * target_height * 3

    if num_frames is None:
        num_frames = int(duration * fps)
        if num_frames < 3:
            num_frames = 3

    # 2. Calculate timestamps (e.g., at 10%, 30%, 50%, 70%, 90% of the video)
    timestamps = np.linspace(0, duration, num_frames + 2)[1:-1]

    # 3. Extract frames
    frames = []
    for ts in timestamps:
        process = (
            ffmpeg
            .input(video_path, ss=ts, hwaccel='videotoolbox')
            .filter('scale', width, target_height)
            .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
            .run_async(pipe_stdout=True, quiet=True)
        )
        raw_data = process.stdout.read(frame_bytes)
        frame = np.frombuffer(raw_data, dtype='uint8').reshape((target_height, width, 3)).copy()
        frames.append(frame)
    return frames

def stream_frames(video_path, fps=0.5, width=448, height=448):
    """
    Extracts frames using Hardware Acceleration (VideoToolbox) on Mac.
    Note: Qwen handles variable resolution, but 448x448 is a good balance for speed.
    """

    probe = ffmpeg.probe(video_path)
    video_stream = next((s for s in probe['streams'] if s['codec_type'] == 'video'), None)
    orig_w = int(video_stream['width'])
    orig_h = int(video_stream['height'])

    # 2. Calculate proportional height (must be even for most codecs)
    aspect_ratio = orig_h / orig_w
    target_height = int(width * aspect_ratio)
    if target_height % 2 != 0: target_height += 1 # Ensure even number

    # 3. Calculate exactly how many bytes to read per frame
    frame_bytes = width * target_height * 3

    process = (
        ffmpeg
        .input(video_path, hwaccel='videotoolbox')
        .filter('fps', fps=fps)
        .filter('scale', width, target_height)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .run_async(pipe_stdout=True, quiet=True)
    )

    frame_idx = 1
    while True:
        start_read = time.time()
        raw_data = process.stdout.read(frame_bytes)
        read_time = time.time() - start_read

        if len(raw_data) != frame_bytes: break

        timestamp = frame_idx / fps
        yield np.frombuffer(raw_data, dtype='uint8').reshape((target_height, width, 3)).copy(), timestamp, read_time
        frame_idx += 1
    process.wait()

# ==========================================
# 3. The Video Processor Class
# ==========================================
class VideoProcessor:
    """
    Process videos with Qwen vision-language model for description generation.
    """
    def __init__(self, model_path="mlx-community/Qwen2-VL-7B-Instruct-4bit", logging_level="INFO"):
        """Initialize video processor with Qwen model."""
        self.vlm = QwenVisionWrapper(model_path=model_path, logging_level=logging_level)

    def process_video_by_frame(self, video_paths, output_json, fps=0.5, store_frame=False, store_image_path="data/frames", system_prompt="You are a video analysis assistant. Your task is to watch the video frame by frame and describe what is happening in each frame in 50 words or less. Avoid introductory phrases like \"In this frame\". Keep it short and concise."):
        """
        Process videos frame-by-frame with individual descriptions.
        """
        if isinstance(video_paths, str):
            video_paths = [video_paths]

        # Setup
        results_data = []
        batch_size = 1 # Qwen is large; processing 1 by 1 is usually more stable on M1 Pro
        frame_queue = queue.Queue(maxsize=4)
        stop_event = threading.Event()

        def producer():
            for path in video_paths:
                if not os.path.exists(path): continue
                stream = stream_frames(path, fps=fps)
                for frame, ts, r_time in stream:
                    if stop_event.is_set(): break
                    frame_queue.put((frame, ts, os.path.abspath(path)))
                    logger.debug(f"read time: {r_time}")
            frame_queue.put(None)

        threading.Thread(target=producer, daemon=True).start()

        # Consumer
        logger.debug(f"--- Starting Inference ---")
        while True:
            item = frame_queue.get()
            if item is None: break

            frame, ts, abs_path = item

            # We call the model (1 frame at a time for stability)
            descriptions, inf_time = self.vlm.describe_batch([frame], system_prompt=system_prompt)

            if store_frame:
                os.makedirs(store_image_path, exist_ok=True)
                frame_filename = f"{os.path.basename(abs_path)}_{ts:.2f}.png"
                Image.fromarray(frame).save(os.path.join(store_image_path, frame_filename))


            entry = {
                "video": abs_path,
                "timestamp": ts,
                "description": descriptions[0].text
            }
            results_data.append(entry)
            logger.debug(f"inference time: {inf_time}")

        # Save results
        with open(output_json, 'w') as f:
            json.dump(results_data, f, indent=4)
        logger.info(f"\nSaved {len(results_data)} descriptions to {output_json}")

    def process_video_as_sequence(self, video_list, output_json=None, fps=0.5, num_frames=5, store_frame=False, store_image_path="data/test_frames", system_prompt="You are a video analysis assistant."):
        """
        Process videos as frame sequences for temporal understanding.

        Args:
            video_list: List of video paths
            output_json: If provided, saves results to this path
            fps: Frames per second to extract
            num_frames: Number of frames to extract per video
            store_frame: Whether to save frame images
            store_image_path: Path to save frame images
            system_prompt: Custom system prompt

        Returns:
            List of dicts with 'video', 'num_frames', 'description'
        """
        results_data = []

        for video_path in video_list:
            if not os.path.exists(video_path): continue

            abs_path = os.path.abspath(video_path)

            frames = get_video_as_frame_sequence(abs_path, fps=fps, width=448)

            description, inf_time = self.vlm.describe_image_sequence(frames, fps, system_prompt=system_prompt)

            frame_idx = 0
            if store_frame:
                os.makedirs(store_image_path, exist_ok=True)
                for frame in frames:
                    frame_filename = f"{os.path.basename(abs_path)}_{frame_idx:.2f}.png"
                    Image.fromarray(frame).save(os.path.join(store_image_path, frame_filename))
                    frame_idx += 1

            entry = {
                "video": abs_path,
                "num_frames": len(frames),
                "description": description.text
            }
            results_data.append(entry)
            logger.debug(f"video: {abs_path}, inference time: {inf_time}")

        # Save to disk if output_json provided
        if output_json:
            with open(output_json, 'w') as f:
                json.dump(results_data, f, indent=4)
            logger.info(f"\nSaved {len(results_data)} descriptions to {output_json}")

        # ALWAYS return results for pipeline chaining
        return results_data

# ==========================================
# Backward Compatibility Functions
# ==========================================
def process_video_by_frame(video_paths, output_json, fps=0.5, store_frame=False, store_image_path="data/frames", system_prompt="You are a video analysis assistant. Your task is to watch the video frame by frame and describe what is happening in each frame in 50 words or less. Avoid introductory phrases like \"In this frame\". Keep it short and concise."):
    """
    Legacy function for backward compatibility.
    Maintains original API while using new VideoProcessor class.
    """
    processor = VideoProcessor()
    processor.process_video_by_frame(video_paths, output_json, fps=fps, store_frame=store_frame, store_image_path=store_image_path, system_prompt=system_prompt)

def process_video_as_sequence(video_list, output_json, fps=0.5, num_frames=5, store_frame=True, store_image_path="data/test_frames", system_prompt="You are a video analysis assistant."):
    """
    Legacy function for backward compatibility.
    Maintains original API while using new VideoProcessor class.
    """
    processor = VideoProcessor()
    processor.process_video_as_sequence(video_list, output_json, fps=fps, num_frames=num_frames, store_frame=store_frame, store_image_path=store_image_path, system_prompt=system_prompt)

if __name__ == "__main__":
    video_list = [
        "./data/DJI_20250803143631_0217_D_short.MP4",
        "./data/DJI_20250803144140_0220_D.MP4",
        "./data/DJI_20250801134716_0192_D.MP4",
        "./data/DJI_20250802150946_0204_D.MP4",
        "./data/DJI_20250802144037_0199_D.MP4",
        "./data/DJI_20250803143754_0218_D.MP4",
        "./data/DJI_20250514105245_0135_D.MP4"
    ]

    # system_prompt = """
    # You are a video analysis assistant. Your task is to watch the video frame by frame and describe what is happening in each frame in 50 words or less. Avoid introductory phrases like "In this frame". Keep it short and concise.
    # """
    # process_video_by_frame(video_list, "data/qwen_analysis.json", fps=0.2, store_frame=True, store_image_path="data/test_frames", system_prompt=system_prompt) # Sample every 5 seconds
    fps = 0.1
    seconds = int(1/fps)
    system_prompt = (
        "You are an expert cinematic analyst. The input is a sequence of images from one video. "
        "Watch the image sequence to understand the video and describe it in a narrative style. "
        "Pay attention to the environment, the movement of the camera, and the plot of the video. "
        "Describe it as a continuous sequence in **less than 100 words**."
    )
    process_video_as_sequence(video_list, "data/qwen_analysis_sequence.json", fps=fps, store_frame=True, store_image_path="data/test_frames_sequence", system_prompt=system_prompt) # Sample every 5 seconds
