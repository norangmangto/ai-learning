"""
Utility functions for image-to-video generation.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import cv2
from pathlib import Path


def create_video_grid(
    frames_list: List[List[Image.Image]],
    rows: int = 1,
    cols: int = 1
) -> List[Image.Image]:
    """
    Create a grid of multiple videos side-by-side.
    
    Args:
        frames_list: List of video frame lists
        rows: Number of rows in grid
        cols: Number of columns in grid
        
    Returns:
        List of combined frames
    """
    if len(frames_list) != rows * cols:
        raise ValueError(f"Expected {rows * cols} videos, got {len(frames_list)}")
    
    num_frames = min(len(frames) for frames in frames_list)
    combined_frames = []
    
    for frame_idx in range(num_frames):
        # Get all frames at this index
        current_frames = [frames_list[i][frame_idx] for i in range(len(frames_list))]
        
        # Convert to numpy arrays
        frame_arrays = [np.array(frame) for frame in current_frames]
        
        # Arrange in grid
        grid_rows = []
        for r in range(rows):
            row_frames = frame_arrays[r * cols:(r + 1) * cols]
            grid_rows.append(np.hstack(row_frames))
        
        grid = np.vstack(grid_rows)
        combined_frames.append(Image.fromarray(grid))
    
    return combined_frames


def interpolate_frames(
    frames: List[Image.Image],
    target_fps: int,
    source_fps: int
) -> List[Image.Image]:
    """
    Interpolate between frames to increase FPS.
    
    Args:
        frames: Input frames
        target_fps: Desired FPS
        source_fps: Original FPS
        
    Returns:
        Interpolated frames
    """
    if target_fps <= source_fps:
        return frames
    
    ratio = target_fps / source_fps
    interpolated = []
    
    for i in range(len(frames) - 1):
        frame1 = np.array(frames[i])
        frame2 = np.array(frames[i + 1])
        
        interpolated.append(frames[i])
        
        # Add interpolated frames
        num_intermediate = int(ratio) - 1
        for j in range(1, num_intermediate + 1):
            alpha = j / (num_intermediate + 1)
            blended = (1 - alpha) * frame1 + alpha * frame2
            interpolated.append(Image.fromarray(blended.astype(np.uint8)))
    
    interpolated.append(frames[-1])
    return interpolated


def add_text_overlay(
    frames: List[Image.Image],
    text: str,
    position: Tuple[int, int] = (10, 10),
    font_scale: float = 1.0,
    color: Tuple[int, int, int] = (255, 255, 255)
) -> List[Image.Image]:
    """
    Add text overlay to video frames.
    
    Args:
        frames: Input frames
        text: Text to overlay
        position: (x, y) position
        font_scale: Font size scale
        color: Text color (R, G, B)
        
    Returns:
        Frames with text overlay
    """
    overlayed_frames = []
    
    for frame in frames:
        # Convert to OpenCV format
        frame_cv = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        # Add text
        cv2.putText(
            frame_cv,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color[::-1],  # BGR format
            2,
            cv2.LINE_AA
        )
        
        # Convert back to PIL
        frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
        overlayed_frames.append(Image.fromarray(frame_rgb))
    
    return overlayed_frames


def create_loop_video(
    frames: List[Image.Image],
    num_loops: int = 2,
    reverse: bool = True
) -> List[Image.Image]:
    """
    Create a looping video by repeating or ping-ponging frames.
    
    Args:
        frames: Input frames
        num_loops: Number of times to loop
        reverse: If True, play forward then backward (ping-pong)
        
    Returns:
        Looped frames
    """
    if reverse:
        # Ping-pong: forward then backward
        forward = frames[:]
        backward = frames[-2:0:-1]  # Exclude first and last to avoid duplicate
        loop_frames = forward + backward
    else:
        # Simple repeat
        loop_frames = frames[:]
    
    return loop_frames * num_loops


def extract_keyframes(
    frames: List[Image.Image],
    num_keyframes: int = 5,
    method: str = "uniform"
) -> List[Tuple[int, Image.Image]]:
    """
    Extract keyframes from a video sequence.
    
    Args:
        frames: Input frames
        num_keyframes: Number of keyframes to extract
        method: "uniform" or "difference" based extraction
        
    Returns:
        List of (frame_index, frame) tuples
    """
    if method == "uniform":
        # Uniform sampling
        indices = np.linspace(0, len(frames) - 1, num_keyframes, dtype=int)
        return [(idx, frames[idx]) for idx in indices]
    
    elif method == "difference":
        # Select frames with maximum difference from previous
        keyframes = [(0, frames[0])]  # Always include first frame
        
        for _ in range(num_keyframes - 1):
            max_diff = 0
            max_idx = 0
            
            for i in range(len(frames)):
                # Skip already selected frames
                if any(i == kf[0] for kf in keyframes):
                    continue
                
                # Compute difference from nearest keyframe
                nearest_kf = min(keyframes, key=lambda kf: abs(kf[0] - i))
                diff = np.mean(np.abs(
                    np.array(frames[i]).astype(float) - 
                    np.array(nearest_kf[1]).astype(float)
                ))
                
                if diff > max_diff:
                    max_diff = diff
                    max_idx = i
            
            keyframes.append((max_idx, frames[max_idx]))
        
        return sorted(keyframes, key=lambda x: x[0])
    
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_temporal_consistency(frames: List[Image.Image]) -> float:
    """
    Compute temporal consistency score (lower is better).
    
    Args:
        frames: Input frames
        
    Returns:
        Consistency score (average frame difference)
    """
    if len(frames) < 2:
        return 0.0
    
    differences = []
    for i in range(len(frames) - 1):
        frame1 = np.array(frames[i]).astype(float)
        frame2 = np.array(frames[i + 1]).astype(float)
        diff = np.mean(np.abs(frame2 - frame1))
        differences.append(diff)
    
    return np.mean(differences)


def save_frames_as_gif(
    frames: List[Image.Image],
    output_path: str,
    duration: int = 100,
    loop: int = 0
):
    """
    Save frames as an animated GIF.
    
    Args:
        frames: Input frames
        output_path: Path to save GIF
        duration: Duration per frame in milliseconds
        loop: Number of loops (0 = infinite)
    """
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )
    print(f"✓ Saved GIF: {output_path}")


def create_side_by_side_comparison(
    original_frames: List[Image.Image],
    generated_frames: List[Image.Image],
    labels: Tuple[str, str] = ("Original", "Generated")
) -> List[Image.Image]:
    """
    Create side-by-side comparison video.
    
    Args:
        original_frames: Original video frames
        generated_frames: Generated video frames
        labels: Text labels for each side
        
    Returns:
        Combined frames
    """
    # Ensure same number of frames
    num_frames = min(len(original_frames), len(generated_frames))
    
    combined = []
    for i in range(num_frames):
        # Convert to numpy
        orig = np.array(original_frames[i])
        gen = np.array(generated_frames[i])
        
        # Ensure same height
        if orig.shape[0] != gen.shape[0]:
            target_height = min(orig.shape[0], gen.shape[0])
            orig = cv2.resize(orig, (orig.shape[1], target_height))
            gen = cv2.resize(gen, (gen.shape[1], target_height))
        
        # Concatenate horizontally
        combined_frame = np.hstack([orig, gen])
        
        # Add labels
        combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
        cv2.putText(combined_frame, labels[0], (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined_frame, labels[1], (orig.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB)
        
        combined.append(Image.fromarray(combined_frame))
    
    return combined


def analyze_motion(frames: List[Image.Image]) -> dict:
    """
    Analyze motion characteristics in a video.
    
    Args:
        frames: Input frames
        
    Returns:
        Dictionary with motion statistics
    """
    if len(frames) < 2:
        return {"error": "Need at least 2 frames"}
    
    # Convert to grayscale arrays
    gray_frames = [
        cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)
        for frame in frames
    ]
    
    # Compute optical flow
    flows = []
    for i in range(len(gray_frames) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i],
            gray_frames[i + 1],
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        flows.append(flow)
    
    # Compute statistics
    magnitudes = [np.sqrt(flow[..., 0]**2 + flow[..., 1]**2) for flow in flows]
    
    return {
        "avg_motion": float(np.mean([m.mean() for m in magnitudes])),
        "max_motion": float(np.max([m.max() for m in magnitudes])),
        "min_motion": float(np.min([m.min() for m in magnitudes])),
        "motion_variance": float(np.var([m.mean() for m in magnitudes]))
    }
