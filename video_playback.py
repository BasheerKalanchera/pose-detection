"""
Video Playback Utilities
Helper function for video playback in Streamlit
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os


def create_video_from_frames(frames, fps=10):
    """
    Create a video file from a list of frames.
    
    Args:
        frames: List of numpy arrays (frames)
        fps: Frames per second
    
    Returns:
        Path to the created video file
    """
    if not frames:
        return None
    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_path = temp_file.name
    temp_file.close()
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame in frames:
        out.write(frame)
    
    out.release()
    
    return temp_path


def display_video_player(frames):
    """
    Display video player with frame-by-frame controls.
    
    Args:
        frames: List of frames to display
    """
    if not frames:
        st.warning("No video frames recorded")
        return
    
    st.info(f"📹 Recorded {len(frames)} frames")
    
    # Frame slider
    frame_idx = st.slider("Frame", 0, len(frames) - 1, 0)
    
    # Display current frame
    frame = frames[frame_idx]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, caption=f"Frame {frame_idx + 1}/{len(frames)}", use_container_width=True)
    
    # Playback controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⏮️ Previous", use_container_width=True):
            if frame_idx > 0:
                st.rerun()
    
    with col2:
        if st.button("▶️ Play All", use_container_width=True):
            # Show all frames as animation
            placeholder = st.empty()
            for i, frame in enumerate(frames):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                placeholder.image(frame_rgb, caption=f"Frame {i + 1}/{len(frames)}")
                st.time.sleep(0.1)  # 10 FPS
    
    with col3:
        if st.button("⏭️ Next", use_container_width=True):
            if frame_idx < len(frames) - 1:
                st.rerun()
