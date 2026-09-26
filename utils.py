"""
Utility Functions
Helper functions for visualization and display.
"""

import cv2
import numpy as np
from typing import Tuple, Dict


def draw_angle_text(frame, point: Tuple[float, float], angle: float, color: Tuple[int, int, int]):
    """
    Draw angle measurement text near a joint.
    
    Args:
        frame: Video frame to draw on
        point: (x, y) position to draw near
        angle: Angle value to display
        color: BGR color tuple
    """
    text = f"{int(angle)}"
    x, y = int(point[0]), int(point[1])
    
    # Position text slightly offset from point
    text_pos = (x + 15, y - 15)
    
    # Draw text with background for visibility
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Get text size for background
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw semi-transparent background
    cv2.rectangle(frame, 
                 (text_pos[0] - 5, text_pos[1] - text_height - 5),
                 (text_pos[0] + text_width + 5, text_pos[1] + baseline + 5),
                 (0, 0, 0), -1)
    
    # Draw text
    cv2.putText(frame, text, text_pos, font, font_scale, color, thickness, cv2.LINE_AA)


def draw_colored_skeleton(frame, keypoints, analysis_result: Dict, exercise_template: Dict):
    """
    Draw skeleton overlay with color-coded joints based on form quality.
    
    Args:
        frame: Video frame to draw on
        keypoints: Detected keypoints
        analysis_result: Form analysis result with scores
        exercise_template: Exercise template
    """
    if not analysis_result.get("detected", False):
        return
    
    # Get joint positions
    joints = analysis_result["keypoints"]
    scores = analysis_result["scores"]
    
    # Define skeleton connections for squat (simplified)
    connections = [
        ("shoulder", "hip"),
        ("hip", "knee"),
        ("knee", "ankle")
    ]
    
    # Draw connections
    for joint1_name, joint2_name in connections:
        if joint1_name in joints and joint2_name in joints:
            pt1 = joints[joint1_name]
            pt2 = joints[joint2_name]
            
            # Get color based on joint score
            if joint2_name in scores:
                color = get_score_color(scores[joint2_name])
            else:
                color = (200, 200, 200)  # Gray for non-scored joints
            
            # Draw line
            cv2.line(frame, 
                    (int(pt1[0]), int(pt1[1])), 
                    (int(pt2[0]), int(pt2[1])), 
                    color, 3)
    
    # Draw joints as circles
    for joint_name, point in joints.items():
        if joint_name in scores:
            color = get_score_color(scores[joint_name])
            radius = 8
        else:
            color = (200, 200, 200)
            radius = 6
        
        cv2.circle(frame, (int(point[0]), int(point[1])), radius, color, -1)
        cv2.circle(frame, (int(point[0]), int(point[1])), radius + 2, (255, 255, 255), 2)


def get_score_color(score: float) -> Tuple[int, int, int]:
    """
    Get BGR color based on form score.
    
    Args:
        score: Form score (0-100)
    
    Returns:
        BGR color tuple
    """
    if score >= 85:
        return (0, 255, 0)      # Green - Perfect
    elif score >= 70:
        return (0, 255, 255)    # Yellow - Good
    elif score >= 50:
        return (0, 165, 255)    # Orange - Fair
    else:
        return (0, 0, 255)      # Red - Poor


def draw_feedback_panel(frame, analysis_result: Dict, rep_info: Dict):
    """
    Draw compact feedback panel with form score, reps, and messages.
    
    Args:
        frame: Video frame to draw on
        analysis_result: Form analysis result
        rep_info: Rep counter information
    """
    h, w, _ = frame.shape
    
    # Compact panel in top-right corner
    panel_width = min(w - 20, 220)
    panel_x = w - panel_width - 10
    panel_y = 10
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = panel_y + 25
    
    # Semi-transparent background
    overlay = frame.copy()
    
    # Rep count - Large and prominent
    rep_text = f"REPS: {rep_info['rep_count']}"
    cv2.rectangle(overlay, (panel_x, y_offset - 20), (panel_x + panel_width, y_offset + 10), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, rep_text, (panel_x + 10, y_offset), font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    y_offset += 40
    
    if analysis_result.get("detected", False):
        score = analysis_result["overall_score"]
        score_color = get_score_color(score)
        quality = analysis_result["form_quality"]
        
        # Form score with quality
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (panel_x, y_offset - 20), (panel_x + panel_width, y_offset + 10), (50, 50, 50), -1)
        cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
        
        form_text = f"FORM: {int(score)}% - {quality}"
        cv2.putText(frame, form_text, (panel_x + 10, y_offset), font, 0.5, score_color, 2, cv2.LINE_AA)
        y_offset += 35
        
        # Current phase
        phase_text = rep_info.get('phase_display', rep_info['current_phase'])
        cv2.putText(frame, phase_text, (panel_x + 10, y_offset), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y_offset += 30
        
        # Feedback message - simplified and clearer
        if analysis_result.get("feedback"):
            feedback_text = analysis_result["feedback"][0]
            
            # Shorten feedback for readability
            max_chars = 25
            if len(feedback_text) > max_chars:
                # Split into multiple lines
                words = feedback_text.split()
                lines = []
                current_line = ""
                for word in words:
                    if len(current_line) + len(word) + 1 <= max_chars:
                        current_line += word + " "
                    else:
                        if current_line:
                            lines.append(current_line.strip())
                        current_line = word + " "
                if current_line:
                    lines.append(current_line.strip())
                
                # Background for feedback
                feedback_height = len(lines[:2]) * 22 + 10
                overlay3 = frame.copy()
                cv2.rectangle(overlay3, (panel_x, y_offset - 18), 
                            (panel_x + panel_width, y_offset + feedback_height), 
                            (40, 40, 80), -1)
                cv2.addWeighted(overlay3, 0.8, frame, 0.2, 0, frame)
                
                for line in lines[:2]:  # Max 2 lines
                    cv2.putText(frame, line, (panel_x + 8, y_offset), font, 0.45, (100, 255, 255), 1, cv2.LINE_AA)
                    y_offset += 22
            else:
                overlay3 = frame.copy()
                cv2.rectangle(overlay3, (panel_x, y_offset - 18), 
                            (panel_x + panel_width, y_offset + 12), 
                            (40, 40, 80), -1)
                cv2.addWeighted(overlay3, 0.8, frame, 0.2, 0, frame)
                cv2.putText(frame, feedback_text, (panel_x + 8, y_offset), font, 0.45, (100, 255, 255), 1, cv2.LINE_AA)
    else:
        # Not detected message
        msg = "STAND BACK - SHOW FULL BODY"
        cv2.putText(frame, msg, (panel_x + 10, y_offset), font, 0.45, (0, 0, 255), 1, cv2.LINE_AA)


def draw_angle_arc(frame, vertex: Tuple[float, float], point1: Tuple[float, float], 
                   point2: Tuple[float, float], angle: float, color: Tuple[int, int, int]):
    """
    Draw an arc showing the angle at a joint (optional enhancement).
    
    Args:
        frame: Video frame to draw on
        vertex: Vertex point of the angle (e.g., knee)
        point1: First point (e.g., hip)
        point2: Second point (e.g., ankle)
        angle: Angle value in degrees
        color: BGR color
    """
    # Calculate angles for arc
    v = np.array(vertex)
    p1 = np.array(point1)
    p2 = np.array(point2)
    
    angle1 = np.degrees(np.arctan2(p1[1] - v[1], p1[0] - v[0]))
    angle2 = np.degrees(np.arctan2(p2[1] - v[1], p2[0] - v[0]))
    
    # Draw arc
    radius = 40
    cv2.ellipse(frame, (int(v[0]), int(v[1])), (radius, radius), 
               0, int(angle1), int(angle2), color, 2)
