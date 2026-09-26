"""
Pose Analysis Module
Calculates angles between keypoints and analyzes exercise form quality.
"""

import numpy as np
from typing import Tuple, Dict, Optional


def calculate_angle(point1: Tuple[float, float], 
                   point2: Tuple[float, float], 
                   point3: Tuple[float, float]) -> float:
    """
    Calculate the angle between three points.
    
    Args:
        point1: First point (x, y) - e.g., hip
        point2: Vertex point (x, y) - e.g., knee
        point3: Third point (x, y) - e.g., ankle
    
    Returns:
        Angle in degrees (0-180)
    """
    # Convert points to numpy arrays
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Clamp to avoid numerical errors with arccos
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    
    # Convert to degrees
    return np.degrees(angle)


def get_keypoint(keypoints, index: int) -> Optional[Tuple[float, float]]:
    """
    Extract a specific keypoint from YOLO pose results.
    
    Args:
        keypoints: Keypoints array from YOLO results
        index: Index of the keypoint to extract
    
    Returns:
        Tuple of (x, y) coordinates or None if not detected
    """
    if keypoints is None or len(keypoints) <= index:
        return None
    
    kp = keypoints[index]
    x, y = float(kp[0]), float(kp[1])
    
    # Check if keypoint is valid (not 0,0)
    if x == 0 and y == 0:
        return None
    
    return (x, y)


def analyze_squat_form(keypoints, exercise_template: Dict) -> Dict:
    """
    Analyze squat form quality based on keypoints.
    
    Args:
        keypoints: Detected keypoints from YOLO
        exercise_template: Exercise template with ideal ranges
    
    Returns:
        Dictionary with form analysis including angles, scores, and feedback
    """
    # Extract key joints for squat (using right side for analysis)
    hip = get_keypoint(keypoints, 12)      # Right hip
    knee = get_keypoint(keypoints, 14)     # Right knee
    ankle = get_keypoint(keypoints, 16)    # Right ankle
    shoulder = get_keypoint(keypoints, 6)  # Right shoulder
    
    # Check if all required keypoints are detected
    if not all([hip, knee, ankle, shoulder]):
        return {
            "detected": False,
            "message": "Cannot detect full body. Please step back from camera."
        }
    
    # Calculate key angles
    knee_angle = calculate_angle(hip, knee, ankle)
    
    # Calculate hip angle (shoulder-hip-knee for body position)
    hip_angle = calculate_angle(shoulder, hip, knee)
    
    # Analyze form quality
    ideal_knee_range = exercise_template["ideal_angles"]["knee_angle"]
    form_feedback = []
    angle_scores = {}
    
    # Determine squat phase based on knee angle
    if knee_angle > 160:
        phase = "standing"
        target_range = ideal_knee_range["at_top"]
    else:
        phase = "squatting"
        target_range = ideal_knee_range["at_bottom"]
    
    # Score knee angle (0-100)
    knee_score = score_angle(knee_angle, target_range)
    angle_scores["knee"] = knee_score
    
    # Check for common form mistakes
    # Check for common form mistakes using template thresholds
    form_checks = exercise_template.get("form_checks", {})
    
    if phase == "squatting":
        # Check depth (not deep enough)
        not_deep_check = form_checks.get("not_deep_enough", {})
        if not_deep_check and knee_angle > not_deep_check.get("threshold", 110):
            form_feedback.append(not_deep_check.get("message", "⬇️ Squat deeper!"))
            
        # Check depth (too deep)
        too_deep_check = form_checks.get("too_deep", {})
        if too_deep_check and knee_angle < too_deep_check.get("threshold", 75):
            form_feedback.append(too_deep_check.get("message", "⚠️ Too low - protect knees"))
    
    # Check body alignment (back rounding / chest up)
    back_check = form_checks.get("back_rounding", {})
    if back_check and hip_angle < back_check.get("threshold", 140):
        form_feedback.append(back_check.get("message", "📐 Chest up!"))
    
    # Add positive reinforcement
    if knee_score >= 90 and not form_feedback:
        form_feedback.append("✅ Perfect form!")
    
    # Overall form score (average of all angle scores)
    overall_score = np.mean(list(angle_scores.values()))
    
    # Classify form quality
    if overall_score >= 85:
        form_quality = "Perfect"
    elif overall_score >= 70:
        form_quality = "Good"
    elif overall_score >= 50:
        form_quality = "Fair"
    else:
        form_quality = "Poor"
    
    return {
        "detected": True,
        "phase": phase,
        "angles": {
            "knee": knee_angle,
            "hip": hip_angle
        },
        "scores": angle_scores,
        "overall_score": overall_score,
        "form_quality": form_quality,
        "feedback": form_feedback if form_feedback else ["👍 Good form!"],
        "keypoints": {
            "hip": hip,
            "knee": knee,
            "ankle": ankle,
            "shoulder": shoulder
        }
    }


def score_angle(current_angle: float, ideal_range: Tuple[float, float]) -> float:
    """
    Score an angle based on how close it is to the ideal range.
    
    Args:
        current_angle: Current measured angle
        ideal_range: Tuple of (min, max) ideal angle
    
    Returns:
        Score from 0-100
    """
    min_ideal, max_ideal = ideal_range
    
    # Perfect score if within ideal range
    if min_ideal <= current_angle <= max_ideal:
        return 100.0
    
    # Calculate distance from ideal range
    if current_angle < min_ideal:
        distance = min_ideal - current_angle
    else:
        distance = current_angle - max_ideal
    
    # Score decreases with distance (10 degrees off = 50 points lost)
    score = max(0, 100 - (distance * 5))
    
    return score


def get_joint_color(score: float) -> Tuple[int, int, int]:
    """
    Get BGR color for joint based on form score.
    
    Args:
        score: Form score (0-100)
    
    Returns:
        BGR tuple for OpenCV
    """
    if score >= 85:
        return (0, 255, 0)      # Green - Perfect
    elif score >= 70:
        return (0, 255, 255)    # Yellow - Good
    elif score >= 50:
        return (0, 165, 255)    # Orange - Fair
    else:
        return (0, 0, 255)      # Red - Poor
