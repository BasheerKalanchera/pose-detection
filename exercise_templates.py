"""
Exercise Templates
Define ideal form parameters for different exercises.
"""

# YOLO Pose Keypoint Indices (COCO format)
KEYPOINT_INDICES = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16
}


SQUAT_TEMPLATE = {
    "name": "Squat",
    "description": "Standard bodyweight squat - lower body strength exercise",
    
    # Key joints to monitor (using right side for simplicity in MVP)
    "key_joints": {
        "hip": KEYPOINT_INDICES["right_hip"],
        "knee": KEYPOINT_INDICES["right_knee"],
        "ankle": KEYPOINT_INDICES["right_ankle"],
        "shoulder": KEYPOINT_INDICES["right_shoulder"]
    },
    
    # Ideal angle ranges for good form
    "ideal_angles": {
        "knee_angle": {
            "at_bottom": (70, 100),   # Interior hip-knee-ankle angle: parallel ~90°, deeper squats go below 90°
            "at_top": (160, 180)      # Standing position (nearly straight legs)
        },
        "hip_angle": {
            "at_bottom": (95, 135),   # Parallel: 95-110°, Deep: >125° hip angle
            "at_top": (160, 180)      # Standing upright
        },
        "back_angle": {
            "range": (140, 180)       # Torso should stay relatively upright
        }
    },
    
    # Form check conditions
    "form_checks": {
        "not_deep_enough": {
            "threshold": 100,          # If knee > 100°, not reaching parallel
            "message": "⬇️ Go deeper - aim for parallel (thighs level)"
        },
        "too_deep": {
            "threshold": 30,           # Extreme depth may indicate form breakdown
            "message": "⚠️ Very deep - ensure you maintain form"
        },
        "back_rounding": {
            "threshold": 90,           # Hip < 90° suggests excessive forward lean
            "message": "📐 Chest up - avoid excessive forward lean"
        }
    },
    
    # Rep counting configuration
    "rep_counter": {
        "phases": ["standing", "descending", "bottom", "ascending"],
        "triggers": {
            "start_descending": 170,   # Knee angle drops below this
            "reached_bottom": 100,     # Knee angle reaches this or lower
            "start_ascending": 5,      # Knee angle increases by this amount from bottom
            "complete_rep": 160        # Knee angle exceeds this to complete rep
        },
        "minimum_bottom_hold": 0.1     # Seconds to hold at bottom (prevents bouncing)
    },
    
    # Coaching cues
    "coaching_tips": [
        "Keep your chest up and core engaged",
        "Push through your heels",
        "Knees should track over your toes",
        "Break parallel - hips below knees",
        "Maintain a neutral spine"
    ]
}


# Dictionary of all available exercises (expandable in future)
EXERCISE_TEMPLATES = {
    "squat": SQUAT_TEMPLATE
}


def get_exercise_template(exercise_name: str):
    """
    Get the template for a specific exercise.
    
    Args:
        exercise_name: Name of the exercise (lowercase)
    
    Returns:
        Exercise template dictionary or None if not found
    """
    return EXERCISE_TEMPLATES.get(exercise_name.lower())
