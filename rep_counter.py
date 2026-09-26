"""
Rep Counter Module
State machine for tracking exercise repetitions.
"""

import time
from typing import Dict, Optional


class RepCounter:
    """
    Tracks repetitions for exercises using a state machine approach.
    """
    
    def __init__(self, exercise_template: Dict):
        """
        Initialize rep counter with exercise template.
        
        Args:
            exercise_template: Exercise configuration with rep counting triggers
        """
        self.template = exercise_template
        self.config = exercise_template["rep_counter"]
        
        # State tracking
        self.current_phase = "standing"
        self.rep_count = 0
        self.current_rep_start_time = None
        self.bottom_reached_time = None
        self.min_knee_angle = 180  # Track minimum knee angle in current rep
        self.min_hip_angle = 180   # Track minimum hip angle in current rep
        
        # Rep quality tracking
        self.rep_scores = []
        self.current_rep_scores = []
        
        # Phase history (for debugging)
        self.phase_history = []
    
    def reset(self):
        """Reset the counter for a new set."""
        self.current_phase = "standing"
        self.rep_count = 0
        self.current_rep_start_time = None
        self.bottom_reached_time = None
        self.min_knee_angle = 180
        self.min_hip_angle = 180
        self.rep_scores = []
        self.current_rep_scores = []
        self.phase_history = []
    
    def update(self, knee_angle: float, form_score: float, hip_angle: Optional[float] = None) -> Dict:
        """
        Update rep counter based on current knee angle.
        
        Args:
            knee_angle: Current knee angle in degrees
            form_score: Current form quality score (0-100)
            hip_angle: Current hip angle in degrees (optional)
        
        Returns:
            Dictionary with rep status information
        """
        triggers = self.config["triggers"]
        phase_changed = False
        rep_completed = False
        
        # Track minimum knee angle (deepest point)
        if knee_angle < self.min_knee_angle:
            self.min_knee_angle = knee_angle
        
        # Track minimum hip angle
        if hip_angle is not None and hip_angle < self.min_hip_angle:
            self.min_hip_angle = hip_angle
        
        # Track form scores during rep
        self.current_rep_scores.append(form_score)
        
        # State machine logic
        if self.current_phase == "standing":
            # Waiting for descent to begin
            if knee_angle < triggers["start_descending"]:
                self.current_phase = "descending"
                self.current_rep_start_time = time.time()
                self.min_knee_angle = knee_angle
                self.current_rep_scores = [form_score]
                phase_changed = True
        
        elif self.current_phase == "descending":
            # Going down
            if knee_angle <= triggers["reached_bottom"]:
                self.current_phase = "bottom"
                self.bottom_reached_time = time.time()
                phase_changed = True
        
        elif self.current_phase == "bottom":
            # At the bottom - check if starting to come up
            # Must hold bottom for minimum time to prevent bouncing
            if self.bottom_reached_time:
                time_at_bottom = time.time() - self.bottom_reached_time
                if time_at_bottom >= self.config["minimum_bottom_hold"]:
                    if knee_angle > self.min_knee_angle + triggers["start_ascending"]:
                        self.current_phase = "ascending"
                        phase_changed = True
        
        elif self.current_phase == "ascending":
            # Coming back up
            if knee_angle >= triggers["complete_rep"]:
                # Rep complete!
                self.current_phase = "standing"
                self.rep_count += 1
                phase_changed = True
                rep_completed = True
                
                # Calculate average form score for this rep
                avg_rep_score = sum(self.current_rep_scores) / len(self.current_rep_scores) if self.current_rep_scores else 0
                self.rep_scores.append({
                    "rep_number": self.rep_count,
                    "form_score": avg_rep_score,
                    "min_knee_angle": self.min_knee_angle,
                    "min_hip_angle": self.min_hip_angle
                })
                
                # Reset for next rep
                self.min_knee_angle = 180
                self.min_hip_angle = 180
                self.current_rep_scores = []
        
        # Track phase changes
        if phase_changed:
            self.phase_history.append({
                "phase": self.current_phase,
                "time": time.time(),
                "knee_angle": knee_angle
            })
        
        # Calculate average form score for the set
        avg_set_score = 0
        if self.rep_scores:
            avg_set_score = sum(r["form_score"] for r in self.rep_scores) / len(self.rep_scores)
        
        return {
            "rep_count": self.rep_count,
            "current_phase": self.current_phase,
            "phase_changed": phase_changed,
            "rep_completed": rep_completed,
            "avg_set_score": avg_set_score,
            "rep_scores": self.rep_scores
        }
    
    def get_phase_display(self) -> str:
        """
        Get user-friendly display name for current phase.
        
        Returns:
            Formatted phase name
        """
        phase_display = {
            "standing": "Ready",
            "descending": "Going Down ↓",
            "bottom": "At Bottom ⬇",
            "ascending": "Coming Up ↑"
        }
        return phase_display.get(self.current_phase, self.current_phase)
