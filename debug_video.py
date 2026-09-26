import cv2
import numpy as np
from ultralytics import YOLO
# Import functions directly instead of class
from pose_analyzer import analyze_squat_form
from exercise_templates import get_exercise_template
from rep_counter import RepCounter
import os

def analyze_video(video_path):
    print(f"🎬 analyzing: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at {video_path}")
        return

    # Load model and tools
    try:
        model = YOLO('yolov8n-pose.pt')
        template = get_exercise_template("squat")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return

    # Tracking variables
    frame_count = 0
    rep_counter = RepCounter(template)
    
    print("🚀 Starting analysis...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        
        # Run inference
        results = model(frame, verbose=False)
        
        for result in results:
            if result.keypoints is not None and len(result.keypoints) > 0:
                keypoints = result.keypoints.data[0].cpu().numpy()
                landmarks = keypoints # Use keypoints directly
        
                if landmarks is not None:
                    # Get analysis using function directly
                    analysis = analyze_squat_form(landmarks, template)
                    
                    if analysis.get("detected", False):
                        # Update rep counter using same logic as app
                        rep_info = rep_counter.update(
                            analysis["angles"]["knee"],
                            analysis["overall_score"],
                            analysis["angles"]["hip"]
                        )
                        
                        # Print rep summary when completed
                        if rep_info["rep_completed"]:
                            # New rep just completed
                            last_rep = rep_counter.rep_scores[-1]
                            rep_num = len(rep_counter.rep_scores)
                            
                            print(f"\n📊 REP {rep_num} ANALYSIS (Frame {frame_count})")
                            print(f"   📉 Lowest Point Stats:")
                            print(f"   • Knee Angle: {last_rep['min_knee_angle']:.1f}° (Target: 40-105°)")
                            print(f"   • Hip Angle:  {last_rep['min_hip_angle']:.1f}° (Target: >140° for good posture)")
                            print(f"   • Form Score: {last_rep['form_score']:.1f}%")
                            
                            # Diagnosis
                            score = last_rep['form_score']
                            knee = last_rep['min_knee_angle']
                            hip = last_rep['min_hip_angle']
                            
                            if score < 70:
                                print("   ❌ WHY LOW SCORE?")
                                issues = []
                                # Knee diagnostics: Parallel 60-70°, Deep 35-60°
                                if knee < 30:
                                    issues.append("Extreme depth - may indicate form breakdown")
                                elif knee > 100:
                                    issues.append("Not deep enough - didn't reach parallel")
                                
                                # Hip diagnostics: Good range 95-135°
                                if hip < 90:
                                    issues.append(f"Excessive forward lean (Hip: {hip:.0f}° < 90°)")
                                elif hip > 140:
                                    issues.append("Too upright, may not be deep enough")
                                
                                for issue in issues:
                                    print(f"      -> {issue}")
                                
                                if not issues:
                                    print("      -> Form issues during rep motion (not just at bottom)")
                            elif score >= 90:
                                print("   ✅ Perfect form!")

    cap.release()
    print("\n✅ Analysis Complete")

if __name__ == "__main__":
    # Use the filename provided by the user
    analyze_video("Video Project 4.mp4")
