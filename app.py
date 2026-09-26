import streamlit as st
import cv2
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Import our new modules
from pose_analyzer import analyze_squat_form
from exercise_templates import get_exercise_template
from rep_counter import RepCounter
from utils import draw_colored_skeleton, draw_feedback_panel, draw_angle_text

# Set page configuration
st.set_page_config(
    page_title="Workout Form Analyzer",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🏋️ Workout Form Analyzer")
st.write("AI-powered real-time form analysis and rep counting for your workouts")

# Load the pre-trained YOLOv8-Pose model
@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt')

model = load_model()

# Define RTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Class to process video frames
class VideoProcessor:
    def __init__(self):
        self.exercise_template = get_exercise_template("squat")
        # Create our own rep_counter instance (can't rely on session_state in WebRTC thread)
        self.rep_counter = RepCounter(self.exercise_template)
        self.rep_count = 0
        self.current_phase = "Ready"
        self.form_score = 0
        self.avg_set_score = 0
        
        # Video recording
        self.recorded_frames = []  # Store annotated frames for playback
        self.is_recording = True  # Auto-start recording
        self.last_record_time = 0 # Track time for stable FPS recording
    
    def stop_recording(self):
        self.is_recording = False
        
    def start_recording(self):
        self.is_recording = True
        
    def clear_recording(self):
        self.recorded_frames = []
        self.rep_count = 0
        self.avg_set_score = 0
        self.rep_counter = RepCounter(self.exercise_template)
    
    def analyze_recorded_frames(self):
        """Analyze all recorded frames post-workout (like debug_video.py)"""
        # Reset rep counter for fresh analysis
        rep_counter = RepCounter(self.exercise_template)
        
        for frame_data in self.recorded_frames:
            keypoints = frame_data['keypoints']
            
            # Analyze form
            analysis = analyze_squat_form(keypoints, self.exercise_template)
            
            if analysis.get("detected", False):
                # Update rep counter
                rep_counter.update(
                    analysis["angles"]["knee"],
                    analysis["overall_score"],
                    analysis["angles"]["hip"]
                )
        
        # Store results
        self.rep_count = rep_counter.rep_count
        # Calculate average set score from rep_scores
        if rep_counter.rep_scores:
            self.avg_set_score = sum(r["form_score"] for r in rep_counter.rep_scores) / len(rep_counter.rep_scores)
        else:
            self.avg_set_score = 0
        self.rep_counter = rep_counter
        
        return rep_counter.rep_scores
    
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the frame to a NumPy array (BGR format)
        img = frame.to_ndarray(format="bgr24")
        
        # Resize for full-body view (tall enough to see feet)
        img = cv2.resize(img, (320, 480))

        # Run pose estimation on the frame
        results = model(img, stream=True, verbose=False)

        # Process results
        for result in results:
            # Get keypoints
            if result.keypoints is not None and len(result.keypoints) > 0:
                keypoints = result.keypoints.data[0].cpu().numpy()
                
                # Draw basic skeleton overlay (no analysis, just visual)
                annotated_frame = result.plot()
                
                # Record frame for post-workout analysis (Time-based: 3 FPS)
                import time
                current_time = time.time()
                if self.is_recording:
                    # Record if 0.33 seconds have passed (3 FPS)
                    if current_time - self.last_record_time >= 0.33:
                        # Store both the frame AND keypoints for later analysis
                        self.recorded_frames.append({
                            'frame': annotated_frame.copy(),
                            'keypoints': keypoints.copy()
                        })
                        self.last_record_time = current_time
                    
            else:
                # No keypoints detected
                annotated_frame = result.plot()
                cv2.putText(annotated_frame, "Step back - Show full body", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Convert the annotated frame back to a VideoFrame
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Sidebar - Exercise Selection & Info
st.sidebar.header("🎯 Exercise Selection")
st.sidebar.info("**Current Exercise: Squats**")
st.sidebar.markdown("---")

# Sidebar - Coaching Tips
st.sidebar.header("💡 Coaching Tips")
squat_template = get_exercise_template("squat")
for tip in squat_template["coaching_tips"][:3]:  # Show top 3 tips
    st.sidebar.markdown(f"• {tip}")

st.sidebar.markdown("---")

# Sidebar - Live Stats Info
st.sidebar.header("📊 Live Stats")
st.sidebar.info("Rep count, form score, and phase are displayed on the video feed in real-time!")

st.sidebar.markdown("---")

# Sidebar - Controls
st.sidebar.header("⚙️ Controls")
st.sidebar.write(" **Tip:** Stop and restart the camera to reset your rep counter")

# Main app interface - Two column layout
col_video, col_stats = st.columns([1, 1])

with col_video:
    st.subheader("📹 Live Feed")
    # Video Stream
    webrtc_ctx = webrtc_streamer(
        key="workout-form-analyzer",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 320}, 
                "height": {"ideal": 480}
            }, 
            "audio": False
        },
        async_processing=True,
    )
    
    # Status indicator
    if webrtc_ctx.state.playing:
        st.success("✅ Camera active")
    else:
        st.info("👆 Click START to begin")

with col_stats:
    st.subheader("📊 Workout Summary")
    
    st.info("� **Tip**: Focus on your form during exercise. Review your stats here after!")
    
    # Show current rep count only
    if webrtc_ctx.state.playing:
        try:
            if hasattr(webrtc_ctx, 'video_processor') and webrtc_ctx.video_processor is not None:
                processor = webrtc_ctx.video_processor
                st.metric("Current Reps", processor.rep_count, delta=None)
        except:
            st.metric("Current Reps", 0)
    else:
        st.metric("Current Reps", 0)
    
    st.markdown("---")
    
    # Initialize session state for summary visibility
    if 'show_summary' not in st.session_state:
        st.session_state.show_summary = False
    
    # Summary button controls
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("📋 Show Workout Summary", use_container_width=True, type="primary"):
            st.session_state.show_summary = True
            # Pause recording while viewing summary
            if hasattr(webrtc_ctx, 'video_processor') and webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.stop_recording()
    
    with col_btn2:
        if st.button("🔄 Reset Workout", use_container_width=True):
            st.session_state.show_summary = False
            # Clear all data and reset recording
            if hasattr(webrtc_ctx, 'video_processor') and webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.clear_recording()
                webrtc_ctx.video_processor.start_recording()
            st.rerun()
            
    with col_btn3:
        if st.button("❌ Close Summary", use_container_width=True):
            st.session_state.show_summary = False
            # Resume recording
            if hasattr(webrtc_ctx, 'video_processor') and webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.start_recording()
    
    # Display summary if flag is set
    if st.session_state.show_summary:
        if hasattr(webrtc_ctx, 'video_processor') and webrtc_ctx.video_processor:
            try:
                processor = webrtc_ctx.video_processor
                
                # Analyze recorded frames post-workout (same logic as debug_video.py)
                reps = processor.analyze_recorded_frames()
                
                st.markdown("---")
                st.subheader("🎯 Workout Results")
                
                # Video playback section
                if processor.recorded_frames and len(processor.recorded_frames) > 1:
                    st.markdown("### 📹 Workout Video Replay")
                    st.info(f"Recorded {len(processor.recorded_frames)} frames (~{len(processor.recorded_frames)/5:.1f} seconds)")
                    
                    # Initialize frame index in session state
                    if 'video_frame_idx' not in st.session_state:
                        st.session_state.video_frame_idx = 0
                    
                    # Ensure frame index is within bounds
                    if st.session_state.video_frame_idx >= len(processor.recorded_frames):
                        st.session_state.video_frame_idx = 0
                    
                    # Playback controls
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    
                    with col1:
                        if st.button("⏮️ First", use_container_width=True, key="btn_first"):
                            st.session_state.video_frame_idx = 0
                    
                    with col2:
                        if st.button("⏪ Prev", use_container_width=True, key="btn_prev"):
                            if st.session_state.video_frame_idx > 0:
                                st.session_state.video_frame_idx -= 1
                    
                    with col3:
                        if st.button("▶️ Play", use_container_width=True, key="btn_play"):
                            # Auto-play through frames with slower speed
                            video_placeholder = st.empty()
                            import time
                            for i in range(st.session_state.video_frame_idx, len(processor.recorded_frames)):
                                st.session_state.video_frame_idx = i
                                frame = processor.recorded_frames[i]['frame']
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                video_placeholder.image(frame_rgb, 
                                                       caption=f"Playing... Frame {i + 1}/{len(processor.recorded_frames)}", 
                                                       use_container_width=True)
                                time.sleep(0.5)  # 2 FPS playback (slower)
                            st.session_state.video_frame_idx = len(processor.recorded_frames) - 1
                    
                    with col4:
                        if st.button("⏩ Next", use_container_width=True, key="btn_next"):
                            if st.session_state.video_frame_idx < len(processor.recorded_frames) - 1:
                                st.session_state.video_frame_idx += 1
                    
                    with col5:
                        if st.button("⏭️ Last", use_container_width=True, key="btn_last"):
                            st.session_state.video_frame_idx = len(processor.recorded_frames) - 1
                    
                    with col6:
                        # Download video button
                        if st.button("💾 Save", use_container_width=True, key="btn_download"):
                            # Create video file from frames
                            import tempfile
                            import os
                            
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                            temp_path = temp_file.name
                            temp_file.close()
                            
                            # Get frame dimensions from first frame
                            first_frame = processor.recorded_frames[0]['frame']
                            height, width = first_frame.shape[:2]
                            
                            # Create video writer
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            out = cv2.VideoWriter(temp_path, fourcc, 3.0, (width, height))
                            
                            # Write all frames
                            for frame_data in processor.recorded_frames:
                                out.write(frame_data['frame'])
                            out.release()
                            
                            # Read file for download
                            with open(temp_path, 'rb') as f:
                                video_bytes = f.read()
                            
                            # Clean up temp file
                            os.unlink(temp_path)
                            
                            # Store in session state for download button
                            st.session_state.video_download = video_bytes
                            st.success("✅ Video ready for download!")
                    
                    # Show download button if video is ready
                    if 'video_download' in st.session_state and st.session_state.video_download:
                        st.download_button(
                            label="⬇️ Download Workout Video (MP4)",
                            data=st.session_state.video_download,
                            file_name="workout_analysis.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                    
                    # Frame slider with session state
                    new_frame_idx = st.slider(
                        "Frame position", 
                        0, 
                        len(processor.recorded_frames) - 1, 
                        st.session_state.video_frame_idx,
                        key="video_frame_slider_widget"
                    )
                    
                    # Update session state if slider moved (without rerun to prevent issues)
                    if new_frame_idx != st.session_state.video_frame_idx:
                        st.session_state.video_frame_idx = new_frame_idx
                    
                    # Display current frame
                    current_frame = processor.recorded_frames[st.session_state.video_frame_idx]['frame']
                    frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                    
                    st.image(frame_rgb, 
                           caption=f"Frame {st.session_state.video_frame_idx + 1}/{len(processor.recorded_frames)}", 
                           use_container_width=True)
                    
                    st.caption("💡 **Controls**: Prev/Next for frame-by-frame | ▶️ Play for auto-playback (2 FPS) | 💾 Save to download video")
                    st.caption("🎨 **Colors**: 🟢 Green skeleton = excellent form | 🟡 Yellow = good | 🔴 Red = needs improvement")
                    
                    st.markdown("---")
                elif processor.recorded_frames and len(processor.recorded_frames) == 1:
                    st.markdown("### 📹 Workout Video")
                    st.info("Only 1 frame recorded - do more reps to see progression!")
                    
                    # Display the single frame
                    frame_rgb = cv2.cvtColor(processor.recorded_frames[0]['frame'], cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, caption="Recorded frame", use_container_width=True)
                    st.markdown("---")
                else:
                    st.warning("⚠️ No video frames recorded. Make sure camera is active and you perform some squats!")
                
                # Total reps
                st.success(f"**Total Reps Completed:** {processor.rep_count}")
                
                # Average form score
                if processor.rep_count > 0:
                    avg_score = int(processor.avg_set_score)
                    if avg_score >= 85:
                        st.success(f"**Average Form Score:** {avg_score}% ✅ Excellent!")
                    elif avg_score >= 70:
                        st.warning(f"**Average Form Score:** {avg_score}% ⚠️ Good")
                    else:
                        st.error(f"**Average Form Score:** {avg_score}% ❌ Needs Improvement")
                    
                    # Rep-by-rep breakdown
                    st.markdown("### 📈 Rep Breakdown")
                    
                    if reps:
                        for rep in reps:
                            rep_num = rep['rep_number']
                            score = int(rep['form_score'])
                            min_angle = int(rep['min_knee_angle'])
                            
                            # Color code based on score
                            if score >= 85:
                                st.success(f"Rep {rep_num}: {score}% (Knee: {min_angle}°) ✅")
                            elif score >= 70:
                                st.warning(f"Rep {rep_num}: {score}% (Knee: {min_angle}°) ⚠️")
                            else:
                                st.error(f"Rep {rep_num}: {score}% (Knee: {min_angle}°) ❌")
                else:
                    st.info("No reps completed yet. Start your workout!")
                    
            except Exception as e:
                st.error(f"Error loading workout data: {str(e)}")
        else:
            st.warning("Start the camera first to track your workout!")

# Instructions
with st.expander("📋 How to Use"):
    st.markdown("""
    ### Getting Started
    1. **Position yourself** so your full body is visible in the camera
    2. **Stand sideways** to the camera for best squat analysis
    3. **Click START** to begin pose detection
    
    ### Understanding Feedback
    - **Green skeleton** = Perfect form ✅
    - **Yellow skeleton** = Good form, minor adjustment needed ⚠️
    - **Red skeleton** = Poor form, needs correction ❌
    
    ### Rep Counting
    - Reps are counted automatically as you complete each squat
    - Must reach proper depth (90° knee angle) for rep to count
    - Form score is tracked for each rep
    
    ### Tips for Best Results
    - Ensure good lighting
    - Keep entire body in frame
    - Perform movements slowly and controlled
    - Follow the on-screen feedback messages
    """)