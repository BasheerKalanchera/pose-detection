import streamlit as st
import cv2
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Set page configuration
st.set_page_config(
    page_title="MediaPipe Pose Estimation",
    page_icon="🤸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤸 Real-Time Pose Estimation with MediaPipe")
st.write("This app uses Google's MediaPipe Pose model to perform real-time human pose estimation on a live webcam feed.")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define RTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Class to process video frames
class VideoProcessor:
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # --- MediaPipe Processing ---
        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image to find the pose
        results = pose.process(image_rgb)
        
        # Draw the pose annotation on the original BGR image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=img,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

        # Draw the resolution on the frame
        h, w, _ = img.shape
        resolution_text = f"Resolution: {w}x{h}"
        cv2.putText(img, resolution_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Main app interface
webrtc_streamer(
    key="pose-estimation-mediapipe",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.sidebar.header("About")
st.sidebar.info("This application uses MediaPipe for real-time human pose estimation, deployed on Streamlit Cloud.")