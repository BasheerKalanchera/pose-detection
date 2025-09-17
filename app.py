import streamlit as st
import cv2
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Set page configuration
st.set_page_config(
    page_title="YOLOv8 Pose Estimation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Real-Time Pose Estimation using YOLOv8")
st.write("This app uses YOLOv8-Pose to perform real-time human pose estimation on a live webcam feed.")

# Load the pre-trained YOLOv8-Pose model
model = YOLO('yolov8n-pose.pt')

# Define RTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Class to process video frames
class VideoProcessor:
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model(img, stream=True)

        for result in results:
            annotated_frame = result.plot()

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# Main app interface
webrtc_ctx = webrtc_streamer(
    key="pose-estimation",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640}, 
            "height": {"ideal": 480}
        }, 
        "audio": False
    },
    async_processing=True,
)

st.sidebar.header("About")
st.sidebar.info("This application is a demonstration of real-time human pose estimation using YOLOv8.")

if webrtc_ctx.state.playing:
    st.success("Webcam is active!")
else:
    st.info("Click 'START' to begin.")