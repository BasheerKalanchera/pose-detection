import streamlit as st
import cv2
import av
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Set page configuration
st.set_page_config(
    page_title="Pose Estimation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Real-Time Pose Estimation")
st.write("This app uses AI to perform real-time human pose estimation on a live webcam feed.")

# Load the pre-trained YOLOv8-Pose model
model = YOLO('yolov8n-pose.pt')

# Define RTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Class to process video frames
class VideoProcessor:
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the frame to a NumPy array (BGR format)
        img = frame.to_ndarray(format="bgr24")
        
        # --- Force resize to guarantee performance ---
        img = cv2.resize(img, (640, 480))
        # ---------------------------------------------

        # Run pose estimation on the frame
        results = model(img, stream=True)

        # Process results
        for result in results:
            # The plot() method returns a BGR NumPy array with all annotations
            annotated_frame = result.plot()
            
            # --- Add this block to draw the resolution on the frame ---
            h, w, _ = annotated_frame.shape
            resolution_text = f"Resolution: {w}x{h}"
            
            # Position the text at the top-left corner of the frame
            cv2.putText(annotated_frame, resolution_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # -------------------------------------------------------------

        # Convert the annotated frame back to a VideoFrame
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
st.sidebar.info("This application is a demonstration of real-time human pose estimation using AI")

if "frame_size" not in st.session_state:
    st.session_state.frame_size = "N/A"

st.sidebar.markdown("---")
st.sidebar.header("Stream Information")
st.sidebar.write(f"**Resolution:** {st.session_state.frame_size}")

if webrtc_ctx.state.playing:
    st.success("Webcam is active!")
else:
    st.info("Click 'START' to begin.")