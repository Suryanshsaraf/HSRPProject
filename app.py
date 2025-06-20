import streamlit as st
import tempfile
import os
import pandas as pd
from speed import process_video  # Assumes you have a function to process video and return results

st.set_page_config(page_title="Vehicle Speed Detection", layout="wide")

st.title("🚗 Vehicle Speed Detection with HSRP")
st.markdown("""
Upload a video of traffic, and this app will detect vehicle speeds and annotate the video. You can view the results and download them for further analysis.
""")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input:
        temp_input.write(uploaded_file.read())
        input_video_path = temp_input.name

    st.video(input_video_path)
    st.info("Processing video. This may take a while depending on video length...")

    # Process the video and get results
    # process_video should return (annotated_video_path, results_df)
    annotated_video_path, results_df = process_video(input_video_path)

    st.success("Processing complete!")

    st.subheader("Annotated Video")
    st.video(annotated_video_path)

    st.subheader("Detected Speeds Table")
    st.dataframe(results_df)

    # Download buttons
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name="detected_speeds.csv",
        mime="text/csv"
    )
    with open(annotated_video_path, "rb") as f:
        st.download_button(
            label="Download Annotated Video",
            data=f,
            file_name="annotated_video.mp4",
            mime="video/mp4"
        )

    # Clean up temp files
    os.remove(input_video_path)
    # Optionally remove annotated_video_path if it's also temp
else:
    st.info("Please upload a video to begin.") 