import os
import sys
import tempfile
import streamlit as st
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transcription.transcription_manager import TranscriptionManager
from summarization.transcript_summarizer import TranscriptSummarizer

def main():
    """
    Streamlit web interface for the Zoom Transcriber and Summarizer
    """
    st.set_page_config(
        page_title="Zoom Transcriber and Summarizer",
        page_icon="🎙️",
        layout="wide"
    )
    
    st.title("Zoom Transcriber and Summarizer")
    st.markdown("""
    This application transcribes Zoom recordings with speaker identification and generates 
    comprehensive meeting summaries, action items, and key points.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Transcription settings
        st.subheader("Transcription Settings")
        whisper_model = st.selectbox(
            "Whisper Model Size",
            options=["tiny", "base", "small", "medium"],
            index=1,
            help="Larger models are more accurate but require more resources"
        )
        
        max_speakers = st.number_input(
            "Maximum Number of Speakers",
            min_value=1,
            max_value=10,
            value=2,
            help="Maximum number of speakers to identify"
        )
        
        language = st.text_input(
            "Language Code (optional)",
            value="",
            help="ISO language code (e.g., 'en' for English). Leave empty for auto-detection."
        )
        
        # Summarization settings
        st.subheader("Summarization Settings")
        huggingface_model = st.selectbox(
            "Summarization Model",
            options=["google/flan-t5-base", "google/flan-t5-large", "facebook/bart-large-cnn"],
            index=0,
            help="Model to use for summarization"
        )
    
    # Main content area
    uploaded_file = st.file_uploader(
        "Upload a Zoom recording (MP4 or M4A)",
        type=["mp4", "m4a", "wav"],
        help="Select a Zoom recording file to transcribe and summarize"
    )
    
    if uploaded_file is not None:
        # Create a temporary file to store the uploaded content
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Create a temporary directory for output
        output_dir = tempfile.mkdtemp()
        
        # Process button
        if st.button("Transcribe and Summarize"):
            # Display progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Transcribe the recording
                status_text.text("Step 1/2: Transcribing audio with speaker identification...")
                
                transcription_manager = TranscriptionManager(
                    whisper_model_size=whisper_model,
                    max_speakers=max_speakers
                )
                
                transcript_data = transcription_manager.process_recording(
                    tmp_path,
                    output_dir=output_dir,
                    language=language if language else None
                )
                
                progress_bar.progress(50)
                
                # Step 2: Summarize the transcript
                status_text.text("Step 2/2: Generating meeting summary...")
                
                summarizer = TranscriptSummarizer(
                    huggingface_model=huggingface_model
                )
                
                summary = summarizer.summarize(
                    transcript_data,
                    output_dir=output_dir
                )
                
                progress_bar.progress(100)
                status_text.text("Processing complete!")
                
                # Display results
                st.success("Transcription and summarization completed successfully!")
                
                # Create tabs for different outputs
                tab1, tab2, tab3 = st.tabs(["Summary", "Transcript", "Download Files"])
                
                with tab1:
                    st.header("Meeting Summary")
                    st.write(summary["summary"])
                    
                    st.header("Action Items")
                    st.write(summary["action_items"])
                    
                    st.header("Decisions")
                    st.write(summary["decisions"])
                    
                    st.header("Timeline")
                    st.write(summary["timeline"])
                
                with tab2:
                    st.header("Full Transcript")
                    
                    # Display transcript with speaker identification
                    for segment in transcript_data["segments"]:
                        speaker = segment.get("speaker", "Unknown Speaker")
                        text = segment.get("text", "")
                        start_time = format_time(segment.get("start", 0))
                        
                        st.markdown(f"**[{start_time}] {speaker}:** {text}")
                
                with tab3:
                    st.header("Download Files")
                    
                    # Get file paths
                    base_name = Path(uploaded_file.name).stem
                    transcript_json_path = os.path.join(output_dir, f"{base_name}_transcript.json")
                    transcript_text_path = os.path.join(output_dir, f"{base_name}_transcript.txt")
                    summary_json_path = os.path.join(output_dir, f"{base_name}_summary.json")
                    summary_text_path = os.path.join(output_dir, f"{base_name}_summary.txt")
                    
                    # Read file contents
                    with open(transcript_json_path, "rb") as f:
                        transcript_json_data = f.read()
                    
                    with open(transcript_text_path, "rb") as f:
                        transcript_text_data = f.read()
                    
                    with open(summary_json_path, "rb") as f:
                        summary_json_data = f.read()
                    
                    with open(summary_text_path, "rb") as f:
                        summary_text_data = f.read()
                    
                    # Create download buttons
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.download_button(
                            label="Download Transcript (JSON)",
                            data=transcript_json_data,
                            file_name=f"{base_name}_transcript.json",
                            mime="application/json"
                        )
                        
                        st.download_button(
                            label="Download Transcript (Text)",
                            data=transcript_text_data,
                            file_name=f"{base_name}_transcript.txt",
                            mime="text/plain"
                        )
                    
                    with col2:
                        st.download_button(
                            label="Download Summary (JSON)",
                            data=summary_json_data,
                            file_name=f"{base_name}_summary.json",
                            mime="application/json"
                        )
                        
                        st.download_button(
                            label="Download Summary (Text)",
                            data=summary_text_data,
                            file_name=f"{base_name}_summary.txt",
                            mime="text/plain"
                        )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
            
            finally:
                # Clean up temporary files
                try:
                    os.unlink(tmp_path)
                except:
                    pass

def format_time(seconds):
    """Format time in seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

if __name__ == "__main__":
    main()
