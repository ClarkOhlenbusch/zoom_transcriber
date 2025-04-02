import os
import subprocess
from pathlib import Path

def extract_audio(video_path, output_path=None, format="wav"):
    """
    Extract audio from video file using FFmpeg
    
    Args:
        video_path (str): Path to the video file
        output_path (str, optional): Path to save the extracted audio. If None, 
                                    will use the same name as video with .wav extension
        format (str, optional): Audio format to extract. Defaults to "wav".
    
    Returns:
        str: Path to the extracted audio file
    """
    video_path = Path(video_path)
    
    if output_path is None:
        output_path = video_path.with_suffix(f".{format}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Run FFmpeg command to extract audio
    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # PCM 16-bit little-endian format
        "-ar", "16000",  # 16kHz sample rate (good for speech recognition)
        "-ac", "1",  # Mono channel
        "-y",  # Overwrite output file if it exists
        str(output_path)
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True)
        return str(output_path)
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e}")
        print(f"FFmpeg stderr: {e.stderr.decode()}")
        raise
