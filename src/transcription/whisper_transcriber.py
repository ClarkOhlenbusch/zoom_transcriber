import os
import tempfile
from pathlib import Path
from faster_whisper import WhisperModel

class WhisperTranscriber:
    """
    Class for transcribing audio using Faster Whisper
    """
    
    def __init__(self, model_size="base"):
        """
        Initialize the WhisperTranscriber with the specified model size
        
        Args:
            model_size (str): Size of the Whisper model to use
                             Options: "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"
        """
        # Use CPU as we're working with limited resources
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.model_size = model_size
    
    def transcribe(self, audio_path, language=None, word_timestamps=True):
        """
        Transcribe audio file using Whisper
        
        Args:
            audio_path (str): Path to the audio file
            language (str, optional): Language code (e.g., "en"). If None, language will be detected.
            word_timestamps (bool, optional): Whether to include word-level timestamps. Defaults to True.
        
        Returns:
            dict: Dictionary containing segments with text and timestamps
        """
        # Ensure the audio file exists
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Run transcription
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            word_timestamps=word_timestamps,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Process segments into a structured format
        result = {
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": []
        }
        
        for segment in segments:
            segment_data = {
                "id": len(result["segments"]),
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "words": []
            }
            
            if word_timestamps and segment.words:
                for word in segment.words:
                    segment_data["words"].append({
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability
                    })
            
            result["segments"].append(segment_data)
        
        return result
