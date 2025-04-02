import os
import re
from pathlib import Path
from .audio_extractor import extract_audio
from .whisper_transcriber import WhisperTranscriber
from .simple_diarizer import SimpleDiarizer

class TranscriptionManager:
    """
    Manager class that coordinates audio extraction, transcription, and speaker diarization
    """
    
    def __init__(self, whisper_model_size="base", max_speakers=2):
        """
        Initialize the TranscriptionManager
        
        Args:
            whisper_model_size (str): Size of the Whisper model to use
            max_speakers (int): Maximum number of speakers to identify
        """
        self.whisper_model_size = whisper_model_size
        self.max_speakers = max_speakers
        self.transcriber = None
        self.diarizer = None
        
    def _initialize_components(self):
        """Initialize transcription and diarization components if not already initialized"""
        if self.transcriber is None:
            self.transcriber = WhisperTranscriber(model_size=self.whisper_model_size)
        
        if self.diarizer is None:
            self.diarizer = SimpleDiarizer(max_speakers=self.max_speakers)
    
    def _sanitize_filename(self, filename):
        """
        Sanitize filename to remove or replace characters that might cause issues on different platforms
        
        Args:
            filename (str): Original filename
            
        Returns:
            str: Sanitized filename
        """
        # Replace problematic characters with underscores
        # Windows doesn't allow: < > : " / \ | ? *
        # Also handle other potentially problematic characters like # and spaces
        sanitized = re.sub(r'[<>:"/\\|?*#]', '_', filename)
        return sanitized
    
    def process_recording(self, recording_path, output_dir=None, language=None):
        """
        Process a Zoom recording: extract audio, transcribe, and identify speakers
        
        Args:
            recording_path (str): Path to the Zoom recording (MP4 or M4A)
            output_dir (str, optional): Directory to save output files
            language (str, optional): Language code for transcription
        
        Returns:
            dict: Dictionary containing the processed transcript with speaker identification
        """
        # Initialize components
        self._initialize_components()
        
        # Create output directory if needed
        if output_dir is None:
            output_dir = os.path.dirname(os.path.abspath(recording_path))
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract audio if needed
        recording_path = Path(recording_path)
        if recording_path.suffix.lower() in ['.mp4', '.m4a']:
            # Sanitize the filename for the audio file
            sanitized_stem = self._sanitize_filename(recording_path.stem)
            audio_path = os.path.join(output_dir, f"{sanitized_stem}.wav")
            extract_audio(str(recording_path), audio_path)
        else:
            audio_path = str(recording_path)
        
        # Transcribe audio
        print(f"Transcribing audio: {audio_path}")
        transcription = self.transcriber.transcribe(audio_path, language=language)
        
        # Perform speaker diarization
        print(f"Performing speaker diarization")
        diarization = self.diarizer.diarize(audio_path, max_speakers=self.max_speakers)
        
        # Merge transcription with speaker identification
        result = self._merge_transcription_with_diarization(transcription, diarization)
        
        # Save results with sanitized filename
        sanitized_stem = self._sanitize_filename(recording_path.stem)
        output_path = os.path.join(output_dir, f"{sanitized_stem}_transcript.json")
        self._save_results(result, output_path)
        
        text_output_path = os.path.join(output_dir, f"{sanitized_stem}_transcript.txt")
        self._save_text_transcript(result, text_output_path)
        
        return result
    
    def _merge_transcription_with_diarization(self, transcription, diarization):
        """
        Merge transcription with speaker diarization results
        
        Args:
            transcription (dict): Transcription results from WhisperTranscriber
            diarization (dict): Diarization results from SimpleDiarizer
        
        Returns:
            dict: Merged results with speaker identification
        """
        # Create a mapping of time ranges to speaker IDs
        speaker_ranges = {}
        for segment in diarization["segments"]:
            speaker_ranges[(segment["start"], segment["end"])] = segment["speaker"]
        
        # Assign speakers to transcription segments based on overlap
        for segment in transcription["segments"]:
            segment_start = segment["start"]
            segment_end = segment["end"]
            
            # Find the speaker with the most overlap
            max_overlap = 0
            assigned_speaker = "UNKNOWN_SPEAKER"
            
            for (diar_start, diar_end), speaker in speaker_ranges.items():
                # Calculate overlap
                overlap_start = max(segment_start, diar_start)
                overlap_end = min(segment_end, diar_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > max_overlap:
                    max_overlap = overlap
                    assigned_speaker = speaker
            
            # Assign speaker to segment
            segment["speaker"] = assigned_speaker
        
        # Add diarization info to result
        result = transcription.copy()
        result["num_speakers"] = diarization["num_speakers"]
        
        return result
    
    def _save_results(self, result, output_path):
        """Save results to JSON file"""
        import json
        try:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Successfully saved JSON to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving JSON to {output_path}: {str(e)}")
            raise
    
    def _save_text_transcript(self, result, output_path):
        """Save transcript in human-readable text format"""
        try:
            with open(output_path, 'w') as f:
                f.write(f"Transcript\n")
                f.write(f"==========\n\n")
                
                for segment in result["segments"]:
                    speaker = segment["speaker"]
                    text = segment["text"]
                    start_time = self._format_time(segment["start"])
                    end_time = self._format_time(segment["end"])
                    
                    f.write(f"[{start_time} - {end_time}] {speaker}: {text}\n\n")
            
            print(f"Successfully saved text transcript to: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error saving text transcript to {output_path}: {str(e)}")
            raise
    
    def _format_time(self, seconds):
        """Format time in seconds to HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
