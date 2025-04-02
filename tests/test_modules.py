import os
import sys
import unittest
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from src.transcription.audio_extractor import extract_audio
from src.transcription.whisper_transcriber import WhisperTranscriber
from src.transcription.simple_diarizer import SimpleDiarizer
from src.transcription.transcription_manager import TranscriptionManager
from src.summarization.transcript_summarizer import TranscriptSummarizer

class TestTranscriptionModule(unittest.TestCase):
    """Test cases for the transcription module"""
    
    def test_audio_extractor(self):
        """Test audio extraction functionality"""
        # This is a mock test since we don't have actual audio files
        # In a real scenario, we would test with actual files
        self.assertTrue(callable(extract_audio))
    
    def test_whisper_transcriber_init(self):
        """Test WhisperTranscriber initialization"""
        try:
            transcriber = WhisperTranscriber(model_size="tiny")
            self.assertIsNotNone(transcriber)
        except Exception as e:
            self.fail(f"WhisperTranscriber initialization failed: {e}")
    
    def test_simple_diarizer_init(self):
        """Test SimpleDiarizer initialization"""
        try:
            diarizer = SimpleDiarizer()
            self.assertIsNotNone(diarizer)
        except Exception as e:
            self.fail(f"SimpleDiarizer initialization failed: {e}")
    
    def test_transcription_manager_init(self):
        """Test TranscriptionManager initialization"""
        try:
            manager = TranscriptionManager(whisper_model_size="tiny")
            self.assertIsNotNone(manager)
        except Exception as e:
            self.fail(f"TranscriptionManager initialization failed: {e}")

class TestSummarizationModule(unittest.TestCase):
    """Test cases for the summarization module"""
    
    def test_transcript_summarizer_init(self):
        """Test TranscriptSummarizer initialization"""
        try:
            summarizer = TranscriptSummarizer(huggingface_model="google/flan-t5-base")
            self.assertIsNotNone(summarizer)
        except Exception as e:
            self.fail(f"TranscriptSummarizer initialization failed: {e}")
    
    def test_format_transcript(self):
        """Test transcript formatting for summarization"""
        summarizer = TranscriptSummarizer()
        
        # Create a mock transcript
        mock_transcript = {
            "segments": [
                {"speaker": "SPEAKER_0", "text": "Hello, this is a test."},
                {"speaker": "SPEAKER_1", "text": "Yes, it is a test."}
            ]
        }
        
        formatted = summarizer._format_transcript_for_summarization(mock_transcript)
        self.assertIsInstance(formatted, str)
        self.assertIn("SPEAKER_0", formatted)
        self.assertIn("SPEAKER_1", formatted)
        self.assertIn("Hello, this is a test.", formatted)
        self.assertIn("Yes, it is a test.", formatted)
    
    def test_split_transcript(self):
        """Test transcript splitting functionality"""
        summarizer = TranscriptSummarizer()
        
        # Create a long mock transcript
        long_text = "Test " * 1000
        
        chunks = summarizer._split_transcript(long_text)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 0)

if __name__ == "__main__":
    unittest.main()
