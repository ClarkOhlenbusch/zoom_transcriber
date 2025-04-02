import numpy as np
from scipy.io import wavfile
import librosa
import os
from pathlib import Path

class SimpleDiarizer:
    """
    A simple speaker diarization class that uses energy-based segmentation
    and basic audio features to differentiate between speakers.
    
    This is a lightweight alternative to pyannote.audio when disk space is limited.
    """
    
    def __init__(self, min_silence_duration=0.5, energy_threshold=0.05, 
                 min_speech_duration=1.0, max_speakers=2):
        """
        Initialize the SimpleDiarizer
        
        Args:
            min_silence_duration (float): Minimum duration of silence in seconds
            energy_threshold (float): Threshold for silence detection (0-1)
            min_speech_duration (float): Minimum duration of speech segment in seconds
            max_speakers (int): Maximum number of speakers to identify
        """
        self.min_silence_duration = min_silence_duration
        self.energy_threshold = energy_threshold
        self.min_speech_duration = min_speech_duration
        self.max_speakers = max_speakers
    
    def _load_audio(self, audio_path):
        """Load audio file using librosa"""
        try:
            audio, sample_rate = librosa.load(audio_path, sr=None)
            return audio, sample_rate
        except Exception as e:
            print(f"Error loading audio: {e}")
            raise
    
    def _detect_speech_segments(self, audio, sample_rate):
        """
        Detect speech segments based on energy levels
        
        Returns:
            list: List of tuples (start_time, end_time) for each speech segment
        """
        # Calculate energy
        energy = np.abs(audio)
        energy_mean = np.mean(energy)
        
        # Normalize energy to 0-1 range
        energy_norm = energy / np.max(energy)
        
        # Apply threshold to detect speech
        is_speech = energy_norm > (self.energy_threshold * energy_mean)
        
        # Convert to samples
        min_silence_samples = int(self.min_silence_duration * sample_rate)
        min_speech_samples = int(self.min_speech_duration * sample_rate)
        
        # Find speech segments
        speech_segments = []
        in_speech = False
        speech_start = 0
        
        for i in range(len(is_speech)):
            if not in_speech and is_speech[i]:
                # Start of speech
                in_speech = True
                speech_start = i
            elif in_speech and not is_speech[i]:
                # End of speech
                if i - speech_start >= min_speech_samples:
                    speech_segments.append((speech_start / sample_rate, i / sample_rate))
                in_speech = False
        
        # Handle case where audio ends during speech
        if in_speech and len(audio) - speech_start >= min_speech_samples:
            speech_segments.append((speech_start / sample_rate, len(audio) / sample_rate))
        
        return speech_segments
    
    def _extract_features(self, audio, sample_rate, segments):
        """
        Extract audio features for each segment
        
        Returns:
            list: List of feature vectors for each segment
        """
        features = []
        
        for start_time, end_time in segments:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            segment_audio = audio[start_sample:end_sample]
            
            # Extract simple features
            mfccs = librosa.feature.mfcc(y=segment_audio, sr=sample_rate, n_mfcc=13)
            mfcc_mean = np.mean(mfccs, axis=1)
            
            # Add pitch information
            pitch, _ = librosa.piptrack(y=segment_audio, sr=sample_rate)
            pitch_mean = np.mean(pitch, axis=1)
            
            # Combine features
            segment_features = np.concatenate([mfcc_mean, [np.mean(pitch_mean)]])
            features.append(segment_features)
        
        return features
    
    def _cluster_speakers(self, features, max_speakers=None):
        """
        Cluster segments into speakers using K-means
        
        Returns:
            list: Speaker labels for each segment
        """
        if max_speakers is None:
            max_speakers = self.max_speakers
        
        from sklearn.cluster import KMeans
        
        # Determine number of speakers (clusters)
        n_clusters = min(max_speakers, len(features))
        
        if n_clusters <= 1:
            return [0] * len(features)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(features)
        
        return labels
    
    def diarize(self, audio_path, max_speakers=None):
        """
        Perform speaker diarization on audio file
        
        Args:
            audio_path (str): Path to audio file
            max_speakers (int, optional): Maximum number of speakers to identify
        
        Returns:
            dict: Dictionary with diarization results
        """
        # Load audio
        audio, sample_rate = self._load_audio(audio_path)
        
        # Detect speech segments
        segments = self._detect_speech_segments(audio, sample_rate)
        
        # Extract features
        features = self._extract_features(audio, sample_rate, segments)
        
        # Cluster speakers
        speaker_labels = self._cluster_speakers(features, max_speakers)
        
        # Format results
        results = []
        for i, ((start_time, end_time), speaker) in enumerate(zip(segments, speaker_labels)):
            results.append({
                "segment_id": i,
                "start": start_time,
                "end": end_time,
                "speaker": f"SPEAKER_{speaker}"
            })
        
        return {
            "segments": results,
            "num_speakers": len(set(speaker_labels))
        }
