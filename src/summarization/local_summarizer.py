import os
import json
from typing import Dict, List, Optional, Any
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx

class LocalTranscriptSummarizer:
    """
    Class for summarizing meeting transcripts using local extractive summarization
    without requiring external API access
    """
    
    def __init__(self):
        """Initialize the LocalTranscriptSummarizer"""
        # Download required NLTK resources if not already present
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        
        self.stop_words = set(stopwords.words('english'))
    
    def _format_transcript_for_summarization(self, transcript_data: Dict[str, Any]) -> str:
        """
        Format transcript data into a string for summarization
        
        Args:
            transcript_data (dict): Transcript data from TranscriptionManager
        
        Returns:
            str: Formatted transcript text
        """
        formatted_text = ""
        
        for segment in transcript_data["segments"]:
            speaker = segment.get("speaker", "Unknown Speaker")
            text = segment.get("text", "")
            formatted_text += f"{speaker}: {text}\n"
        
        return formatted_text
    
    def _sentence_similarity(self, sent1: List[str], sent2: List[str]) -> float:
        """
        Calculate similarity between two sentences
        
        Args:
            sent1 (list): First sentence tokens
            sent2 (list): Second sentence tokens
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Create vectors
        all_words = list(set(sent1 + sent2))
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)
        
        # Build the vectors
        for w in sent1:
            if w in all_words:
                vector1[all_words.index(w)] += 1
        
        for w in sent2:
            if w in all_words:
                vector2[all_words.index(w)] += 1
        
        # Handle zero vectors
        if sum(vector1) == 0 or sum(vector2) == 0:
            return 0.0
        
        # Calculate cosine similarity
        return 1 - cosine_distance(vector1, vector2)
    
    def _build_similarity_matrix(self, sentences: List[str]) -> np.ndarray:
        """
        Build similarity matrix for all sentences
        
        Args:
            sentences (list): List of sentences
            
        Returns:
            numpy.ndarray: Similarity matrix
        """
        # Create an empty similarity matrix
        similarity_matrix = np.zeros((len(sentences), len(sentences)))
        
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i == j:
                    continue
                
                # Tokenize and filter stop words
                sent1 = [word.lower() for word in nltk.word_tokenize(sentences[i]) if word.lower() not in self.stop_words]
                sent2 = [word.lower() for word in nltk.word_tokenize(sentences[j]) if word.lower() not in self.stop_words]
                
                similarity_matrix[i][j] = self._sentence_similarity(sent1, sent2)
                
        return similarity_matrix
    
    def _extract_summary(self, text: str, num_sentences: int = 5) -> str:
        """
        Extract summary from text using TextRank algorithm
        
        Args:
            text (str): Text to summarize
            num_sentences (int): Number of sentences to include in summary
            
        Returns:
            str: Extractive summary
        """
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # Limit number of sentences if requested number is greater than available
        num_sentences = min(num_sentences, len(sentences))
        
        # Handle very short texts
        if len(sentences) <= num_sentences:
            return text
        
        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(sentences)
        
        # Create graph and apply PageRank
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)
        
        # Sort sentences by score and select top ones
        ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
        
        # Get top sentences and sort by original position
        top_sentences = sorted(ranked_sentences[:num_sentences], key=lambda x: x[1])
        
        # Combine sentences into summary
        summary = ' '.join([s for _, _, s in top_sentences])
        
        return summary
    
    def _extract_action_items(self, text: str) -> List[str]:
        """
        Extract potential action items from text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            list: List of potential action items
        """
        action_items = []
        sentences = sent_tokenize(text)
        
        # Keywords that might indicate action items
        action_keywords = ['need to', 'should', 'will', 'going to', 'must', 'have to', 'task', 'action', 'follow up', 'follow-up', 'assign']
        
        for sentence in sentences:
            lower_sentence = sentence.lower()
            
            # Check if sentence contains action keywords
            if any(keyword in lower_sentence for keyword in action_keywords):
                action_items.append(sentence)
        
        return action_items
    
    def _extract_decisions(self, text: str) -> List[str]:
        """
        Extract potential decisions from text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            list: List of potential decisions
        """
        decisions = []
        sentences = sent_tokenize(text)
        
        # Keywords that might indicate decisions
        decision_keywords = ['decide', 'decided', 'agreed', 'agreement', 'conclusion', 'resolved', 'approved', 'confirmed', 'finalized']
        
        for sentence in sentences:
            lower_sentence = sentence.lower()
            
            # Check if sentence contains decision keywords
            if any(keyword in lower_sentence for keyword in decision_keywords):
                decisions.append(sentence)
        
        return decisions
    
    def _create_timeline(self, transcript_data: Dict[str, Any], num_points: int = 5) -> List[str]:
        """
        Create a timeline of main discussion points
        
        Args:
            transcript_data (dict): Transcript data
            num_points (int): Number of timeline points to extract
            
        Returns:
            list: Timeline points
        """
        # Get all segments
        segments = transcript_data.get("segments", [])
        
        # If there are very few segments, return all of them
        if len(segments) <= num_points:
            return [segment.get("text", "") for segment in segments]
        
        # Divide transcript into equal parts and take a representative segment from each
        timeline_points = []
        chunk_size = len(segments) // num_points
        
        for i in range(num_points):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_points - 1 else len(segments)
            
            # Get the chunk of segments
            chunk = segments[start_idx:end_idx]
            
            # Find the longest segment in the chunk (likely most informative)
            longest_segment = max(chunk, key=lambda x: len(x.get("text", "")))
            
            # Add to timeline with timestamp
            start_time = self._format_time(longest_segment.get("start", 0))
            timeline_points.append(f"[{start_time}] {longest_segment.get('text', '')}")
        
        return timeline_points
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def summarize(self, transcript_data: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Summarize a meeting transcript
        
        Args:
            transcript_data (dict): Transcript data from TranscriptionManager
            output_dir (str, optional): Directory to save output files
        
        Returns:
            dict: Dictionary containing summary, action items, and key points
        """
        # Format transcript for summarization
        transcript_text = self._format_transcript_for_summarization(transcript_data)
        
        # Generate summary
        summary = self._extract_summary(transcript_text, num_sentences=10)
        
        # Extract action items
        action_items = self._extract_action_items(transcript_text)
        action_items_text = "\n".join([f"- {item}" for item in action_items]) if action_items else "No clear action items identified."
        
        # Extract decisions
        decisions = self._extract_decisions(transcript_text)
        decisions_text = "\n".join([f"- {decision}" for decision in decisions]) if decisions else "No clear decisions identified."
        
        # Create timeline
        timeline = self._create_timeline(transcript_data)
        timeline_text = "\n".join([f"- {point}" for point in timeline]) if timeline else "Timeline could not be generated."
        
        # Prepare results
        results = {
            "summary": summary,
            "action_items": action_items_text,
            "decisions": decisions_text,
            "timeline": timeline_text
        }
        
        # Save results if output directory is provided
        if output_dir:
            self._save_summary(results, transcript_data, output_dir)
        
        return results
    
    def _save_summary(self, summary_sections: Dict[str, str], transcript_data: Dict[str, Any], output_dir: str) -> None:
        """
        Save summary to files
        
        Args:
            summary_sections (dict): Dictionary with summary sections
            transcript_data (dict): Original transcript data
            output_dir (str): Directory to save output files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine base filename from transcript data
        if "filename" in transcript_data:
            base_name = os.path.splitext(os.path.basename(transcript_data["filename"]))[0]
        else:
            base_name = "meeting"
        
        # Save JSON summary
        json_path = os.path.join(output_dir, f"{base_name}_summary.json")
        with open(json_path, 'w') as f:
            json.dump(summary_sections, f, indent=2)
        
        # Save text summary
        text_path = os.path.join(output_dir, f"{base_name}_summary.txt")
        with open(text_path, 'w') as f:
            f.write("# Meeting Summary\n\n")
            f.write(summary_sections["summary"])
            f.write("\n\n# Action Items\n\n")
            f.write(summary_sections["action_items"])
            f.write("\n\n# Decisions\n\n")
            f.write(summary_sections["decisions"])
            f.write("\n\n# Timeline\n\n")
            f.write(summary_sections["timeline"])
        
        print(f"Summary saved to {json_path} and {text_path}")
