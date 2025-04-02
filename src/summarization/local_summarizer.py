import os
import json
from typing import Dict, List, Optional, Any
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import re

class ImprovedLocalTranscriptSummarizer:
    """
    Enhanced class for summarizing meeting transcripts using local extractive summarization
    without requiring external API access, with improvements for more concise, intelligent summaries
    """
    
    def __init__(self):
        """Initialize the ImprovedLocalTranscriptSummarizer"""
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
        Format transcript data into a string for summarization with improved formatting
        
        Args:
            transcript_data (dict): Transcript data from TranscriptionManager
        
        Returns:
            str: Formatted transcript text
        """
        formatted_text = ""
        
        for segment in transcript_data["segments"]:
            speaker = segment.get("speaker", "Unknown Speaker")
            text = segment.get("text", "")
            timestamp = self._format_time(segment.get("start", 0))
            formatted_text += f"[{timestamp}] {speaker}: {text}\n"
        
        return formatted_text
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS format"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
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
    
    def _extract_summary(self, text: str, num_sentences: int = 7) -> str:
        """
        Extract summary from text using TextRank algorithm with improved sentence selection
        
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
        Extract potential action items from text with improved keyword detection
        
        Args:
            text (str): Text to analyze
            
        Returns:
            list: List of potential action items
        """
        action_items = []
        sentences = sent_tokenize(text)
        
        # Enhanced keywords that might indicate action items
        action_keywords = [
            'need to', 'should', 'will', 'going to', 'must', 'have to', 
            'task', 'action', 'follow up', 'follow-up', 'assign', 'responsible',
            'take care of', 'handle', 'complete', 'finish', 'implement', 'deliver',
            'due by', 'deadline', 'by tomorrow', 'by next', 'by monday', 'by tuesday',
            'by wednesday', 'by thursday', 'by friday', 'assigned to'
        ]
        
        # Improved pattern matching for action items
        for sentence in sentences:
            lower_sentence = sentence.lower()
            
            # Check if sentence contains action keywords
            if any(keyword in lower_sentence for keyword in action_keywords):
                # Look for potential assignees (names followed by verbs)
                assignee_match = re.search(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)(?:\s+will|\s+should|\s+needs to|\s+is going to)', sentence)
                
                # If we found a potential assignee, highlight it
                if assignee_match:
                    name = assignee_match.group(1)
                    # Highlight the assignee in the action item
                    highlighted = sentence.replace(name, f"**{name}**")
                    action_items.append(highlighted)
                else:
                    action_items.append(sentence)
        
        return action_items
    
    def _extract_decisions(self, text: str) -> List[str]:
        """
        Extract potential decisions from text with improved keyword detection
        
        Args:
            text (str): Text to analyze
            
        Returns:
            list: List of potential decisions
        """
        decisions = []
        sentences = sent_tokenize(text)
        
        # Enhanced keywords that might indicate decisions
        decision_keywords = [
            'decide', 'decided', 'agreed', 'agreement', 'conclusion', 
            'resolved', 'approved', 'confirmed', 'finalized', 'consensus',
            'settled on', 'determined', 'concluded', 'established', 
            'voted', 'selected', 'chosen', 'opted for', 'went with',
            'moving forward with', 'proceeding with', 'green light'
        ]
        
        for sentence in sentences:
            lower_sentence = sentence.lower()
            
            # Check if sentence contains decision keywords
            if any(keyword in lower_sentence for keyword in decision_keywords):
                decisions.append(sentence)
        
        return decisions
    
    def _create_timeline(self, transcript_data: Dict[str, Any], num_points: int = 5) -> List[str]:
        """
        Create a timeline of main discussion points with improved selection logic
        
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
            return [f"[{self._format_time(segment.get('start', 0))}] {segment.get('text', '')}" for segment in segments]
        
        # Improved approach: identify key segments based on content significance
        # Look for segments that contain topic transition indicators
        topic_indicators = [
            'moving on', 'next topic', 'next item', 'let\'s discuss', 'turning to',
            'shifting to', 'regarding', 'about the', 'let\'s talk about', 'now for',
            'to address', 'let me introduce', 'starting with', 'beginning with'
        ]
        
        # Identify potential topic transition points
        topic_transitions = []
        for i, segment in enumerate(segments):
            text = segment.get("text", "").lower()
            if any(indicator in text for indicator in topic_indicators):
                topic_transitions.append((i, segment))
        
        # If we found enough transition points, use them
        if len(topic_transitions) >= num_points:
            # Select evenly distributed transition points
            step = len(topic_transitions) // num_points
            selected_transitions = [topic_transitions[i * step] for i in range(num_points)]
            
            # Create timeline points from selected transitions
            timeline_points = []
            for _, segment in selected_transitions:
                start_time = self._format_time(segment.get("start", 0))
                timeline_points.append(f"[{start_time}] {segment.get('text', '')}")
            
            return timeline_points
        
        # Fallback: divide transcript into equal parts and take a representative segment from each
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
    
    def _post_process_summary(self, summary: str) -> str:
        """
        Post-process the summary to make it more concise and readable
        
        Args:
            summary (str): Raw summary text
            
        Returns:
            str: Processed summary
        """
        # Remove redundant speaker labels if they appear in the summary
        summary = re.sub(r'(Unknown Speaker|Speaker \d+):\s*', '', summary)
        
        # Remove timestamp patterns if they appear in the summary
        summary = re.sub(r'\[\d+:\d+\]\s*', '', summary)
        
        # Consolidate multiple spaces
        summary = re.sub(r'\s+', ' ', summary)
        
        # Split into sentences
        sentences = sent_tokenize(summary)
        
        # Remove very short sentences (likely fragments)
        sentences = [s for s in sentences if len(s.split()) > 3]
        
        # Rejoin sentences
        processed_summary = ' '.join(sentences)
        
        return processed_summary
    
    def summarize(self, transcript_data: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Summarize a meeting transcript with improved techniques for more concise, intelligent summaries
        
        Args:
            transcript_data (dict): Transcript data from TranscriptionManager
            output_dir (str, optional): Directory to save output files
        
        Returns:
            dict: Dictionary containing summary, action items, and key points
        """
        # Format transcript for summarization
        transcript_text = self._format_transcript_for_summarization(transcript_data)
        
        # Generate improved summary
        raw_summary = self._extract_summary(transcript_text, num_sentences=10)
        summary = self._post_process_summary(raw_summary)
        
        # Extract action items with improved detection
        action_items = self._extract_action_items(transcript_text)
        action_items_text = "\n".join([f"- {item}" for item in action_items]) if action_items else "No clear action items identified."
        
        # Extract decisions with improved detection
        decisions = self._extract_decisions(transcript_text)
        decisions_text = "\n".join([f"- {decision}" for decision in decisions]) if decisions else "No clear decisions identified."
        
        # Create timeline with improved selection
        timeline = self._create_timeline(transcript_data)
        timeline_text = "\n".join([f"- {point}" for point in timeline]) if timeline else "Timeline could not be generated."
        
        # Create a more structured, concise summary format
        if not summary.strip():
            summary = "No clear summary could be generated from the transcript."
        else:
            # Add a title based on content analysis
            title = self._generate_title(transcript_text)
            summary = f"## {title}\n\n{summary}"
        
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
    
    def _generate_title(self, transcript_text: str) -> str:
        """
        Generate a title for the meeting based on content analysis
        
        Args:
            transcript_text (str): Transcript text
            
        Returns:
            str: Generated title
        """
        # Look for common meeting title indicators
        title_patterns = [
            (r'meeting about ([\w\s]+)', r'Meeting: \1'),
            (r'discussion (?:about|on) ([\w\s]+)', r'\1 Discussion'),
            (r'today we(?:\'re| are) (?:going to |)(?:talk|discuss) about ([\w\s]+)', r'\1 Meeting'),
            (r'welcome to (?:the |)([\w\s]+) meeting', r'\1 Meeting'),
            (r'this is (?:the |)([\w\s]+) (?:meeting|call|discussion)', r'\1 Meeting')
        ]
        
        lower_text = transcript_text.lower()
        
        for pattern, replacement in title_patterns:
            match = re.search(pattern, lower_text)
            if match:
                return re.sub(pattern, replacement, match.group(0)).title()
        
        # Default title if no pattern matches
        return "Meeting Summary"
    
    def _save_summary(self, summary_sections: Dict[str, str], transcript_data: Dict[str, Any], output_dir: str) -> None:
        """
        Save summary to files with improved formatting
        
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
        
        # Save text summary with improved formatting
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
