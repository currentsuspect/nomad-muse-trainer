"""Musical Evaluation Metrics for Multi-Expert Nomad Muse Trainer.

Provides comprehensive evaluation beyond perplexity:
- Drum groove stability
- Chord progression plausibility  
- Chord-tone ratio for melodies
- Melodic contour statistics
- Memorization detection
"""

import json
import logging
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any
import statistics

from tokenizers_drums import MuseDrumsTokenizer
from tokenizers_harmony import MuseHarmonyTokenizer
from tokenizers_melody import MuseMelodyTokenizer
from nomad_format import NomadFormatBuilder

logger = logging.getLogger(__name__)


class GrooveStabilityAnalyzer:
    """Analyzes drum groove stability and consistency."""
    
    def __init__(self, max_bars: int = 4):
        """Initialize analyzer.
        
        Args:
            max_bars: Maximum bars to analyze for patterns
        """
        self.max_bars = max_bars
    
    def analyze_groove_stability(self, nomad_data: List[Dict]) -> Dict[str, float]:
        """Analyze groove stability in drum tracks.
        
        Args:
            nomad_data: List of Nomad Symbolic Format dictionaries
            
        Returns:
            Groove stability metrics
        """
        groove_patterns = []
        
        # Extract drum patterns
        for data in nomad_data:
            drum_track = self._find_drum_track(data)
            if drum_track:
                pattern = self._extract_drum_pattern(drum_track)
                if pattern:
                    groove_patterns.append(pattern)
        
        if len(groove_patterns) < 2:
            return {"error": "Insufficient drum patterns for analysis"}
        
        # Calculate stability metrics
        consistency_scores = []
        for i in range(len(groove_patterns)):
            for j in range(i + 1, len(groove_patterns)):
                consistency = self._calculate_pattern_consistency(
                    groove_patterns[i], groove_patterns[j]
                )
                consistency_scores.append(consistency)
        
        # Repetition vs variation analysis
        repetition_scores = self._analyze_repetition_vs_variation(groove_patterns)
        
        # Hit density analysis
        density_stats = self._analyze_hit_density(groove_patterns)
        
        return {
            "groove_consistency": np.mean(consistency_scores),
            "groove_consistency_std": np.std(consistency_scores),
            "repetition_score": repetition_scores["repetition"],
            "variation_score": repetition_scores["variation"],
            "hit_density_mean": density_stats["mean"],
            "hit_density_std": density_stats["std"],
            "num_patterns_analyzed": len(groove_patterns)
        }
    
    def _find_drum_track(self, nomad_data: Dict) -> Optional[Dict]:
        """Find drum track in Nomad data."""
        for track in nomad_data.get("tracks", []):
            if track["classification"]["type"] == "drums":
                return track
        return None
    
    def _extract_drum_pattern(self, drum_track: Dict) -> Optional[List[Dict]]:
        """Extract drum pattern from track."""
        events = [e for e in drum_track["events"] if e["type"] == "note_on"]
        if not events:
            return None
        
        # Group by bars
        patterns = []
        current_bar = 0
        current_bar_events = []
        
        for event in events:
            # Estimate bar from time (simplified)
            time_seconds = event["time_seconds"]
            estimated_bar = int(time_seconds // 4.0)  # Assume 4/4 at 120 BPM
            
            if estimated_bar != current_bar:
                if current_bar_events:
                    patterns.append(current_bar_events)
                current_bar = estimated_bar
                current_bar_events = []
            
            current_bar_events.append({
                "pitch": event["pitch"],
                "velocity": event["velocity"],
                "time_in_bar": time_seconds % 4.0
            })
        
        if current_bar_events:
            patterns.append(current_bar_events)
        
        return patterns[:self.max_bars]  # Limit to max_bars
    
    def _calculate_pattern_consistency(self, pattern1: List[Dict], pattern2: List[Dict]) -> float:
        """Calculate consistency between two patterns."""
        if len(pattern1) != len(pattern2):
            return 0.0
        
        total_similarity = 0.0
        for bar1, bar2 in zip(pattern1, pattern2):
            similarity = self._calculate_bar_consistency(bar1, bar2)
            total_similarity += similarity
        
        return total_similarity / len(pattern1)
    
    def _calculate_bar_consistency(self, bar1: List[Dict], bar2: List[Dict]) -> float:
        """Calculate consistency within a single bar."""
        if not bar1 or not bar2:
            return 0.0
        
        # Create timing grids
        grid1 = self._create_timing_grid(bar1)
        grid2 = self._create_timing_grid(bar2)
        
        # Compare grid similarity
        if len(grid1) != len(grid2):
            min_len = min(len(grid1), len(grid2))
            grid1 = grid1[:min_len]
            grid2 = grid2[:min_len]
        
        # Calculate pitch/velocity similarity
        similarities = []
        for hit1, hit2 in zip(grid1, grid2):
            if hit1 is None or hit2 is None:
                similarities.append(0.0)
            else:
                # Similarity based on pitch and velocity difference
                pitch_sim = 1.0 - abs(hit1["pitch"] - hit2["pitch"]) / 50.0  # Normalize
                velocity_sim = 1.0 - abs(hit1["velocity"] - hit2["velocity"]) / 127.0
                similarities.append((pitch_sim + velocity_sim) / 2.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _create_timing_grid(self, hits: List[Dict], grid_resolution: float = 0.25) -> List[Optional[Dict]]:
        """Create timing grid from hits."""
        grid = []
        for time_pos in np.arange(0, 4.0, grid_resolution):  # 4/4 bar
            # Find closest hit
            closest_hit = None
            min_time_diff = float('inf')
            
            for hit in hits:
                time_diff = abs(hit["time_in_bar"] - time_pos)
                if time_diff < min_time_diff and time_diff <= grid_resolution / 2:
                    closest_hit = hit
                    min_time_diff = time_diff
            
            grid.append(closest_hit)
        
        return grid
    
    def _analyze_repetition_vs_variation(self, patterns: List[List[Dict]]) -> Dict[str, float]:
        """Analyze repetition vs variation in patterns."""
        if len(patterns) < 2:
            return {"repetition": 0.0, "variation": 0.0}
        
        # Calculate pattern fingerprints
        fingerprints = []
        for pattern in patterns:
            fingerprint = self._create_pattern_fingerprint(pattern)
            fingerprints.append(fingerprint)
        
        # Count identical patterns
        identical_pairs = 0
        total_pairs = 0
        
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                total_pairs += 1
                if fingerprints[i] == fingerprints[j]:
                    identical_pairs += 1
        
        repetition_score = identical_pairs / total_pairs if total_pairs > 0 else 0.0
        
        # Variation score (inverse of repetition)
        variation_score = 1.0 - repetition_score
        
        return {
            "repetition": repetition_score,
            "variation": variation_score
        }
    
    def _create_pattern_fingerprint(self, pattern: List[Dict]) -> str:
        """Create fingerprint for pattern."""
        # Simplify pattern to key features
        features = []
        for bar in pattern:
            # Sort by time and create simplified representation
            sorted_hits = sorted(bar, key=lambda x: x["time_in_bar"])
            bar_features = []
            for hit in sorted_hits:
                # Simplified: pitch class + timing bucket
                pitch_class = hit["pitch"] % 12
                time_bucket = int(hit["time_in_bar"] * 4)  # 16th note resolution
                bar_features.append(f"{pitch_class}_{time_bucket}")
            features.append("|".join(bar_features))
        
        return "|".join(features)
    
    def _analyze_hit_density(self, patterns: List[List[Dict]]) -> Dict[str, float]:
        """Analyze hit density across patterns."""
        densities = []
        
        for pattern in patterns:
            total_hits = sum(len(bar) for bar in pattern)
            bars = len(pattern)
            density = total_hits / bars if bars > 0 else 0
            densities.append(density)
        
        return {
            "mean": np.mean(densities),
            "std": np.std(densities),
            "min": np.min(densities),
            "max": np.max(densities)
        }


class ChordProgressionAnalyzer:
    """Analyzes chord progression plausibility."""
    
    def __init__(self):
        """Initialize analyzer."""
        # Common chord progressions (in Roman numeral analysis)
        self.common_progressions = [
            [1, 4, 5, 1],  # I-IV-V-I
            [1, 6, 4, 5],  # I-vi-IV-V
            [2, 5, 1],     # ii-V-I
            [1, 7, 4, 1],  # I-vii°-IV-I
            [1, 3, 6, 4, 5, 1],  # I-iii-vi-IV-V-I
        ]
    
    def analyze_chord_progressions(self, nomad_data: List[Dict]) -> Dict[str, Any]:
        """Analyze chord progressions.
        
        Args:
            nomad_data: List of Nomad Symbolic Format dictionaries
            
        Returns:
            Chord progression metrics
        """
        progressions = []
        
        # Extract chord progressions
        for data in nomad_data:
            chords = data.get("chords", [])
            if len(chords) >= 2:
                progression = self._extract_progression(chords)
                if progression:
                    progressions.append(progression)
        
        if not progressions:
            return {"error": "No valid chord progressions found"}
        
        # Analyze progressions
        plausibility_scores = []
        for progression in progressions:
            score = self._evaluate_progression_plausibility(progression)
            plausibility_scores.append(score)
        
        # Analyze chord distribution
        chord_distribution = self._analyze_chord_distribution(progressions)
        
        # Analyze harmonic rhythm
        rhythm_stats = self._analyze_harmonic_rhythm(progressions)
        
        return {
            "progression_plausibility_mean": np.mean(plausibility_scores),
            "progression_plausibility_std": np.std(plausibility_scores),
            "chord_distribution": chord_distribution,
            "harmonic_rhythm_stats": rhythm_stats,
            "num_progressions_analyzed": len(progressions)
        }
    
    def _extract_progression(self, chords: List[Dict]) -> Optional[List[Dict]]:
        """Extract chord progression from chord events."""
        if len(chords) < 2:
            return None
        
        progression = []
        for chord in chords:
            chord_info = chord.get("chord", {})
            if chord_info.get("confidence", 0) > 0.5:  # Only confident chords
                progression.append({
                    "root": chord_info.get("root", "C"),
                    "quality": chord_info.get("quality", "maj"),
                    "time_start": chord.get("time_start", 0.0)
                })
        
        return progression
    
    def _evaluate_progression_plausibility(self, progression: List[Dict]) -> float:
        """Evaluate plausibility of a chord progression."""
        if len(progression) < 2:
            return 0.0
        
        # Convert to numeric representation for comparison
        numeric_progression = []
        for chord in progression:
            root_num = self._note_to_number(chord["root"])
            quality_score = self._quality_to_score(chord["quality"])
            numeric_progression.append(root_num + quality_score)
        
        # Check against common progressions (simplified)
        best_match = 0.0
        for common_prog in self.common_progressions:
            if len(numeric_progression) >= len(common_prog):
                # Compare subsequences
                for i in range(len(numeric_progression) - len(common_prog) + 1):
                    subsequence = numeric_progression[i:i + len(common_prog)]
                    similarity = self._calculate_sequence_similarity(subsequence, common_prog)
                    best_match = max(best_match, similarity)
        
        return best_match
    
    def _note_to_number(self, note: str) -> int:
        """Convert note name to numeric value."""
        notes = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 
                'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        return notes.get(note, 0)
    
    def _quality_to_score(self, quality: str) -> int:
        """Convert chord quality to numeric score."""
        quality_scores = {
            'maj': 0, 'min': 12, 'dom7': 24, 'min7': 36, 
            'maj7': 48, 'dim': 60, 'sus2': 72, 'sus4': 84
        }
        return quality_scores.get(quality, 0)
    
    def _calculate_sequence_similarity(self, seq1: List[int], seq2: List[int]) -> float:
        """Calculate similarity between two sequences."""
        if len(seq1) != len(seq2):
            return 0.0
        
        # Calculate interval similarity
        intervals1 = [seq1[i+1] - seq1[i] for i in range(len(seq1)-1)]
        intervals2 = [seq2[i+1] - seq2[i] for i in range(len(seq2)-1)]
        
        if len(intervals1) != len(intervals2) or not intervals1:
            return 0.0
        
        # Compare intervals
        interval_similarities = []
        for i1, i2 in zip(intervals1, intervals2):
            # Normalize intervals to same octave
            i1_norm = i1 % 12
            i2_norm = i2 % 12
            similarity = 1.0 - abs(i1_norm - i2_norm) / 6.0  # Max distance is 6
            interval_similarities.append(max(0.0, similarity))
        
        return np.mean(interval_similarities)
    
    def _analyze_chord_distribution(self, progressions: List[List[Dict]]) -> Dict[str, float]:
        """Analyze distribution of chord types."""
        all_chords = []
        for progression in progressions:
            all_chords.extend([f"{chord['root']}_{chord['quality']}" for chord in progression])
        
        if not all_chords:
            return {}
        
        chord_counts = Counter(all_chords)
        total_chords = len(all_chords)
        
        # Calculate distribution metrics
        distribution = {chord: count / total_chords for chord, count in chord_counts.items()}
        
        # Calculate diversity metrics
        unique_chords = len(chord_counts)
        diversity_score = unique_chords / total_chords if total_chords > 0 else 0.0
        
        return {
            "distribution": distribution,
            "unique_chords": unique_chords,
            "diversity_score": diversity_score,
            "total_chords": total_chords
        }
    
    def _analyze_harmonic_rhythm(self, progressions: List[List[Dict]]) -> Dict[str, float]:
        """Analyze harmonic rhythm patterns."""
        durations = []
        
        for progression in progressions:
            for i in range(len(progression) - 1):
                current_time = progression[i]["time_start"]
                next_time = progression[i + 1]["time_start"]
                duration = next_time - current_time
                if duration > 0:
                    durations.append(duration)
        
        if not durations:
            return {"error": "No harmonic rhythm data"}
        
        return {
            "mean_duration": np.mean(durations),
            "std_duration": np.std(durations),
            "min_duration": np.min(durations),
            "max_duration": np.max(durations),
            "median_duration": np.median(durations)
        }


class MelodyAnalyzer:
    """Analyzes melodic characteristics."""
    
    def analyze_melodies(self, nomad_data: List[Dict]) -> Dict[str, Any]:
        """Analyze melodic characteristics.
        
        Args:
            nomad_data: List of Nomad Symbolic Format dictionaries
            
        Returns:
            Melody analysis metrics
        """
        melodies = []
        
        # Extract melodies
        for data in nomad_data:
            melody_track = self._find_melody_track(data)
            if melody_track:
                melody = self._extract_melody(melody_track)
                if melody:
                    melodies.append(melody)
        
        if not melodies:
            return {"error": "No melodies found"}
        
        # Chord-tone ratio analysis
        chord_tone_ratios = []
        for melody in melodies:
            ratio = self._calculate_chord_tone_ratio(melody)
            chord_tone_ratios.append(ratio)
        
        # Melodic contour analysis
        contour_stats = self._analyze_melodic_contour(melodies)
        
        # Phrase coherence analysis
        phrase_stats = self._analyze_phrase_coherence(melodies)
        
        # Note density analysis
        density_stats = self._analyze_note_density(melodies)
        
        return {
            "chord_tone_ratio_mean": np.mean(chord_tone_ratios),
            "chord_tone_ratio_std": np.std(chord_tone_ratios),
            "melodic_contour_stats": contour_stats,
            "phrase_coherence_stats": phrase_stats,
            "note_density_stats": density_stats,
            "num_melodies_analyzed": len(melodies)
        }
    
    def _find_melody_track(self, nomad_data: Dict) -> Optional[Dict]:
        """Find melody track in Nomad data."""
        for track in nomad_data.get("tracks", []):
            track_type = track["classification"]["type"]
            if track_type in ["melody", "bass"]:
                return track
        return None
    
    def _extract_melody(self, melody_track: Dict) -> Optional[List[Dict]]:
        """Extract melody from track."""
        events = [e for e in melody_track["events"] if e["type"] == "note_on"]
        if not events:
            return None
        
        # Sort by time and convert to melody representation
        events.sort(key=lambda x: x["time_seconds"])
        
        melody = []
        for event in events:
            melody.append({
                "pitch": event["pitch"],
                "time": event["time_seconds"],
                "duration": event.get("duration", 0.5),
                "velocity": event.get("velocity", 64)
            })
        
        return melody
    
    def _calculate_chord_tone_ratio(self, melody: List[Dict]) -> float:
        """Calculate ratio of chord tones to non-chord tones."""
        if not melody:
            return 0.0
        
        # This is a simplified implementation
        # In practice, we'd need to align melody with chord progression
        chord_tones = 0
        total_notes = len(melody)
        
        for note in melody:
            pitch_class = note["pitch"] % 12
            # Simplified: consider pitch classes 0, 2, 4, 5, 7, 9, 11 as chord tones
            if pitch_class in [0, 2, 4, 5, 7, 9, 11]:
                chord_tones += 1
        
        return chord_tones / total_notes if total_notes > 0 else 0.0
    
    def _analyze_melodic_contour(self, melodies: List[List[Dict]]) -> Dict[str, float]:
        """Analyze melodic contour characteristics."""
        intervals = []
        
        for melody in melodies:
            if len(melody) >= 2:
                melody_intervals = []
                for i in range(len(melody) - 1):
                    interval = melody[i + 1]["pitch"] - melody[i]["pitch"]
                    melody_intervals.append(interval)
                intervals.extend(melody_intervals)
        
        if not intervals:
            return {"error": "No intervals found"}
        
        return {
            "mean_interval": np.mean(intervals),
            "std_interval": np.std(intervals),
            "max_interval": np.max(intervals),
            "min_interval": np.min(intervals),
            "large_leaps_ratio": np.sum(np.abs(np.array(intervals)) > 5) / len(intervals)
        }
    
    def _analyze_phrase_coherence(self, melodies: List[List[Dict]]) -> Dict[str, float]:
        """Analyze phrase coherence and repetition."""
        repetition_scores = []
        
        for melody in melodies:
            if len(melody) >= 4:
                # Look for repeated motifs (simplified)
                score = self._calculate_motif_repetition(melody)
                repetition_scores.append(score)
        
        if not repetition_scores:
            return {"repetition_score": 0.0}
        
        return {
            "repetition_score": np.mean(repetition_scores),
            "repetition_std": np.std(repetition_scores)
        }
    
    def _calculate_motif_repetition(self, melody: List[Dict]) -> float:
        """Calculate motif repetition in a melody."""
        if len(melody) < 4:
            return 0.0
        
        # Extract pitch sequences (simplified)
        pitches = [note["pitch"] for note in melody]
        
        # Look for repeated 3-note motifs
        motif_length = 3
        motifs = []
        for i in range(len(pitches) - motif_length + 1):
            motif = tuple(pitches[i:i + motif_length])
            motifs.append(motif)
        
        # Count motif occurrences
        motif_counts = Counter(motifs)
        repeated_motifs = sum(1 for count in motif_counts.values() if count > 1)
        
        return repeated_motifs / len(motifs) if motifs else 0.0
    
    def _analyze_note_density(self, melodies: List[List[Dict]]) -> Dict[str, float]:
        """Analyze note density across melodies."""
        densities = []
        
        for melody in melodies:
            if melody:
                total_duration = melody[-1]["time"] - melody[0]["time"]
                if total_duration > 0:
                    density = len(melody) / total_duration
                    densities.append(density)
        
        if not densities:
            return {"error": "No density data"}
        
        return {
            "mean_density": np.mean(densities),
            "std_density": np.std(densities),
            "min_density": np.min(densities),
            "max_density": np.max(densities)
        }


class MemorizationDetector:
    """Detects memorization vs genuine generation."""
    
    def __init__(self, ngram_size: int = 4):
        """Initialize detector.
        
        Args:
            ngram_size: Size of n-grams for comparison
        """
        self.ngram_size = ngram_size
    
    def detect_memorization(self, generated_data: List[Dict], 
                          training_data: List[Dict]) -> Dict[str, Any]:
        """Detect memorization in generated data.
        
        Args:
            generated_data: Generated data to check
            training_data: Training data for comparison
            
        Returns:
            Memorization detection metrics
        """
        # Extract n-grams from training data
        training_ngrams = self._extract_ngrams(training_data)
        
        # Check generated data against training n-grams
        memorization_scores = []
        
        for gen_data in generated_data:
            score = self._calculate_memorization_score(gen_data, training_ngrams)
            memorization_scores.append(score)
        
        return {
            "mean_memorization_score": np.mean(memorization_scores),
            "max_memorization_score": np.max(memorization_scores),
            "high_memorization_ratio": np.sum(np.array(memorization_scores) > 0.8) / len(memorization_scores),
            "ngram_size": self.ngram_size
        }
    
    def _extract_ngrams(self, data_list: List[Dict]) -> set:
        """Extract n-grams from data."""
        ngrams = set()
        
        for data in data_list:
            # Extract sequences from different track types
            for track in data.get("tracks", []):
                track_type = track["classification"]["type"]
                events = [e for e in track["events"] if e["type"] == "note_on"]
                
                if events:
                    # Create sequences based on track type
                    if track_type == "drums":
                        sequence = [f"DRUM_{e['pitch']}" for e in events]
                    elif track_type == "melody":
                        sequence = [f"NOTE_{e['pitch']}" for e in events]
                    else:
                        continue
                    
                    # Extract n-grams
                    for i in range(len(sequence) - self.ngram_size + 1):
                        ngram = tuple(sequence[i:i + self.ngram_size])
                        ngrams.add(ngram)
        
        return ngrams
    
    def _calculate_memorization_score(self, gen_data: Dict, training_ngrams: set) -> float:
        """Calculate memorization score for generated data."""
        total_ngrams = 0
        matching_ngrams = 0
        
        for track in gen_data.get("tracks", []):
            track_type = track["classification"]["type"]
            events = [e for e in track["events"] if e["type"] == "note_on"]
            
            if events:
                # Create sequences based on track type
                if track_type == "drums":
                    sequence = [f"DRUM_{e['pitch']}" for e in events]
                elif track_type == "melody":
                    sequence = [f"NOTE_{e['pitch']}" for e in events]
                else:
                    continue
                
                # Check n-grams
                for i in range(len(sequence) - self.ngram_size + 1):
                    ngram = tuple(sequence[i:i + self.ngram_size])
                    total_ngrams += 1
                    if ngram in training_ngrams:
                        matching_ngrams += 1
        
        return matching_ngrams / total_ngrams if total_ngrams > 0 else 0.0


def evaluate_multi_expert_generation(generated_data: List[Dict], 
                                   training_data: List[Dict],
                                   config: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive evaluation of multi-expert generation.
    
    Args:
        generated_data: Generated data to evaluate
        training_data: Training data for comparison
        config: Configuration dictionary
        
    Returns:
        Comprehensive evaluation metrics
    """
    logger.info("Starting comprehensive musical evaluation...")
    
    evaluation_results = {}
    
    # Groove stability analysis
    try:
        groove_analyzer = GrooveStabilityAnalyzer()
        evaluation_results["drums"] = groove_analyzer.analyze_groove_stability(generated_data)
        logger.info("Completed drum groove analysis")
    except Exception as e:
        logger.error(f"Groove analysis failed: {e}")
        evaluation_results["drums"] = {"error": str(e)}
    
    # Chord progression analysis
    try:
        chord_analyzer = ChordProgressionAnalyzer()
        evaluation_results["harmony"] = chord_analyzer.analyze_chord_progressions(generated_data)
        logger.info("Completed chord progression analysis")
    except Exception as e:
        logger.error(f"Chord analysis failed: {e}")
        evaluation_results["harmony"] = {"error": str(e)}
    
    # Melody analysis
    try:
        melody_analyzer = MelodyAnalyzer()
        evaluation_results["melody"] = melody_analyzer.analyze_melodies(generated_data)
        logger.info("Completed melody analysis")
    except Exception as e:
        logger.error(f"Melody analysis failed: {e}")
        evaluation_results["melody"] = {"error": str(e)}
    
    # Memorization detection
    try:
        mem_detector = MemorizationDetector()
        evaluation_results["memorization"] = mem_detector.detect_memorization(generated_data, training_data)
        logger.info("Completed memorization detection")
    except Exception as e:
        logger.error(f"Memorization detection failed: {e}")
        evaluation_results["memorization"] = {"error": str(e)}
    
    # Overall quality score
    overall_score = calculate_overall_quality_score(evaluation_results)
    evaluation_results["overall_score"] = overall_score
    
    logger.info(f"Musical evaluation complete. Overall score: {overall_score:.3f}")
    
    return evaluation_results


def calculate_overall_quality_score(evaluation_results: Dict[str, Any]) -> float:
    """Calculate overall quality score from individual metrics.
    
    Args:
        evaluation_results: Results from individual analyzers
        
    Returns:
        Overall quality score (0.0 to 1.0)
    """
    scores = []
    
    # Drum groove consistency (prefer moderate consistency)
    if "drums" in evaluation_results and "groove_consistency" in evaluation_results["drums"]:
        groove_score = evaluation_results["drums"]["groove_consistency"]
        # Penalize extremely high or low consistency
        if 0.3 <= groove_score <= 0.8:
            scores.append(groove_score)
        else:
            scores.append(0.5)
    
    # Chord progression plausibility
    if "harmony" in evaluation_results and "progression_plausibility_mean" in evaluation_results["harmony"]:
        chord_score = evaluation_results["harmony"]["progression_plausibility_mean"]
        scores.append(chord_score)
    
    # Melody chord-tone ratio (prefer 0.6-0.9)
    if "melody" in evaluation_results and "chord_tone_ratio_mean" in evaluation_results["melody"]:
        melody_score = evaluation_results["melody"]["chord_tone_ratio_mean"]
        if 0.4 <= melody_score <= 0.9:
            scores.append(melody_score)
        else:
            scores.append(0.5)
    
    # Low memorization is good
    if "memorization" in evaluation_results and "mean_memorization_score" in evaluation_results["memorization"]:
        mem_score = 1.0 - evaluation_results["memorization"]["mean_memorization_score"]
        scores.append(mem_score)
    
    return np.mean(scores) if scores else 0.0