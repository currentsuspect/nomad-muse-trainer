"""MuseHarmony tokenizer - Expert tokenizer for harmonic events.

This tokenizer handles chord progression and harmonic rhythm with support for:
- Stable chord progressions
- Key-aware chord generation
- Harmonic rhythm modeling

Token vocabulary:
- BAR_<n>: Bar markers
- BEAT_<n>: Beat markers  
- KEY_<key>: Key signature tokens
- CHORD_<root>_<quality>: Chord tokens (e.g., CHORD_C_maj, CHORD_F_min7)
- HOLD_<n>: Hold current chord for n beats
- REST: No chord/silent beat
"""

import json
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path

from nomad_format import NomadFormatBuilder
from vocab import MusicVocabulary


class MuseHarmonyVocabulary(MusicVocabulary):
    """Vocabulary for harmonic events."""
    
    def __init__(self, max_bars: int = 16, include_keys: bool = True):
        """Initialize harmony vocabulary.
        
        Args:
            max_bars: Maximum bars to support
            include_keys: Whether to include key tokens
        """
        self.max_bars = max_bars
        self.include_keys = include_keys
        
        # Define chord qualities and their semitone intervals
        self.chord_qualities = {
            "maj": [0, 4, 7],           # Major triad
            "min": [0, 3, 7],           # Minor triad
            "dim": [0, 3, 6],           # Diminished triad
            "aug": [0, 4, 8],           # Augmented triad
            "sus2": [0, 2, 7],          # Suspended 2nd
            "sus4": [0, 5, 7],          # Suspended 4th
            "dom7": [0, 4, 7, 10],      # Dominant 7th
            "maj7": [0, 4, 7, 11],      # Major 7th
            "min7": [0, 3, 7, 10],      # Minor 7th
            "dim7": [0, 3, 6, 9],       # Diminished 7th
            "halfdim7": [0, 3, 6, 10],  # Half-diminished 7th
        }
        
        # All 12 chromatic notes
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        super().__init__(
            time_shift_bins=8,  # Simplified for harmony
            velocity_bins=1,   # Not used for harmony
            duration_bins=8    # Hold durations
        )
    
    def _build_vocab(self):
        """Build harmony-specific vocabulary."""
        tokens = []
        
        # Special tokens
        tokens.extend([self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN])
        
        # Position markers
        for bar in range(self.max_bars):
            tokens.append(f"BAR_{bar}")
        for beat in range(4):  # Assume 4/4 for now
            tokens.append(f"BEAT_{beat}")
        
        # Key signatures (major and minor)
        if self.include_keys:
            for note in self.note_names:
                tokens.append(f"KEY_{note}_maj")
                tokens.append(f"KEY_{note}_min")
        
        # Chord tokens for each root and quality
        for root in self.note_names:
            for quality in self.chord_qualities.keys():
                tokens.append(f"CHORD_{root}_{quality}")
        
        # Rest/no chord
        tokens.append("REST")
        
        # Hold tokens for harmonic rhythm
        for hold_beats in range(1, 9):  # Hold for 1-8 beats
            tokens.append(f"HOLD_{hold_beats}")
        
        # Create mappings
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
    
    def get_chord_intervals(self, chord_token: str) -> Optional[List[int]]:
        """Get interval pattern for a chord token.
        
        Args:
            chord_token: Chord token string
            
        Returns:
            List of intervals or None if not a chord
        """
        if not chord_token.startswith("CHORD_"):
            return None
        
        parts = chord_token.split("_")
        if len(parts) != 3:
            return None
        
        quality = parts[2]
        return self.chord_qualities.get(quality)
    
    def get_chord_pitches(self, chord_token: str, root_octave: int = 4) -> List[int]:
        """Get MIDI pitches for a chord token.
        
        Args:
            chord_token: Chord token string
            root_octave: Octave for root note
            
        Returns:
            List of MIDI pitch numbers
        """
        if not chord_token.startswith("CHORD_"):
            return []
        
        parts = chord_token.split("_")
        if len(parts) != 3:
            return []
        
        root_note = parts[1]
        quality = parts[2]
        
        # Convert root note to MIDI pitch
        root_pc = self.note_names.index(root_note)
        root_pitch = root_octave * 12 + root_pc
        
        intervals = self.get_chord_intervals(chord_token)
        if not intervals:
            return [root_pitch]
        
        return [root_pitch + interval for interval in intervals]


class MuseHarmonyTokenizer:
    """Tokenizer for harmonic events from Nomad Symbolic Format."""
    
    def __init__(self, vocab: MuseHarmonyVocabulary):
        """Initialize tokenizer.
        
        Args:
            vocab: MuseHarmony vocabulary
        """
        self.vocab = vocab
    
    def tokenize_harmony_track(self, nomad_data: Dict) -> List[str]:
        """Tokenize harmony track from Nomad Symbolic Format.
        
        Args:
            nomad_data: Data in Nomad Symbolic Format
            
        Returns:
            List of harmony tokens
        """
        # Use chord events from Nomad format
        chords = nomad_data.get("chords", [])
        
        if not chords:
            # No chord track found - return minimal sequence
            return [self.vocab.BOS_TOKEN, "REST", self.vocab.EOS_TOKEN]
        
        # Convert chord events to tokens
        return self._chords_to_tokens(chords, nomad_data["global"])
    
    def _chords_to_tokens(self, chords: List[Dict], global_info: Dict) -> List[str]:
        """Convert chord events to harmonic tokens.
        
        Args:
            chords: List of chord events
            global_info: Global musical information
            
        Returns:
            List of harmony tokens
        """
        if not chords:
            return [self.vocab.BOS_TOKEN, "REST", self.vocab.EOS_TOKEN]
        
        tokens = [self.vocab.BOS_TOKEN]
        current_bar = 0
        
        # Extract tempo information
        tempo_map = global_info.get("tempo_map", [(0.0, 120.0)])
        beats_per_second = tempo_map[0][1] / 60.0
        
        for i, chord_event in enumerate(chords):
            bar_start = chord_event["bar_start"]
            beat_start = chord_event["beat_start"]
            chord_info = chord_event["chord"]
            
            # Add bar marker if we've moved to a new bar
            if bar_start > current_bar:
                tokens.append(f"BAR_{bar_start % self.vocab.max_bars}")
                current_bar = bar_start
            
            # Add beat marker if not at beat 0
            if beat_start > 0:
                beat_in_bar = int(beat_start) - 1
                if beat_in_bar >= 0 and beat_in_bar < 4:
                    tokens.append(f"BEAT_{beat_in_bar}")
            
            # Add chord token
            chord_root = chord_info.get("root", "C")
            chord_quality = chord_info.get("quality", "maj")
            tokens.append(f"CHORD_{chord_root}_{chord_quality}")
            
            # Calculate hold duration (until next chord)
            next_chord_time = chords[i + 1]["time_start"] if i + 1 < len(chords) else chord_event["time_end"]
            current_chord_duration = next_chord_time - chord_event["time_start"]
            hold_beats = int(current_chord_duration * beats_per_second)
            
            if hold_beats > 1 and i + 1 < len(chords):  # Don't hold the last chord
                tokens.append(f"HOLD_{min(hold_beats, 8)}")
        
        tokens.append(self.vocab.EOS_TOKEN)
        return tokens
    
    def extract_key_changes(self, nomad_data: Dict) -> List[str]:
        """Extract key signature changes from Nomad format.
        
        Args:
            nomad_data: Data in Nomad Symbolic Format
            
        Returns:
            List of key tokens
        """
        # For now, estimate key from first chord root
        # A full implementation would analyze the entire harmonic content
        chords = nomad_data.get("chords", [])
        if not chords:
            return []
        
        # Simple heuristic: use the most common chord root as key
        root_counts = {}
        for chord_event in chords:
            root = chord_event["chord"].get("root", "C")
            root_counts[root] = root_counts.get(root, 0) + 1
        
        if not root_counts:
            return []
        
        most_common_root = max(root_counts.items(), key=lambda x: x[1])[0]
        
        # For simplicity, assume major key
        return [f"KEY_{most_common_root}_maj"]
    
    def decode_to_chord_events(self, tokens: List[str]) -> List[Dict]:
        """Convert harmony tokens back to chord events.
        
        Args:
            tokens: List of harmony tokens
            
        Returns:
            List of chord events suitable for MIDI generation
        """
        events = []
        current_bar = 0
        current_beat = 0
        current_time = 0.0
        
        # Default tempo
        tempo = 120.0
        beats_per_second = tempo / 60.0
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Parse position markers
            if token.startswith("BAR_"):
                current_bar = int(token.split("_")[1])
                current_beat = 0
            elif token.startswith("BEAT_"):
                current_beat = int(token.split("_")[1])
            
            # Parse chord tokens
            elif token.startswith("CHORD_"):
                parts = token.split("_")
                if len(parts) == 3:
                    root = parts[1]
                    quality = parts[2]
                    
                    # Calculate time
                    current_time = self._musical_position_to_time(
                        current_bar, current_beat, beats_per_second
                    )
                    
                    events.append({
                        "time_start": current_time,
                        "time_end": current_time + beats_per_second,  # Default 1 beat
                        "bar_start": current_bar,
                        "beat_start": current_beat + 1,
                        "chord": {
                            "root": root,
                            "quality": quality,
                            "confidence": 0.8
                        }
                    })
            
            # Parse hold tokens
            elif token.startswith("HOLD_"):
                hold_beats = int(token.split("_")[1])
                if events:
                    # Extend the last chord
                    last_event = events[-1]
                    last_event["time_end"] += hold_beats * beats_per_second
            
            i += 1
        
        return events
    
    def _musical_position_to_time(self, bar: int, beat: int, beats_per_second: float) -> float:
        """Convert musical position to time in seconds.
        
        Args:
            bar: Bar number
            beat: Beat in bar (0-based)
            beats_per_second: Beats per second
            
        Returns:
            Time in seconds
        """
        total_beats = bar * 4 + beat  # Assume 4/4 time
        return total_beats / beats_per_second


def create_harmony_tokenizer(config: dict) -> MuseHarmonyTokenizer:
    """Create MuseHarmony tokenizer from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MuseHarmonyTokenizer instance
    """
    harmony_config = config.get("harmony", {})
    
    vocab = MuseHarmonyVocabulary(
        max_bars=harmony_config.get("max_bars", 16),
        include_keys=harmony_config.get("include_keys", True)
    )
    
    return MuseHarmonyTokenizer(vocab)