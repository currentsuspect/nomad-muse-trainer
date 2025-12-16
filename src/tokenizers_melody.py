"""MuseMelody tokenizer - Expert tokenizer for melodic and bass events.

This tokenizer handles melody and bass lines with support for:
- Coherent melodic phrases
- Harmony conditioning
- Bass line generation

Token vocabulary:
- BAR_<n>: Bar markers
- BEAT_<n>: Beat markers
- SUBSTEP_<n>: 16th note markers  
- NOTE_<pitch>: Note events (MIDI pitches 0-127)
- VEL_BIN_<k>: Velocity quantization
- DUR_BIN_<k>: Duration quantization
- REST_<n>: Rest events for timing
- CONDITION_CHORD_<root>_<quality>: Conditioning chord information
- CONDITION_KEY_<key>: Conditioning key information
"""

import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from nomad_format import NomadFormatBuilder
from vocab import MusicVocabulary


class MuseMelodyVocabulary(MusicVocabulary):
    """Vocabulary for melodic events."""
    
    def __init__(self, max_bars: int = 16, time_signature: Tuple[int, int] = (4, 4),
                 velocity_bins: int = 8, duration_bins: int = 16, max_pitch: int = 127):
        """Initialize melody vocabulary.
        
        Args:
            max_bars: Maximum bars to support
            time_signature: (numerator, denominator)
            velocity_bins: Number of velocity quantization levels
            duration_bins: Number of duration quantization levels
            max_pitch: Maximum pitch to include
        """
        self.max_bars = max_bars
        self.time_signature = time_signature
        self.velocity_bins = velocity_bins
        self.duration_bins = duration_bins
        self.max_pitch = max_pitch
        
        # Calculate musical divisions
        self.beats_per_bar = time_signature[0]
        self.substeps_per_beat = 4  # 16th notes
        self.substeps_per_bar = self.beats_per_bar * self.substeps_per_beat
        
        super().__init__(
            time_shift_bins=16,  # 16th note resolution
            velocity_bins=velocity_bins,
            duration_bins=duration_bins
        )
    
    def _build_vocab(self):
        """Build melody-specific vocabulary."""
        tokens = []
        
        # Special tokens
        tokens.extend([self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN])
        
        # Position markers
        for bar in range(self.max_bars):
            tokens.append(f"BAR_{bar}")
        for beat in range(self.beats_per_bar):
            tokens.append(f"BEAT_{beat}")
        for substep in range(self.substeps_per_bar):
            tokens.append(f"SUBSTEP_{substep}")
        
        # Note events for each pitch
        for pitch in range(self.max_pitch + 1):
            tokens.append(f"NOTE_{pitch}")
        
        # Velocity bins
        for i in range(1, self.velocity_bins + 1):
            tokens.append(f"VEL_BIN_{i}")
        
        # Duration bins (musical durations)
        durations = [
            0.25,   # 16th note
            0.5,    # 8th note
            0.75,   # Dotted 8th
            1.0,    # Quarter note
            1.5,    # Dotted quarter
            2.0,    # Half note
            3.0,    # Dotted half
            4.0,    # Whole note
        ]
        
        for i, duration in enumerate(durations, 1):
            tokens.append(f"DUR_BIN_{i}")
        
        # Rest markers (different time granularities)
        for duration in durations:
            tokens.append(f"REST_{duration}")
        
        # Conditioning tokens for harmony
        # Chord conditioning (simplified set)
        chord_roots = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chord_qualities = ['maj', 'min', 'dom7', 'min7', 'maj7']
        
        for root in chord_roots:
            for quality in chord_qualities:
                tokens.append(f"CONDITION_CHORD_{root}_{quality}")
        
        # Key conditioning
        for root in chord_roots:
            tokens.append(f"CONDITION_KEY_{root}_maj")
            tokens.append(f"CONDITION_KEY_{root}_min")
        
        # Create mappings
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}


class MuseMelodyTokenizer:
    """Tokenizer for melodic events from Nomad Symbolic Format."""
    
    def __init__(self, vocab: MuseMelodyVocabulary, conditioning: str = "none"):
        """Initialize tokenizer.
        
        Args:
            vocab: MuseMelody vocabulary
            conditioning: Type of conditioning ("none", "chord", "key", "both")
        """
        self.vocab = vocab
        self.conditioning = conditioning
    
    def tokenize_melody_track(self, nomad_data: Dict) -> List[str]:
        """Tokenize melody track from Nomad Symbolic Format.
        
        Args:
            nomad_data: Data in Nomad Symbolic Format
            
        Returns:
            List of melody tokens
        """
        # Find melody track
        melody_track = None
        for track in nomad_data["tracks"]:
            track_type = track["classification"]["type"]
            if track_type in ["melody", "bass"]:
                melody_track = track
                break
        
        if not melody_track:
            # No melody track found - return minimal sequence
            return [self.vocab.BOS_TOKEN, "REST_1.0", self.vocab.EOS_TOKEN]
        
        # Extract note events
        events = [e for e in melody_track["events"] if e["type"] == "note_on"]
        
        if not events:
            return [self.vocab.BOS_TOKEN, "REST_1.0", self.vocab.EOS_TOKEN]
        
        # Sort by time
        events.sort(key=lambda x: x["time_seconds"])
        
        # Convert to tokens with conditioning
        return self._events_to_tokens(events, nomad_data)
    
    def _events_to_tokens(self, events: List[Dict], nomad_data: Dict) -> List[str]:
        """Convert melodic events to tokens with optional conditioning.
        
        Args:
            events: List of note events
            nomad_data: Full Nomad Symbolic Format data
            
        Returns:
            List of melody tokens
        """
        tokens = [self.vocab.BOS_TOKEN]
        current_time = 0.0
        current_bar = 0
        current_beat = 0
        current_substep = 0
        
        # Extract global tempo information
        global_info = nomad_data["global"]
        tempo_map = global_info.get("tempo_map", [(0.0, 120.0)])
        beats_per_second = tempo_map[0][1] / 60.0
        
        # Add conditioning tokens at the beginning
        if self.conditioning in ["chord", "both"]:
            conditioning_chords = self._extract_conditioning_chords(nomad_data)
            tokens.extend(conditioning_chords[:4])  # Limit to first 4 chords
        
        if self.conditioning in ["key", "both"]:
            conditioning_keys = self._extract_conditioning_keys(nomad_data)
            tokens.extend(conditioning_keys[:2])  # Limit to first 2 keys
        
        for i, event in enumerate(events):
            event_time = event["time_seconds"]
            pitch = event["pitch"]
            velocity = event["velocity"]
            duration = event.get("duration", 0.5)  # Default duration
            
            # Calculate musical position
            bar, beat, substep = self._time_to_musical_position(
                event_time, beats_per_second, current_bar, current_beat, current_substep
            )
            
            # Add position markers if we've advanced
            if bar > current_bar:
                tokens.append(f"BAR_{bar % self.vocab.max_bars}")
                current_bar = bar
                current_beat = 0
                current_substep = 0
            
            if beat > current_beat:
                tokens.append(f"BEAT_{beat}")
                current_beat = beat
                current_substep = 0
            
            if substep > current_substep and substep % 2 == 0:  # Every 8th note
                tokens.append(f"SUBSTEP_{substep}")
                current_substep = substep
            
            # Add note token
            tokens.append(f"NOTE_{pitch}")
            
            # Add velocity token
            vel_bin = self._velocity_to_bin(velocity)
            tokens.append(f"VEL_BIN_{vel_bin}")
            
            # Add duration token
            dur_bin = self._duration_to_bin(duration)
            tokens.append(f"DUR_BIN_{dur_bin}")
            
            current_time = event_time
        
        tokens.append(self.vocab.EOS_TOKEN)
        return tokens
    
    def _extract_conditioning_chords(self, nomad_data: Dict) -> List[str]:
        """Extract chord conditioning tokens.
        
        Args:
            nomad_data: Nomad Symbolic Format data
            
        Returns:
            List of conditioning chord tokens
        """
        chords = nomad_data.get("chords", [])
        if not chords:
            return []
        
        conditioning_tokens = []
        # Use first few chords for conditioning
        for chord_event in chords[:4]:
            chord_info = chord_event["chord"]
            root = chord_info.get("root", "C")
            quality = chord_info.get("quality", "maj")
            conditioning_tokens.append(f"CONDITION_CHORD_{root}_{quality}")
        
        return conditioning_tokens
    
    def _extract_conditioning_keys(self, nomad_data: Dict) -> List[str]:
        """Extract key conditioning tokens.
        
        Args:
            nomad_data: Nomad Symbolic Format data
            
        Returns:
            List of conditioning key tokens
        """
        # Simple heuristic: estimate key from most common chord root
        chords = nomad_data.get("chords", [])
        if not chords:
            return ["CONDITION_KEY_C_maj"]  # Default key
        
        root_counts = {}
        for chord_event in chords:
            root = chord_event["chord"].get("root", "C")
            root_counts[root] = root_counts.get(root, 0) + 1
        
        if not root_counts:
            return ["CONDITION_KEY_C_maj"]
        
        most_common_root = max(root_counts.items(), key=lambda x: x[1])[0]
        return [f"CONDITION_KEY_{most_common_root}_maj"]
    
    def _time_to_musical_position(self, time_seconds: float, beats_per_second: float,
                                  current_bar: int, current_beat: int, 
                                  current_substep: int) -> Tuple[int, int, int]:
        """Convert time in seconds to musical position.
        
        Args:
            time_seconds: Time in seconds
            beats_per_second: Beats per second
            current_bar: Current bar number
            current_beat: Current beat in bar
            current_substep: Current substep in beat
            
        Returns:
            (bar, beat, substep) tuple
        """
        total_beats = time_seconds * beats_per_second
        bar = int(total_beats // self.vocab.beats_per_bar)
        beat_in_bar = int(total_beats % self.vocab.beats_per_bar)
        substep = int((total_beats % 1.0) * self.vocab.substeps_per_beat)
        
        return bar, beat_in_bar, substep
    
    def _velocity_to_bin(self, velocity: int) -> int:
        """Convert MIDI velocity to bin number.
        
        Args:
            velocity: MIDI velocity (0-127)
            
        Returns:
            Velocity bin number
        """
        if velocity <= 0:
            return 1
        bin_idx = min(int((velocity / 127.0) * self.vocab.velocity_bins) + 1, 
                     self.vocab.velocity_bins)
        return bin_idx
    
    def _duration_to_bin(self, duration_beats: float) -> int:
        """Convert duration in beats to bin number.
        
        Args:
            duration_beats: Duration in beats
            
        Returns:
            Duration bin number
        """
        duration_map = {
            0.25: 1,   # 16th note
            0.5: 2,    # 8th note
            0.75: 3,   # Dotted 8th
            1.0: 4,    # Quarter note
            1.5: 5,    # Dotted quarter
            2.0: 6,    # Half note
            3.0: 7,    # Dotted half
            4.0: 8,    # Whole note
        }
        
        # Find closest duration
        closest_duration = min(duration_map.keys(), key=lambda x: abs(x - duration_beats))
        return duration_map[closest_duration]
    
    def decode_to_midi_events(self, tokens: List[str]) -> List[Dict]:
        """Convert melody tokens back to MIDI-like events.
        
        Args:
            tokens: List of melody tokens
            
        Returns:
            List of events suitable for MIDI generation
        """
        events = []
        current_bar = 0
        current_beat = 0
        current_substep = 0
        
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
                current_substep = 0
            elif token.startswith("BEAT_"):
                current_beat = int(token.split("_")[1])
                current_substep = 0
            elif token.startswith("SUBSTEP_"):
                current_substep = int(token.split("_")[1])
            
            # Parse note events
            elif token.startswith("NOTE_"):
                pitch = int(token.split("_")[1])
                
                # Look for velocity and duration
                velocity = 64  # Default
                duration = 0.5  # Default 0.5 beats
                
                # Check next tokens for velocity and duration
                if i + 1 < len(tokens) and tokens[i + 1].startswith("VEL_BIN_"):
                    vel_bin = int(tokens[i + 1].split("_")[2])
                    velocity = self._bin_to_velocity(vel_bin)
                    i += 1
                
                if i + 1 < len(tokens) and tokens[i + 1].startswith("DUR_BIN_"):
                    dur_bin = int(tokens[i + 1].split("_")[2])
                    duration = self._bin_to_duration(dur_bin)
                    i += 1
                
                # Calculate time from position
                time_seconds = self._musical_position_to_time(
                    current_bar, current_beat, current_substep, beats_per_second
                )
                
                events.append({
                    "time": time_seconds,
                    "pitch": pitch,
                    "velocity": velocity,
                    "duration": duration / beats_per_second  # Convert to seconds
                })
            
            # Parse rest events
            elif token.startswith("REST_"):
                rest_duration = float(token.split("_")[1])
                # Rests don't generate events, just advance time
                pass
            
            i += 1
        
        return events
    
    def _bin_to_velocity(self, vel_bin: int) -> int:
        """Convert velocity bin back to MIDI velocity.
        
        Args:
            vel_bin: Velocity bin number
            
        Returns:
            MIDI velocity (0-127)
        """
        velocity = int((vel_bin - 0.5) / self.vocab.velocity_bins * 127)
        return max(1, min(127, velocity))
    
    def _bin_to_duration(self, dur_bin: int) -> float:
        """Convert duration bin back to beats.
        
        Args:
            dur_bin: Duration bin number
            
        Returns:
            Duration in beats
        """
        duration_map = {
            1: 0.25,   # 16th note
            2: 0.5,    # 8th note
            3: 0.75,   # Dotted 8th
            4: 1.0,    # Quarter note
            5: 1.5,    # Dotted quarter
            6: 2.0,    # Half note
            7: 3.0,    # Dotted half
            8: 4.0,    # Whole note
        }
        
        return duration_map.get(dur_bin, 1.0)
    
    def _musical_position_to_time(self, bar: int, beat: int, substep: int, 
                                 beats_per_second: float) -> float:
        """Convert musical position to time in seconds.
        
        Args:
            bar: Bar number
            beat: Beat in bar (0-based)
            substep: Substep in beat (0-based)
            beats_per_second: Beats per second
            
        Returns:
            Time in seconds
        """
        total_beats = (bar * self.vocab.beats_per_bar + beat + 
                      substep / self.vocab.substeps_per_beat)
        return total_beats / beats_per_second


def create_melody_tokenizer(config: dict) -> MuseMelodyTokenizer:
    """Create MuseMelody tokenizer from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MuseMelodyTokenizer instance
    """
    melody_config = config.get("melody", {})
    vocab_config = config.get("tokenization", {})
    
    # Extract time signature
    time_sig = melody_config.get("time_signature", [4, 4])
    if isinstance(time_sig, str):
        # Parse string format like "4/4"
        parts = time_sig.split("/")
        time_sig = (int(parts[0]), int(parts[1]))
    
    vocab = MuseMelodyVocabulary(
        max_bars=melody_config.get("max_bars", 16),
        time_signature=tuple(time_sig),
        velocity_bins=vocab_config.get("velocity_bins", 8),
        duration_bins=vocab_config.get("duration_bins", 16),
        max_pitch=melody_config.get("max_pitch", 127)
    )
    
    conditioning = melody_config.get("conditioning", "none")
    
    return MuseMelodyTokenizer(vocab, conditioning)