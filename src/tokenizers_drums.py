"""MuseDrums tokenizer - Expert tokenizer for percussion events.

This tokenizer handles drum events only, with support for:
- Groove continuation
- Fill generation  
- Pattern variations

Token vocabulary:
- BAR_<n>: Bar markers (0-15 for 4/4 time)
- BEAT_<n>: Beat markers (0-3 for 4/4 time)  
- SUBSTEP_<n>: 16th note markers (0-15)
- DRUM_HIT_<pitch>: GM drum pitch events (35-81 typically)
- VEL_BIN_<k>: Velocity quantization (8 bins)
- REST_<n>: Rest events for timing
- FILL_START, FILL_END: Fill markers (future extension)
"""

import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from nomad_format import NomadFormatBuilder, NomadFormatBuilder
from vocab import MusicVocabulary


class MuseDrumsVocabulary(MusicVocabulary):
    """Vocabulary for drum events only."""
    
    def __init__(self, time_signature: Tuple[int, int] = (4, 4), 
                 velocity_bins: int = 8, max_bars: int = 16):
        """Initialize drums vocabulary.
        
        Args:
            time_signature: (numerator, denominator) e.g. (4, 4)
            velocity_bins: Number of velocity quantization levels
            max_bars: Maximum bars to support
        """
        self.time_signature = time_signature
        self.velocity_bins = velocity_bins
        self.max_bars = max_bars
        
        # Calculate beats and substeps per bar
        self.beats_per_bar = time_signature[0]
        self.substeps_per_beat = 4  # 16th notes
        self.substeps_per_bar = self.beats_per_bar * self.substeps_per_beat
        
        # GM drum pitch ranges (standard MIDI drums)
        self.drum_pitches = list(range(35, 82))  # 35-81 are typical drum pitches
        
        super().__init__(
            time_shift_bins=16,  # Simplified for drums
            velocity_bins=velocity_bins,
            duration_bins=4  # Simplified for drums
        )
    
    def _build_vocab(self):
        """Build drum-specific vocabulary."""
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
        
        # Drum hits for each GM pitch
        for pitch in self.drum_pitches:
            tokens.append(f"DRUM_HIT_{pitch}")
        
        # Velocity bins (8 levels typical for drums)
        for i in range(1, self.velocity_bins + 1):
            tokens.append(f"VEL_BIN_{i}")
        
        # Rest markers (different time granularities)
        for duration in [0.25, 0.5, 1.0, 2.0]:  # 16th, 8th, quarter, half beats
            tokens.append(f"REST_{duration}")
        
        # Fill markers (future extension)
        tokens.extend(["FILL_START", "FILL_END"])
        
        # Create mappings
        self.token_to_id = {token: idx for idx, token in enumerate(tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}


class MuseDrumsTokenizer:
    """Tokenizer for drum events from Nomad Symbolic Format."""
    
    def __init__(self, vocab: MuseDrumsVocabulary):
        """Initialize tokenizer.
        
        Args:
            vocab: MuseDrums vocabulary
        """
        self.vocab = vocab
    
    def tokenize_drum_track(self, nomad_data: Dict) -> List[str]:
        """Tokenize drum track from Nomad Symbolic Format.
        
        Args:
            nomad_data: Data in Nomad Symbolic Format
            
        Returns:
            List of drum tokens
        """
        # Find drum track
        drum_track = None
        for track in nomad_data["tracks"]:
            if track["classification"]["type"] == "drums":
                drum_track = track
                break
        
        if not drum_track:
            # No drum track found - return empty sequence
            return [self.vocab.EOS_TOKEN]
        
        # Extract drum events
        events = [e for e in drum_track["events"] if e["type"] == "note_on"]
        
        if not events:
            return [self.vocab.EOS_TOKEN]
        
        # Sort by time
        events.sort(key=lambda x: x["time_seconds"])
        
        # Convert to musical time and generate tokens
        return self._events_to_tokens(events, nomad_data["global"])
    
    def _events_to_tokens(self, events: List[Dict], global_info: Dict) -> List[str]:
        """Convert drum events to tokens with musical timing.
        
        Args:
            events: List of drum note events
            global_info: Global musical information
            
        Returns:
            List of drum tokens
        """
        tokens = [self.vocab.BOS_TOKEN]
        current_time = 0.0
        
        # Extract tempo information
        tempo_map = global_info.get("tempo_map", [(0.0, 120.0)])
        beats_per_second = tempo_map[0][1] / 60.0
        
        # Calculate bar/beat/substep progression
        current_bar = 0
        current_beat = 0
        current_substep = 0
        
        for event in events:
            event_time = event["time_seconds"]
            pitch = event["pitch"]
            velocity = event["velocity"]
            
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
            
            if substep > current_substep:
                # Add substep markers for significant positions
                if substep % 2 == 0:  # Every 8th note
                    tokens.append(f"SUBSTEP_{substep}")
                current_substep = substep
            
            # Add drum hit token
            tokens.append(f"DRUM_HIT_{pitch}")
            
            # Add velocity token
            vel_bin = self._velocity_to_bin(velocity)
            tokens.append(f"VEL_BIN_{vel_bin}")
            
            # Calculate time until next event for rest markers
            current_time = event_time
        
        tokens.append(self.vocab.EOS_TOKEN)
        return tokens
    
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
            Velocity bin number (1-8)
        """
        if velocity <= 0:
            return 1
        bin_idx = min(int((velocity / 127.0) * self.vocab.velocity_bins) + 1, 
                     self.vocab.velocity_bins)
        return bin_idx
    
    def decode_to_midi_events(self, tokens: List[str]) -> List[Dict]:
        """Convert drum tokens back to MIDI-like events.
        
        Args:
            tokens: List of drum tokens
            
        Returns:
            List of events suitable for MIDI generation
        """
        events = []
        current_bar = 0
        current_beat = 0
        current_substep = 0
        
        # Tempo information for conversion
        tempo = 120.0  # Default
        beats_per_second = tempo / 60.0
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Parse position markers
            if token.startswith("BAR_"):
                current_bar = int(token.split("_")[1])
            elif token.startswith("BEAT_"):
                current_beat = int(token.split("_")[1])
            elif token.startswith("SUBSTEP_"):
                current_substep = int(token.split("_")[1])
            
            # Parse drum hits
            elif token.startswith("DRUM_HIT_"):
                pitch = int(token.split("_")[2])
                
                # Look for following velocity
                velocity = 64  # Default
                if i + 1 < len(tokens) and tokens[i + 1].startswith("VEL_BIN_"):
                    vel_bin = int(tokens[i + 1].split("_")[2])
                    velocity = self._bin_to_velocity(vel_bin)
                    i += 1  # Skip velocity token
                
                # Calculate time from position
                time_seconds = self._musical_position_to_time(
                    current_bar, current_beat, current_substep, beats_per_second
                )
                
                events.append({
                    "time": time_seconds,
                    "pitch": pitch,
                    "velocity": velocity,
                    "duration": 0.1  # Default short duration for drums
                })
            
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


def create_drums_tokenizer(config: dict) -> MuseDrumsTokenizer:
    """Create MuseDrums tokenizer from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        MuseDrumsTokenizer instance
    """
    drums_config = config.get("drums", {})
    vocab_config = config.get("tokenization", {})
    
    # Extract time signature
    time_sig = drums_config.get("time_signature", [4, 4])
    if isinstance(time_sig, str):
        # Parse string format like "4/4"
        parts = time_sig.split("/")
        time_sig = (int(parts[0]), int(parts[1]))
    
    vocab = MuseDrumsVocabulary(
        time_signature=tuple(time_sig),
        velocity_bins=vocab_config.get("velocity_bins", 8),
        max_bars=drums_config.get("max_bars", 16)
    )
    
    return MuseDrumsTokenizer(vocab)