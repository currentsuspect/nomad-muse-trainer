"""Canonical Nomad Symbolic Format - The single source of truth for multi-expert music modeling.

This module defines the intermediate representation that all expert tokenizers derive from.
It captures musical structure in a format optimized for:
- Track classification (drums/melody/bass/chords)
- Expert-specific tokenization
- Reproducible analysis
- Efficient storage

Format:
{
    "global": {
        "tempo_map": [(time_seconds, bpm), ...],
        "time_signature_map": [(time_seconds, (num, den)), ...],
        "ppq": int,  # Pulses per quarter note
        "duration_seconds": float
    },
    "tracks": [
        {
            "id": int,
            "name": str,
            "is_drum": bool,
            "instrument": str,
            "classification": {
                "type": "drums|melody|bass|chords|unknown",
                "confidence": float,
                "reasons": [str, ...]
            },
            "events": [
                {
                    "time_seconds": float,
                    "type": "note_on|note_off|tempo_change|time_signature",
                    "pitch": int,      # For note events (0-127, GM drum pitches for drums)
                    "velocity": int,   # For note_on events (0-127)
                    "duration": float  # For note_on events (seconds)
                },
                ...
            ]
        },
        ...
    ],
    "chords": [
        {
            "time_start": float,
            "time_end": float,
            "bar_start": int,
            "beat_start": float,
            "chord": {
                "root": str,      # e.g., "C", "F#"
                "quality": str,   # e.g., "maj", "min", "dom7", "dim"
                "inversion": int, # 0=root position, 1=first inversion, etc.
                "bass_note": str, # Optional bass note if different from root
                "confidence": float
            }
        },
        ...
    ],
    "metadata": {
        "source_file": str,
        "processing_timestamp": str,
        "track_classifier_version": str,
        "chord_extractor_version": str,
        "quantization_grid": str  # e.g., "16th", "swing_16th"
    }
}
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pretty_midi

logger = logging.getLogger(__name__)


def extract_tempo_events(pm):
    """
    Returns a list of (time_sec, bpm) tuples.
    Works across pretty_midi versions.
    """
    if hasattr(pm, "get_tempo_changes"):
        times, bpms = pm.get_tempo_changes()
        return list(zip(times.tolist() if hasattr(times, "tolist") else times,
                        bpms.tolist() if hasattr(bpms, "tolist") else bpms))

    # ultra-defensive fallback (shouldn't happen, but keeps tests alive)
    return [(0.0, 120.0)]


def extract_tempo_single(pm):
    """Extract single BPM value from PrettyMIDI object."""
    tempo_events = extract_tempo_events(pm)
    return tempo_events[0][1] if tempo_events else 120.0


@dataclass
class NoteEvent:
    """Single note event."""
    time_seconds: float
    type: str  # "note_on" or "note_off"
    pitch: int
    velocity: Optional[int] = None
    duration: Optional[float] = None  # Only for note_on


@dataclass
class TrackInfo:
    """Information about a musical track."""
    id: int
    name: str
    is_drum: bool
    instrument: str
    classification: Dict[str, Any]
    events: List[NoteEvent]


@dataclass
class ChordEvent:
    """Chord event with timing information."""
    time_start: float
    time_end: float
    bar_start: int
    beat_start: float
    chord: Dict[str, Any]


class NomadFormatBuilder:
    """Builder for the Canonical Nomad Symbolic Format."""
    
    def __init__(self, ppq: int = 480):
        """Initialize builder.
        
        Args:
            ppq: Pulses per quarter note for quantization
        """
        self.ppq = ppq
        
    def from_midi(self, midi_path: Path) -> Dict[str, Any]:
        """Convert MIDI file to Nomad Symbolic Format.
        
        Args:
            midi_path: Path to MIDI file
            
        Returns:
            Dictionary in Nomad Symbolic Format
        """
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as e:
            raise ValueError(f"Failed to parse MIDI file {midi_path}: {e}")
        
        # Extract global information
        global_info = self._extract_global_info(midi)
        
        # Extract tracks and classify them
        tracks = self._extract_and_classify_tracks(midi)
        
        # Extract chord progression
        chords = self._extract_chords(midi, tracks)
        
        # Build final structure
        nomad_format = {
            "global": global_info,
            "tracks": tracks,
            "chords": chords,
            "metadata": {
                "source_file": str(midi_path),
                "processing_timestamp": np.datetime64('now').astype(str),
                "track_classifier_version": "v1.0",
                "chord_extractor_version": "v1.0",
                "quantization_grid": "16th"
            }
        }
        
        return nomad_format
    
    def _extract_global_info(self, midi: pretty_midi.PrettyMIDI) -> Dict[str, Any]:
        """Extract global musical information."""
        # Tempo map
        tempo_map = extract_tempo_events(midi)
        
        # Time signature map
        time_sig_map = []
        if midi.time_signature_changes:
            for ts_change in midi.time_signature_changes:
                time_sig_map.append((ts_change.time, (ts_change.numerator, ts_change.denominator)))
        else:
            time_sig_map.append((0.0, (4, 4)))  # Default time signature
        
        # Overall duration
        duration = max(
            [max([note.end for note in instr.notes]) if instr.notes else 0.0 for instr in midi.instruments]
        )
        
        return {
            "tempo_map": tempo_map,
            "time_signature_map": time_sig_map,
            "ppq": self.ppq,
            "duration_seconds": duration
        }
    
    def _get_instrument_name(self, instrument: pretty_midi.Instrument) -> str:
        """Get instrument name from pretty_midi.Instrument."""
        # Map common GM programs to names
        program_names = {
            0: "Acoustic Grand Piano",
            1: "Acoustic Bright Piano", 
            2: "Electric Grand Piano",
            3: "Honky-tonk Piano",
            4: "Electric Piano 1",
            5: "Electric Piano 2",
            6: "Harpsichord",
            7: "Clavi",
            8: "Celesta",
            9: "Glockenspiel",
            10: "Music Box",
            11: "Vibraphone",
            12: "Marimba",
            13: "Xylophone",
            14: "Tubular Bells",
            15: "Dulcimer",
            16: "Drawbar Organ",
            17: "Percussive Organ",
            18: "Rock Organ",
            19: "Church Organ",
            20: "Reed Organ",
            21: "Accordion",
            22: "Harmonica",
            23: "Tango Accordion",
            24: "Acoustic Guitar (nylon)",
            25: "Acoustic Guitar (steel)",
            26: "Electric Guitar (jazz)",
            27: "Electric Guitar (clean)",
            28: "Electric Guitar (muted)",
            29: "Overdriven Guitar",
            30: "Distortion Guitar",
            31: "Guitar harmonics",
            32: "Acoustic Bass",
            33: "Electric Bass (finger)",
            34: "Electric Bass (pick)",
            35: "Fretless Bass",
            36: "Slap Bass 1",
            37: "Slap Bass 2",
            38: "Synth Bass 1",
            39: "Synth Bass 2",
            40: "Violin",
            41: "Viola",
            42: "Cello",
            43: "Contrabass",
            44: "Tremolo Strings",
            45: "Pizzicato Strings",
            46: "Orchestral Harp",
            47: "Timpani",
            48: "String Ensemble 1",
            49: "String Ensemble 2",
            50: "SynthStrings 1",
            51: "SynthStrings 2",
            52: "Choir Aahs",
            53: "Voice Oohs",
            54: "Synth Voice",
            55: "Orchestra Hit",
            56: "Trumpet",
            57: "Trombone",
            58: "Tuba",
            59: "Muted Trumpet",
            60: "French Horn",
            61: "Brass Section",
            62: "SynthBrass 1",
            63: "SynthBrass 2",
            64: "Soprano Sax",
            65: "Alto Sax",
            66: "Tenor Sax",
            67: "Baritone Sax",
            68: "Oboe",
            69: "English Horn",
            70: "Bassoon",
            71: "Clarinet",
            72: "Piccolo",
            73: "Flute",
            74: "Recorder",
            75: "Pan Flute",
            76: "Blown Bottle",
            77: "Shakuhachi",
            78: "Whistle",
            79: "Ocarina",
            80: "Lead 1 (square)",
            81: "Lead 2 (sawtooth)",
            82: "Lead 3 (calliope)",
            83: "Lead 4 (chiff)",
            84: "Lead 5 (charang)",
            85: "Lead 6 (voice)",
            86: "Lead 7 (fifths)",
            87: "Lead 8 (bass + lead)",
            88: "Pad 1 (new age)",
            89: "Pad 2 (warm)",
            90: "Pad 3 (polysynth)",
            91: "Pad 4 (choir)",
            92: "Pad 5 (bowed)",
            93: "Pad 6 (metallic)",
            94: "Pad 7 (halo)",
            95: "Pad 8 (sweep)",
            96: "FX 1 (rain)",
            97: "FX 2 (soundtrack)",
            98: "FX 3 (crystal)",
            99: "FX 4 (atmosphere)",
            100: "FX 5 (brightness)",
            101: "FX 6 (goblins)",
            102: "FX 7 (echoes)",
            103: "FX 8 (sci-fi)",
            104: "Sitar",
            105: "Banjo",
            106: "Shamisen",
            107: "Koto",
            108: "Kalimba",
            109: "Bag pipe",
            110: "Fiddle",
            111: "Shanai",
            112: "Tinkle Bell",
            113: "Agogo",
            114: "Steel Drums",
            115: "Woodblock",
            116: "Taiko Drum",
            117: "Melodic Tom",
            118: "Synth Drum",
            119: "Reverse Cymbal",
            120: "Guitar Fret Noise",
            121: "Breath Noise",
            122: "Seashore",
            123: "Bird Tweet",
            124: "Telephone Ring",
            125: "Helicopter",
            126: "Applause",
            127: "Gunshot"
        }
        
        if instrument.is_drum:
            return "Drums"
        elif hasattr(instrument, 'program') and instrument.program in program_names:
            return program_names[instrument.program]
        else:
            return f"Program {getattr(instrument, 'program', 'Unknown')}"
    
    def _extract_and_classify_tracks(self, midi: pretty_midi.PrettyMIDI) -> List[Dict[str, Any]]:
        """Extract tracks and classify their musical role."""
        tracks = []
        
        for i, instrument in enumerate(midi.instruments):
            # Extract note events
            events = []
            for note in instrument.notes:
                # Note on
                events.append(NoteEvent(
                    time_seconds=note.start,
                    type="note_on",
                    pitch=note.pitch,
                    velocity=note.velocity,
                    duration=note.end - note.start
                ))
                # Note off
                events.append(NoteEvent(
                    time_seconds=note.end,
                    type="note_off",
                    pitch=note.pitch
                ))
            
            # Sort events by time
            events.sort(key=lambda e: e.time_seconds)
            
            # Convert to dictionaries
            track_events = []
            for event in events:
                event_dict = {
                    "time_seconds": event.time_seconds,
                    "type": event.type,
                    "pitch": event.pitch
                }
                if event.velocity is not None:
                    event_dict["velocity"] = event.velocity
                if event.duration is not None:
                    event_dict["duration"] = event.duration
                track_events.append(event_dict)
            
            # Classify track
            classification = self._classify_track(instrument, events)
            
            track_info = {
                "id": i,
                "name": instrument.name or f"Track_{i}",
                "is_drum": instrument.is_drum,
                "instrument": self._get_instrument_name(instrument),
                "classification": classification,
                "events": track_events
            }
            
            tracks.append(track_info)
        
        return tracks
    
    def _classify_track(self, instrument: pretty_midi.Instrument, events: List[NoteEvent]) -> Dict[str, Any]:
        """Classify a track's musical role."""
        if not events:
            return {
                "type": "unknown",
                "confidence": 0.0,
                "reasons": ["No events"]
            }
        
        reasons = []
        confidence = 0.0
        
        # Drum classification
        if instrument.is_drum:
            return {
                "type": "drums",
                "confidence": 1.0,
                "reasons": ["GM percussion channel"]
            }
        
        # Extract pitch statistics
        pitches = [e.pitch for e in events if e.type == "note_on"]
        if not pitches:
            return {
                "type": "unknown",
                "confidence": 0.0,
                "reasons": ["No note_on events"]
            }
        
        avg_pitch = np.mean(pitches)
        filtered_pitches = [pitch for pitch in pitches if pitch < 127]  # Avoid drum notes
        max_pitch = np.max(filtered_pitches) if filtered_pitches else 0
        
        # Polyphony analysis
        note_ons = [e for e in events if e.type == "note_on"]
        simultaneous_notes = self._count_simultaneous_notes(note_ons)
        avg_polyphony = np.mean(list(simultaneous_notes.values())) if simultaneous_notes else 1.0
        
        # Bass classification
        if max_pitch < 48:  # C3 and below
            confidence += 0.7
            reasons.append(f"Low register (max pitch: {max_pitch})")
        
        # Melody classification  
        if avg_pitch > 60:  # Above middle C
            confidence += 0.6
            reasons.append(f"High register (avg pitch: {avg_pitch:.1f})")
            
        # Polyphonic tracks likely contain chords
        if avg_polyphony > 1.5:
            confidence += 0.5
            reasons.append(f"Polyphonic (avg: {avg_polyphony:.1f} simultaneous notes)")
        
        # Determine final classification
        if confidence > 0.6:
            if max_pitch < 48:
                track_type = "bass"
            elif avg_polyphony > 1.5:
                track_type = "chords"
            else:
                track_type = "melody"
        elif avg_polyphony > 1.5:
            track_type = "chords"
        elif max_pitch < 48:
            track_type = "bass"
        else:
            track_type = "melody"
        
        return {
            "type": track_type,
            "confidence": min(confidence, 1.0),
            "reasons": reasons
        }
    
    def _count_simultaneous_notes(self, note_ons: List[NoteEvent]) -> Dict[float, int]:
        """Count simultaneous notes at each time point."""
        time_counts = {}
        for event in note_ons:
            # Round to nearest millisecond to group simultaneous notes
            time_key = round(event.time_seconds * 1000) / 1000.0
            time_counts[time_key] = time_counts.get(time_key, 0) + 1
        return time_counts
    
    def _extract_chords(self, midi: pretty_midi.PrettyMIDI, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract chord progression from polyphonic tracks."""
        chords = []
        
        # Find chord tracks
        chord_tracks = [t for t in tracks if t["classification"]["type"] == "chords"]
        if not chord_tracks:
            # Fallback: analyze all polyphonic tracks
            all_tracks = [t for t in tracks if t["classification"]["confidence"] > 0.3]
            polyphonic_tracks = []
            for track in all_tracks:
                events = track["events"]
                note_ons = [e for e in events if e["type"] == "note_on"]
                if len(note_ons) > 1:
                    # Check for simultaneous notes
                    time_groups = {}
                    for event in note_ons:
                        time_key = round(event["time_seconds"] * 1000) / 1000.0
                        if time_key not in time_groups:
                            time_groups[time_key] = []
                        time_groups[time_key].append(event)
                    
                    # If any time has more than 1 note, it's polyphonic
                    if any(len(group) > 1 for group in time_groups.values()):
                        polyphonic_tracks.append(track)
            
            chord_tracks = polyphonic_tracks
        
        if not chord_tracks:
            return chords  # No chord tracks found
        
        # Process chord track (use first one for now)
        chord_track = chord_tracks[0]
        
        # Get time signature for bar/beat calculation
        time_sig_map = {}
        if midi.time_signature_changes:
            for ts in midi.time_signature_changes:
                time_sig_map[ts.time] = (ts.numerator, ts.denominator)
        
        # Analyze in beat windows
        tempo_map = extract_tempo_events(midi)
        
        # Define window size (1 beat or 1 bar)
        window_size = self._get_window_size(midi, time_sig_map)
        
        events = chord_track["events"]
        note_ons = [e for e in events if e["type"] == "note_on"]
        
        if not note_ons:
            return chords
        
        # Group events into time windows
        windows = self._group_into_windows(note_ons, window_size)
        
        for window in windows:
            chord_info = self._analyze_window_chord(window, time_sig_map, tempo_map)
            if chord_info:
                chords.append(chord_info)
        
        return chords
    
    def _get_window_size(self, midi: pretty_midi.PrettyMIDI, time_sig_map: Dict) -> float:
        """Get chord analysis window size in seconds."""
        # Use 1 beat as default window
        default_tempo = extract_tempo_single(midi)
        return 60.0 / default_tempo  # 1 beat in seconds
    
    def _group_into_windows(self, note_ons: List[Dict], window_size: float) -> List[List[Dict]]:
        """Group note events into time windows."""
        if not note_ons:
            return []
        
        windows = []
        current_window = []
        window_start = note_ons[0]["time_seconds"]
        
        for event in note_ons:
            if event["time_seconds"] - window_start >= window_size:
                if current_window:
                    windows.append(current_window)
                current_window = [event]
                window_start = event["time_seconds"]
            else:
                current_window.append(event)
        
        if current_window:
            windows.append(current_window)
        
        return windows
    
    def _analyze_window_chord(self, window: List[Dict], time_sig_map: Dict, tempo_map: Dict) -> Optional[Dict[str, Any]]:
        """Analyze chord in a time window."""
        if not window:
            return None
        
        # Get pitches active in this window
        pitches = [event["pitch"] for event in window]
        if not pitches:
            return None
        
        # Get time information
        time_start = window[0]["time_seconds"]
        time_end = max(event["time_seconds"] + event.get("duration", 0) for event in window)
        
        # Convert to bar/beat
        bar_start, beat_start = self._time_to_bar_beat(time_start, time_sig_map, tempo_map)
        
        # Analyze chord
        chord_info = self._pitch_set_to_chord(pitches)
        if not chord_info:
            return None
        
        return {
            "time_start": time_start,
            "time_end": time_end,
            "bar_start": bar_start,
            "beat_start": beat_start,
            "chord": chord_info
        }
    
    def _time_to_bar_beat(self, time_seconds: float, time_sig_map: Dict, tempo_map: Dict) -> Tuple[int, float]:
        """Convert time in seconds to bar and beat number."""
        # Simplified conversion - assumes constant tempo and time signature
        current_time = 0.0
        current_bar = 0
        current_beat = 0.0
        
        # Use default values if no changes
        tempo = 120.0
        time_sig = (4, 4)
        
        if tempo_map:
            tempo = tempo_map[0][1]
        if time_sig_map:
            time_sig = list(time_sig_map.values())[0]
        
        beat_duration = 60.0 / tempo
        bar_duration = beat_duration * time_sig[0]  # numerator beats per bar
        
        current_bar = int(time_seconds // bar_duration)
        beat_in_bar = (time_seconds % bar_duration) / beat_duration
        current_beat = beat_in_bar + 1  # 1-indexed
        
        return current_bar, current_beat
    
    def _pitch_set_to_chord(self, pitches: List[int]) -> Optional[Dict[str, Any]]:
        """Convert pitch set to chord information."""
        if not pitches:
            return None
        
        # Convert to pitch classes (0-11)
        pitch_classes = sorted(set(pitch % 12 for pitch in pitches))
        
        # Simple chord recognition
        # This is a simplified version - a full implementation would be more sophisticated
        if len(pitch_classes) < 2:
            return None
        
        # Try to match against common chord templates
        templates = {
            (0, 4, 7): {"root": "C", "quality": "maj"},
            (0, 3, 7): {"root": "C", "quality": "min"},
            (0, 4, 7, 10): {"root": "C", "quality": "dom7"},
            (0, 3, 7, 10): {"root": "C", "quality": "min7"},
            (0, 4, 7, 11): {"root": "C", "quality": "maj7"},
        }
        
        for template, chord_info in templates.items():
            if self._matches_template(pitch_classes, template):
                # Adjust root based on actual pitch classes
                root_pc = pitch_classes[0]
                chord_info["root"] = self._pitch_class_to_note_name(root_pc)
                chord_info["confidence"] = 0.7  # Simplified confidence
                chord_info["inversion"] = 0
                return chord_info
        
        # No match found
        return None
    
    def _matches_template(self, pitch_classes: List[int], template: Tuple[int, ...]) -> bool:
        """Check if pitch classes match a chord template."""
        if len(pitch_classes) != len(template):
            return False
        
        # Try all rotations (inversions)
        for i in range(len(template)):
            rotated_template = tuple((template[j] - template[0]) % 12 for j in range(len(template)))
            if pitch_classes == sorted(rotated_template):
                return True
            
            # Rotate template
            template = tuple((template[j] - template[1]) % 12 for j in range(len(template)))
        
        return False
    
    def _pitch_class_to_note_name(self, pitch_class: int) -> str:
        """Convert pitch class (0-11) to note name."""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return note_names[pitch_class % 12]
    
    def save(self, data: Dict[str, Any], path: Path):
        """Save Nomad Symbolic Format to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Nomad format saved to {path}")
    
    def load(self, path: Path) -> Dict[str, Any]:
        """Load Nomad Symbolic Format from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Nomad format loaded from {path}")
        return data


def create_nomad_format(midi_path: Path, output_path: Optional[Path] = None) -> Dict[str, Any]:
    """Convenience function to create Nomad Symbolic Format from MIDI.
    
    Args:
        midi_path: Path to MIDI file
        output_path: Optional path to save the format
        
    Returns:
        Nomad Symbolic Format dictionary
    """
    builder = NomadFormatBuilder()
    format_data = builder.from_midi(midi_path)
    
    if output_path:
        builder.save(format_data, output_path)
    
    return format_data