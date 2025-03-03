#!/usr/bin/env python
# wiz_music.py - Music synchronization for WizLighting

import time
import threading
import logging
import os
import sys
import importlib.util
from audio_analysis_cache import AudioAnalysisCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WizMusic")

# Try to import optional audio libraries
audio_libraries_available = False

try:
    import numpy as np
    audio_numpy_available = True
except ImportError:
    audio_numpy_available = False
    logger.warning("NumPy not available, advanced audio processing will be limited")

# Conditional imports for audio processing
try:
    from pydub import AudioSegment
    from pydub.playback import play as pydub_play
    audio_pydub_available = True
except ImportError:
    audio_pydub_available = False
    logger.warning("PyDub not available, audio file playback will be disabled")

try:
    import librosa
    audio_librosa_available = True
except ImportError:
    audio_librosa_available = False
    logger.warning("Librosa not available, advanced audio analysis will be disabled")

# Set overall audio availability flag
audio_libraries_available = audio_numpy_available and audio_pydub_available and audio_librosa_available

# Add temp directory fix
import tempfile
from pathlib import Path
temp_dir = Path(__file__).parent / "temp"
temp_dir.mkdir(exist_ok=True)
tempfile.tempdir = str(temp_dir)
os.environ["TEMP"] = str(temp_dir)
os.environ["TMP"] = str(temp_dir)

# Import local modules
from wiz_core import WizController
import wiz_patterns as patterns

class SimpleBeatDetector:
    """Simple beat detector that works without external libraries"""
    
    def __init__(self, threshold=1.2, min_interval=0.2):
        """Initialize the simple beat detector
        
        Args:
            threshold: Energy threshold to detect a beat
            min_interval: Minimum time between beats in seconds
        """
        self.threshold = threshold
        self.min_interval = min_interval
        self.last_beat_time = 0
        self.energy_history = []
        self.max_history = 10
    
    def detect_beat(self, energy):
        """Detect if the current energy level represents a beat
        
        Args:
            energy: Current audio energy level
        
        Returns:
            bool: True if a beat was detected, False otherwise
        """
        current_time = time.time()
        
        # Make sure enough time has passed since the last beat
        if current_time - self.last_beat_time < self.min_interval:
            return False
        
        # Add energy to history and trim if necessary
        self.energy_history.append(energy)
        if len(self.energy_history) > self.max_history:
            self.energy_history.pop(0)
        
        # Calculate average energy
        if len(self.energy_history) < 2:
            return False
        
        avg_energy = sum(self.energy_history) / len(self.energy_history)
        
        # Check if current energy is above threshold
        if energy > avg_energy * self.threshold:
            self.last_beat_time = current_time
            return True
        
        return False

class MusicSyncController:
    """Controller for syncing lights to music"""
    
    def __init__(self, controller=None, use_cache=True, cache_dir="analysis_cache"):
        """Initialize the music sync controller
        
        Args:
            controller: WizController instance (creates a new one if None)
            use_cache: Whether to use caching for audio analysis
            cache_dir: Directory for cache files
        """
        self.controller = controller if controller else WizController()
        self.running = False
        self.beat_detector = SimpleBeatDetector()
        self.active_threads = []
        
        # Initialize cache
        self.use_cache = use_cache
        if use_cache:
            self.cache = AudioAnalysisCache(cache_dir)
        else:
            self.cache = None
    
    def extract_features_with_cache(self, file_path, audio_data, sr, frame_length=2048, hop_length=512):
        """Extract audio features using librosa with caching
        
        Args:
            file_path: Path to the audio file (for cache key)
            audio_data: Audio samples as numpy array
            sr: Sample rate
            frame_length: Frame length for STFT
            hop_length: Hop length for STFT
        
        Returns:
            dict: Dictionary of audio features
        """
        # Check if we can use cache
        if self.use_cache and self.cache and file_path:
            # Try to load from cache
            features = self.cache.load_analysis(file_path)
            if features:
                logger.info("Using cached audio analysis")
                return features
        
        # No cache available, run the full analysis
        logger.info("Performing full audio analysis (this may take a while)...")
        
        # Add simple progress indicators
        print("Analyzing audio: ", end="", flush=True)
        
        # Spectral centroid (brightness)
        print(".", end="", flush=True)
        cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=hop_length)[0]
        
        # Energy in frequency bands
        print(".", end="", flush=True)
        spec = np.abs(librosa.stft(audio_data, n_fft=frame_length, hop_length=hop_length))
        
        # Define frequency bands (in Hz)
        bands = {
            'bass': (20, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'high': (4000, sr//2)
        }
        
        # Convert frequency to spectrogram bin indices
        band_energies = {}
        for i, (name, (low, high)) in enumerate(bands.items()):
            if i % 2 == 0:
                print(".", end="", flush=True)
            low_bin = max(0, int(low * frame_length / sr))
            high_bin = min(spec.shape[0], int(high * frame_length / sr))
            band_energies[name] = np.sum(spec[low_bin:high_bin, :], axis=0)
        
        print(" Done!")
        
        features = {
            'centroid': librosa.frames_to_time(np.arange(len(cent)), sr=sr, hop_length=hop_length),
            'cent_values': cent,
            'times': librosa.frames_to_time(np.arange(spec.shape[1]), sr=sr, hop_length=hop_length),
            'bands': band_energies
        }
        
        # Save to cache if enabled
        if self.use_cache and self.cache and file_path:
            self.cache.save_analysis(file_path, features)
        
        return features
    
    def check_requirements(self):
        """Check if required libraries are available
        
        Returns:
            bool: True if all required libraries are available, False otherwise
        """
        return audio_libraries_available
    
    def play_with_light_sync(self, file_path, ips, mode="beat", **kwargs):
        """Play an audio file and sync lights to it
        
        Args:
            file_path: Path to the audio file
            ips: List of IP addresses
            mode: Sync mode ('beat', 'spectral', or 'frequency')
            **kwargs: Additional arguments for the sync mode
        
        Returns:
            bool: True if playback started successfully, False otherwise
        """
        if not audio_libraries_available:
            logger.error("Audio libraries not available. Cannot sync to music.")
            return False
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        # Stop any existing playback
        self.stop()
        
        # Start new playback in a separate thread
        self.running = True
        playback_thread = threading.Thread(
            target=self._run_playback,
            args=(file_path, ips, mode),
            kwargs=kwargs
        )
        playback_thread.daemon = True
        playback_thread.start()
        self.active_threads.append(playback_thread)
        
        return True
    
    def stop(self):
        """Stop any active playback and light sync"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.active_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        self.active_threads.clear()
        
        return True
    
    def _run_playback(self, file_path, ips, mode="beat", **kwargs):
        """Run audio playback with light sync
        
        Args:
            file_path: Path to the audio file
            ips: List of IP addresses
            mode: Sync mode ('beat', 'spectral', or 'frequency')
            **kwargs: Additional arguments for the sync mode
        """
        play_thread = None
        
        try:
            # Check if we should force a fresh analysis
            force_analyze = kwargs.pop('force_analyze', False)
            if force_analyze and self.use_cache and self.cache:
                self.cache.clear_cache(file_path)
                logger.info("Forcing fresh analysis, cleared cache")
            
            # Load the audio file
            logger.info(f"Loading audio file: {file_path}")
            
            audio = AudioSegment.from_file(file_path)
            logger.info(f"Audio loaded: {len(audio)/1000}s, {audio.channels} channels")
            
            # Convert stereo to mono if needed
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            samples /= np.max(np.abs(samples))  # Normalize
            
            # Extract audio features with caching
            logger.info("Analyzing audio features...")
            sr = audio.frame_rate
            features = self.extract_features_with_cache(file_path, samples, sr)
            
            # Play audio in a separate thread
            logger.info("Starting playback with light sync...")
            
            def play_audio():
                try:
                    pydub_play(audio)
                    logger.info("Playback finished")
                except Exception as e:
                    logger.error(f"Error during audio playback: {e}")
                finally:
                    self.running = False
            
            play_thread = threading.Thread(target=play_audio)
            play_thread.daemon = True
            play_thread.start()
            
            # Sync lights
            start_time = time.time()
            current_time = 0
            
            while self.running and current_time < features['times'][-1]:
                current_time = time.time() - start_time
                
                # Find the closest time index
                time_idx = np.argmin(np.abs(features['times'] - current_time))
                
                # Apply the appropriate sync mode
                self._apply_sync_mode(mode, ips, features, time_idx, **kwargs)
                
                # Sleep a short time to not overwhelm the lights
                time.sleep(0.05)
            
        except Exception as e:
            logger.error(f"Error during playback: {e}")
        finally:
            self.running = False
            # Wait for playback thread to finish if it exists
            if play_thread and play_thread.is_alive():
                play_thread.join(timeout=1.0)

    def _extract_features(self, audio_data, sr, frame_length=2048, hop_length=512):
        """Extract audio features using librosa
        
        Args:
            audio_data: Audio samples as numpy array
            sr: Sample rate
            frame_length: Frame length for STFT
            hop_length: Hop length for STFT
        
        Returns:
            dict: Dictionary of audio features
        """
        # Spectral centroid (brightness)
        cent = librosa.feature.spectral_centroid(y=audio_data, sr=sr, hop_length=hop_length)[0]
        
        # Energy in frequency bands
        spec = np.abs(librosa.stft(audio_data, n_fft=frame_length, hop_length=hop_length))
        
        # Define frequency bands (in Hz)
        bands = {
            'bass': (20, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'high_mid': (2000, 4000),
            'high': (4000, sr//2)
        }
        
        # Convert frequency to spectrogram bin indices
        band_energies = {}
        for name, (low, high) in bands.items():
            low_bin = max(0, int(low * frame_length / sr))
            high_bin = min(spec.shape[0], int(high * frame_length / sr))
            band_energies[name] = np.sum(spec[low_bin:high_bin, :], axis=0)
        
        return {
            'centroid': librosa.frames_to_time(np.arange(len(cent)), sr=sr, hop_length=hop_length),
            'cent_values': cent,
            'times': librosa.frames_to_time(np.arange(spec.shape[1]), sr=sr, hop_length=hop_length),
            'bands': band_energies
        }
    
    def _apply_sync_mode(self, mode, ips, features, time_idx, **kwargs):
        """Apply the appropriate sync mode based on audio features
        
        Args:
            mode: Sync mode ('beat', 'spectral', or 'frequency')
            ips: List of IP addresses
            features: Audio features dictionary
            time_idx: Current time index
            **kwargs: Additional arguments for the sync mode
        """
        if mode == "beat":
            self._apply_beat_sync(ips, features, time_idx, **kwargs)
        elif mode == "spectral":
            self._apply_spectral_sync(ips, features, time_idx, **kwargs)
        elif mode == "frequency":
            self._apply_frequency_sync(ips, features, time_idx, **kwargs)
        else:
            logger.warning(f"Unknown sync mode: {mode}")
    
    def _apply_beat_sync(self, ips, features, time_idx, **kwargs):
        """Apply beat-driven light synchronization
        
        Args:
            ips: List of IP addresses
            features: Audio features dictionary
            time_idx: Current time index
            **kwargs: Additional arguments for beat sync
        """
        # Get bass energy for beat detection
        bass = features['bands']['bass'][time_idx]
        avg_bass = np.mean(features['bands']['bass'])
        
        # Get settings from kwargs
        use_on_off = kwargs.get('use_on_off', False)
        on_color = kwargs.get('on_color', {"r": 255, "g": 255, "b": 255})
        base_color = kwargs.get('base_color', {"r": 0, "g": 0, "b": 100})
        beat_color = kwargs.get('beat_color', {"r": 255, "g": 255, "b": 255})
        brightness = kwargs.get('brightness', 100)
        colors = kwargs.get('colors', None)  # Optional list of colors to cycle through
        
        # Check for beat
        if self.beat_detector.detect_beat(bass):
            if use_on_off:
                # Simple on/off pattern
                self.controller.set_group_color(ips, on_color, brightness)
                time.sleep(0.1)  # Brief flash
                # Properly turn off lights
                for ip in ips:
                    self.controller.turn_off(ip)
            elif colors:
                # Cycle through colors on each beat
                color_idx = int(time.time()) % len(colors)
                patterns.beat_pulse(self.controller, ips, base_color, colors[color_idx], brightness)
            else:
                # Use single beat color
                patterns.beat_pulse(self.controller, ips, base_color, beat_color, brightness)
    def _apply_spectral_sync(self, ips, features, time_idx, **kwargs):
        """Apply spectral centroid-driven light synchronization
        
        Args:
            ips: List of IP addresses
            features: Audio features dictionary
            time_idx: Current time index
            **kwargs: Additional arguments for spectral sync
        """
        # Find spectral centroid value at current time
        cent_idx = np.argmin(np.abs(features['centroid'] - features['times'][time_idx]))
        cent_val = features['cent_values'][cent_idx]
        
        # Normalize to 0-1 range
        cent_min = np.min(features['cent_values'])
        cent_max = np.max(features['cent_values'])
        cent_norm = (cent_val - cent_min) / (cent_max - cent_min) if cent_max > cent_min else 0.5
        
        # Calculate brightness based on overall energy
        energy = np.mean([features['bands'][band][time_idx] for band in features['bands']])
        max_energy = np.max([np.max(features['bands'][band]) for band in features['bands']])
        brightness = int((energy / max_energy) * 100) if max_energy > 0 else 50
        brightness = kwargs.get('brightness', max(30, brightness))
        
        # Map to color using spectral centroid
        patterns.spectral_color_map(self.controller, ips, cent_norm, brightness)
    
    def _apply_frequency_sync(self, ips, features, time_idx, **kwargs):
        """Apply frequency band-driven light synchronization
        
        Args:
            ips: List of IP addresses
            features: Audio features dictionary
            time_idx: Current time index
            **kwargs: Additional arguments for frequency sync
        """
        # Get energy in each frequency band
        bass = features['bands']['bass'][time_idx]
        mid = features['bands']['mid'][time_idx]
        high = features['bands']['high'][time_idx]
        
        # Normalize
        bass_max = np.max(features['bands']['bass'])
        mid_max = np.max(features['bands']['mid'])
        high_max = np.max(features['bands']['high'])
        
        bass_norm = bass / bass_max if bass_max > 0 else 0
        mid_norm = mid / mid_max if mid_max > 0 else 0
        high_norm = high / high_max if high_max > 0 else 0
        
        # Apply frequency-based color mapping
        brightness = kwargs.get('brightness', None)  # Use None to let the function calculate brightness
        patterns.frequency_color_map(self.controller, ips, bass_norm, mid_norm, high_norm, brightness)

# Simple function to list audio files
def list_audio_files(directory="."):
    """List all audio files in a directory
    
    Args:
        directory: Directory path to search
    
    Returns:
        list: List of audio file paths
    """
    audio_exts = [".mp3", ".wav", ".ogg", ".flac", ".m4a"]
    files = []
    
    if os.path.exists(directory):
        for file in os.listdir(directory):
            if any(file.lower().endswith(ext) for ext in audio_exts):
                files.append(os.path.join(directory, file))
    
    return files

# Factory function to create controller with appropriate backend
def create_music_sync_controller():
    """Create a music sync controller with the appropriate backend
    
    Returns:
        MusicSyncController: Configured controller or None if requirements not met
    """
    controller = None
    
    if audio_libraries_available:
        controller = MusicSyncController()
    else:
        logger.warning(
            "Audio processing libraries not available. "
            "Install numpy, pydub, and librosa for full music sync functionality."
        )
    
    return controller
