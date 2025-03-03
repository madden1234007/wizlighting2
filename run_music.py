#!/usr/bin/env python
# run_music.py - Enhanced script for music synchronization with caching

import sys
import os
import time
import logging
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MusicRunner")

# Import local modules
from wiz_core import WizController
from wiz_music import MusicSyncController, audio_libraries_available
from wiz_config import get_light_group, get_color

# Import pydub for direct audio testing
try:
    from pydub import AudioSegment
    from pydub.playback import play
    pydub_available = True
except ImportError:
    pydub_available = False
    logger.warning("PyDub not available")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run music synchronization with WiZ lights')
    
    # Required argument
    parser.add_argument('audio_file', help='Path to the audio file')
    
    # Optional arguments
    parser.add_argument('--group', '-g', default='testing', help='Light group name (default: testing)')
    parser.add_argument('--mode', '-m', default='beat', choices=['beat', 'spectral', 'frequency'], 
                        help='Sync mode (default: beat)')
    parser.add_argument('--pattern', '-p', default='pulse', choices=['pulse', 'on_off'], 
                        help='Beat pattern (default: pulse)')
    parser.add_argument('--brightness', '-b', type=int, default=100, 
                        help='Brightness level 1-100 (default: 100)')
    
    # Cache options
    parser.add_argument('--no-cache', action='store_true', 
                        help='Disable caching of audio analysis')
    parser.add_argument('--force-analyze', action='store_true', 
                        help='Force fresh analysis even if cache exists')
    parser.add_argument('--cache-dir', default='analysis_cache', 
                        help='Directory for cached analyses (default: analysis_cache)')
    
    # Color options
    parser.add_argument('--color', default='white', 
                        help='Color name for beats (from wiz_config colors)')
    parser.add_argument('--base-color', default='blue', 
                        help='Base color name for pulse pattern')
                        
    return parser.parse_args()

def verify_audio_file(file_path):
    """Test if audio file can be loaded and played"""
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        if not pydub_available:
            logger.warning("PyDub not available, skipping audio verification")
            return True
            
        # Try to load the audio file
        logger.info(f"Testing audio file: {file_path}")
        audio = AudioSegment.from_file(file_path)
        logger.info(f"Audio loaded successfully: {len(audio)/1000}s, {audio.channels} channels, {audio.frame_rate}Hz")
        
        return True
    except Exception as e:
        logger.error(f"Error testing audio file: {e}")
        return False

def verify_lights(controller, light_ips):
    """Test if we can connect to and control the lights"""
    try:
        logger.info(f"Testing connection to {len(light_ips)} lights...")
        results = []
        
        # Try to get the state of each light
        for ip in light_ips:
            logger.info(f"Testing light: {ip}")
            result = controller.get_state(ip)
            if result.get("status") == "timeout" or result.get("status") == "error":
                logger.warning(f"Could not connect to light: {ip}, response: {result}")
            else:
                logger.info(f"Light {ip} responded: {result}")
            results.append({"ip": ip, "result": result})
        
        # Check if we got any successful responses
        success_count = sum(1 for r in results if not (r["result"].get("status") in ["timeout", "error"]))
        logger.info(f"Successfully connected to {success_count}/{len(light_ips)} lights")
        
        if success_count > 0:
            return True
        else:
            logger.error("Could not connect to any lights")
            return False
    except Exception as e:
        logger.error(f"Error testing lights: {e}")
        return False

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Get light IPs
    light_ips = get_light_group(args.group)
    if not light_ips:
        logger.error(f"Light group '{args.group}' not found")
        sys.exit(1)
    
    logger.info(f"Starting with: File={args.audio_file}, Group={args.group}, "
               f"Mode={args.mode}, Pattern={args.pattern}, Cache={'disabled' if args.no_cache else 'enabled'}")
    
    # Create controller
    controller = WizController()
    
    # Verify lights
    logger.info("Verifying light connectivity...")
    if not verify_lights(controller, light_ips):
        logger.warning("Light verification failed, but continuing...")
    
    # Verify audio file
    logger.info("Verifying audio file...")
    if not verify_audio_file(args.audio_file):
        logger.error("Audio file verification failed")
        sys.exit(1)
    
    # Create music sync controller with cache options
    sync = MusicSyncController(
        controller=controller,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir
    )
    
    # Check requirements
    if not audio_libraries_available:
        logger.error("Audio libraries not available. Cannot sync to music.")
        logger.error("Please install numpy, pydub, and librosa for full music sync functionality.")
        sys.exit(1)
    
    # Configure sync parameters based on mode and pattern
    kwargs = {
        'force_analyze': args.force_analyze,
        'brightness': args.brightness
    }
    
    if args.mode == "beat":
        # Get colors from config or use defaults
        on_color = get_color(args.color) or {"r": 255, "g": 255, "b": 255}
        base_color = get_color(args.base_color) or {"r": 0, "g": 0, "b": 100}
        
        if args.pattern == "on_off":
            kwargs.update({
                "use_on_off": True,
                "on_color": on_color
            })
        else:  # pulse pattern
            kwargs.update({
                "use_on_off": False,
                "base_color": base_color,
                "beat_color": on_color
            })
    
    # Play with light sync
    logger.info(f"Playing {args.audio_file} with {args.mode} light sync on group '{args.group}' "
               f"using {args.pattern} pattern")
    logger.info(f"Sync parameters: {kwargs}")
    
    # Find the playback method - handle both "_run_playback" and "run_playback" method names
    method_name = None
    for name in dir(sync):
        if name.endswith("run_playback"):
            method_name = name
            break
    
    if not method_name:
        logger.error("Could not find run_playback method in MusicSyncController")
        sys.exit(1)
    
    # Start playback
    logger.info(f"Using playback method: {method_name}")
    sync.running = True
    method = getattr(sync, method_name)
    
    import threading
    thread = threading.Thread(
        target=method,
        args=(args.audio_file, light_ips, args.mode),
        kwargs=kwargs
    )
    thread.daemon = True
    thread.start()
    sync.active_threads.append(thread)
    
    # Keep the script running
    logger.info("Music sync started. Press Ctrl+C to stop")
    try:
        while sync.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logger.info("Stopping on user request")
        sync.stop()
    finally:
        logger.info("Turning lights off")
        for ip in light_ips:
            controller.turn_off(ip)
        logger.info("Done")

if __name__ == "__main__":
    main()