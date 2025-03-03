# test_audio.py
from pydub import AudioSegment
from pydub.playback import play

# Set ffmpeg path if needed
# AudioSegment.converter = r"C:\path\to\ffmpeg.exe"

try:
    print("Loading audio file...")
    audio = AudioSegment.from_file("15E1.mp3")
    print(f"Audio loaded: {len(audio)/1000}s, {audio.channels} channels")
    print("Playing a small segment...")
    play(audio[:5000])  # Play the first 5 seconds
    print("Success!")
except Exception as e:
    print(f"Error: {e}")