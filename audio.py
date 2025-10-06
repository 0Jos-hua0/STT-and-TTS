import time
import numpy as np
import sounddevice as sd
from typing import Callable, Optional

# Audio recording parameters
SAMPLE_RATE = 16000  # Whisper model expects 16kHz
CHANNELS = 1
DTYPE = 'int16'
ACTIVATION_THRESHOLD = 0.05
SILENCE_TIMEOUT = 2.0


def record_audio(sample_rate: int = SAMPLE_RATE,
                channels: int = CHANNELS,
                dtype: str = DTYPE,
                max_duration: float = 10.0) -> Optional[np.ndarray]:
    """
    Records audio from the microphone, waiting for speech activation.

    Args:
        sample_rate: Sample rate for recording
        channels: Number of audio channels
        dtype: Data type for audio samples
        max_duration: Maximum recording duration in seconds

    Returns:
        Recorded audio data as numpy array, or None if no speech detected
    """
    print("🎤 Listening... Speak now.")
    recording = []
    is_speaking = False
    silence_start_time = None
    start_time = time.time()
    chunk_duration = 0.1  # seconds per chunk

    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype=dtype) as stream:
        try:
            while True:
                # Check for maximum recording duration
                if time.time() - start_time > max_duration:
                    print(f"⏱️  Maximum recording duration ({max_duration}s) reached.")
                    break

                audio_chunk, overflowed = stream.read(int(sample_rate * chunk_duration))
                if overflowed:
                    print("⚠️  Audio buffer overflowed, some audio may be lost.")

                volume = np.linalg.norm(audio_chunk) * 10 / (2**15)

                if volume > ACTIVATION_THRESHOLD:
                    if not is_speaking:
                        is_speaking = True
                        print("🗣️  Speech detected. Recording...")
                        start_time = time.time()  # Reset timer when speech starts
                    silence_start_time = None
                elif is_speaking:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif time.time() - silence_start_time > SILENCE_TIMEOUT:
                        print("👂  Silence detected. Stopping recording.")
                        break

                if is_speaking:
                    recording.append(audio_chunk)

        except KeyboardInterrupt:
            print("\nRecording stopped by user.")
            return None

    if not recording:
        print("No speech detected.")
        return None

    print(f"✅ Recorded {len(recording) * chunk_duration:.1f} seconds of audio.")
    return np.concatenate(recording, axis=0)


def record_audio_stream(stop_event: Callable[[], bool] = lambda: False,
                       sample_rate: int = SAMPLE_RATE,
                       channels: int = CHANNELS,
                       dtype: str = DTYPE) -> Optional[np.ndarray]:
    """
    Records audio from the microphone with streaming capability.

    Args:
        stop_event: Function that returns True when recording should stop
        sample_rate: Sample rate for recording
        channels: Number of audio channels
        dtype: Data type for audio samples

    Returns:
        Recorded audio data as numpy array, or None if stopped early
    """
    print("🎤 Recording started...")
    recording = []
    silence_start_time = None
    start_time = time.time()
    chunk_duration = 0.1

    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype=dtype) as stream:
        while not stop_event():
            if time.time() - start_time > 10.0:  # Max 10 seconds
                print("⏱️  Maximum recording duration reached.")
                break

            audio_chunk, overflowed = stream.read(int(sample_rate * chunk_duration))
            if overflowed:
                print("⚠️  Audio buffer overflowed")

            volume = np.linalg.norm(audio_chunk) * 10 / (2**15)

            if volume > ACTIVATION_THRESHOLD:
                if silence_start_time is None:
                    print("🗣️  Speech detected")
                silence_start_time = None
            else:
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > SILENCE_TIMEOUT:
                    print("👂  Silence detected, stopping...")
                    break

            recording.append(audio_chunk)

    if not recording:
        print("No speech detected.")
        return None

    print(f"✅ Recorded {len(recording) * chunk_duration:.1f} seconds of audio")
    return np.concatenate(recording, axis=0)
