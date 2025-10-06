import time
import os
import io
import subprocess
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import whisper
from piper.voice import PiperVoice 
from pydantic import BaseModel, Field

# --- CONFIGURATION ---
# NOTE: The user must download the Whisper 'small' model and a Piper ONNX voice file separately.

# 1. Speech-to-Text (STT) Configuration
WHISPER_MODEL_NAME = "small"
# Audio recording parameters
SAMPLE_RATE = 16000  # Whisper model expects 16kHz
CHANNELS = 1
DTYPE = 'int16'
# Simple activation: starts recording when loudness exceeds this threshold
ACTIVATION_THRESHOLD = 0.05
# Time in seconds after speaking to wait before stopping recording
SILENCE_TIMEOUT = 2.0

# 2. Text-to-Speech (TTS) Configuration
# --- IMPORTANT: REPLACE WITH YOUR DOWNLOADED MODEL PATHS ---
# Update paths to point to your downloaded model files
PIPER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "en_US-amy-medium.onnx")
PIPER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "en_US-amy-medium.json")
OUTPUT_FILE = "temp_output.wav"
STOP_WORD = "quit"  # Word to exit the loop


def record_audio(sample_rate, channels, dtype, max_duration=10.0):
    """Records audio from the microphone, waiting for speech activation."""
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
                        print(f"👂  Silence detected. Stopping recording.")
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

import time
import os
import io
import subprocess
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wavfile
import whisper
from piper.voice import PiperVoice
from pydantic import BaseModel, Field

# --- CONFIGURATION ---
# NOTE: The user must download the Whisper 'small' model and a Piper ONNX voice file separately.

# 1. Speech-to-Text (STT) Configuration
WHISPER_MODEL_NAME = "small"
# Audio recording parameters
SAMPLE_RATE = 16000  # Whisper model expects 16kHz
CHANNELS = 1
DTYPE = 'int16'
# Simple activation: starts recording when loudness exceeds this threshold
ACTIVATION_THRESHOLD = 0.05
# Time in seconds after speaking to wait before stopping recording
SILENCE_TIMEOUT = 2.0

# 2. Text-to-Speech (TTS) Configuration
# --- IMPORTANT: REPLACE WITH YOUR DOWNLOADED MODEL PATHS ---
# Update paths to point to your downloaded model files
PIPER_MODEL_PATH = os.path.join(os.path.dirname(__file__), "en_US-amy-medium.onnx")
PIPER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "en_US-amy-medium.json")
OUTPUT_FILE = "temp_output.wav"
STOP_WORD = "quit"  # Word to exit the loop


def record_audio(sample_rate, channels, dtype, max_duration=10.0):
    """Records audio from the microphone, waiting for speech activation."""
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
                        print(f"👂  Silence detected. Stopping recording.")
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


def speech_to_text(audio_data, whisper_model):
    """
    Transcribes the recorded audio using the local Whisper model.
    Returns the transcribed text or empty string if no speech detected.
    """
    print("🧠 Transcribing audio with Whisper...")

    try:
        # Save the audio to a temporary WAV file
        temp_wav = "temp_audio.wav"
        wavfile.write(temp_wav, SAMPLE_RATE, audio_data.astype(np.int16))

        # Transcribe the audio file
        result = whisper_model.transcribe(temp_wav, fp16=False)

        # Clean up the temporary file
        try:
            os.remove(temp_wav)
        except:
            pass

        return result["text"].strip()

    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""


def text_to_speech(text, voice):
    """
    Synthesizes text to speech using Piper and plays the audio.
    Returns True if successful, False otherwise.
    """
    try:
        if not text.strip():
            print("No text to synthesize.")
            return False

        print(f"🔊 Synthesizing: '{text}'")

        # Synthesize speech - returns a generator of AudioChunk objects
        audio_generator = voice.synthesize(text)

        # Collect all audio chunks into a list
        audio_chunks = []
        sample_rate = None

        for chunk in audio_generator:
            # Extract sample rate from the first chunk
            if sample_rate is None:
                if hasattr(chunk, 'sample_rate'):
                    sample_rate = chunk.sample_rate
                else:
                    sample_rate = voice.config.sample_rate

            # Extract audio data from AudioChunk object
            if hasattr(chunk, 'audio_float_array'):
                audio_data_chunk = chunk.audio_float_array
            elif hasattr(chunk, 'audio_int16_array'):
                audio_data_chunk = chunk.audio_int16_array
            elif hasattr(chunk, 'audio'):
                audio_data_chunk = chunk.audio
            elif hasattr(chunk, 'data'):
                audio_data_chunk = chunk.data
            else:
                # If it's already a numpy array or similar
                audio_data_chunk = chunk

            # Convert to numpy array if needed
            if isinstance(audio_data_chunk, np.ndarray):
                audio_chunks.append(audio_data_chunk)
            else:
                # Try to convert to numpy array
                try:
                    if hasattr(audio_data_chunk, '__array__'):
                        converted = np.array(audio_data_chunk)
                    elif hasattr(audio_data_chunk, 'numpy'):
                        converted = audio_data_chunk.numpy()
                    else:
                        converted = np.array(audio_data_chunk, dtype=np.float32)

                    audio_chunks.append(converted)
                except Exception as e:
                    print(f"Debug: Skipping chunk due to conversion error: {e}")
                    continue

        if not audio_chunks:
            print("Error: No valid audio data generated.")
            return False

        # Combine all audio chunks
        try:
            audio_data = np.concatenate(audio_chunks)
        except Exception as e:
            # If that fails, try to convert each chunk individually
            try:
                converted_chunks = []
                for chunk in audio_chunks:
                    if isinstance(chunk, np.ndarray):
                        converted_chunks.append(chunk)
                    else:
                        converted_chunks.append(np.array(chunk, dtype=np.float32))
                audio_data = np.concatenate(converted_chunks)
            except Exception as e2:
                # Last resort: flatten everything
                audio_data = np.array(audio_chunks).flatten()

        if len(audio_data) == 0:
            print("Error: Empty audio data.")
            return False

        # Ensure we have a valid sample rate
        if sample_rate is None:
            sample_rate = voice.config.sample_rate
            print(f"⚠️  Using voice config sample rate: {sample_rate} Hz")

        # Ensure audio is in the correct format
        if not isinstance(audio_data, np.ndarray):
            if hasattr(audio_data, 'numpy'):  # Handle torch tensors
                audio_data = audio_data.numpy()
            else:
                audio_data = np.array(audio_data, dtype=np.float32)

        # Normalize audio if needed
        if audio_data.dtype != np.float32:
            if np.issubdtype(audio_data.dtype, np.integer):
                # Convert to float32 and normalize to [-1.0, 1.0]
                audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max

        # Play audio
        print(f"▶️  Playing audio (sample rate: {sample_rate} Hz, duration: {len(audio_data)/sample_rate:.2f}s)")
        sd.play(audio_data, sample_rate)
        sd.wait()
        return True

    except Exception as e:
        print(f"Error in speech synthesis: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("--- Local Voice Echo (Verbal Mirror) ---")

    try:
        # Load Whisper model
        whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
        print(f"✅ Whisper model ('{WHISPER_MODEL_NAME}') loaded for STT.")

        # Initialize Piper TTS
        print("Loading Piper TTS model...")
        model_path = "en_US-amy-medium.onnx"
        voice = PiperVoice.load(model_path)
        print(f"✅ Piper TTS model loaded: {model_path}")

        # Start the main loop with the loaded models
        while True:
            # Step 1: Record audio
            audio_data = record_audio(SAMPLE_RATE, CHANNELS, DTYPE)
            if audio_data is None:
                continue

            # Step 2: Transcribe audio
            transcribed_text = speech_to_text(audio_data, whisper_model)

            if not transcribed_text:
                print("🚫 No speech recognized. Listening again...")
                continue

            # Step 3: Check for stop word
            if transcribed_text.lower().strip().replace('.', '') == STOP_WORD:
                text_to_speech("Goodbye.", voice)
                break

    except KeyboardInterrupt:
        print("\nExiting program via Keyboard Interrupt.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        return


if __name__ == "__main__":
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        print("\nFATAL ERROR: FFmpeg is not installed or not in your system PATH.")
        print("Please install FFmpeg to allow Whisper to process audio files.")
        exit(1)
        
    main()
