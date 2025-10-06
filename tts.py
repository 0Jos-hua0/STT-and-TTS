import os
import numpy as np
import sounddevice as sd
from typing import Optional, Union

# Lazy import for piper to avoid DLL issues at startup
def _get_piper_voice():
    """Lazy import of piper modules to avoid DLL issues at startup"""
    try:
        from piper.voice import PiperVoice
        return PiperVoice
    except ImportError as e:
        print(f"Error importing Piper: {e}")
        return None


class TextToSpeech:
    """Text-to-Speech functionality using Piper"""

    def __init__(self, model_path: str = "en_US-amy-medium.onnx"):
        """
        Initialize TTS with Piper voice model

        Args:
            model_path: Path to the Piper ONNX model file
        """
        self.model_path = model_path
        self.voice_model = None
        self._piper_voice_class = None

    def load_model(self) -> bool:
        """
        Load the Piper TTS model

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            print("Loading Piper TTS model...")
            PiperVoice = _get_piper_voice()

            if PiperVoice is None:
                print("❌ Piper TTS not available")
                return False

            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found: {self.model_path}")
                return False

            self._piper_voice_class = PiperVoice
            self.voice_model = PiperVoice.load(self.model_path)
            print("✅ Piper TTS model loaded")
            return True

        except Exception as e:
            print(f"❌ Error loading Piper model: {e}")
            return False

    def synthesize_speech(self, text: str) -> bool:
        """
        Synthesizes text to speech using Piper

        Args:
            text: Text to synthesize

        Returns:
            True if successful, False otherwise
        """
        if not text.strip():
            print("No text to synthesize.")
            return False

        if self.voice_model is None:
            if not self.load_model():
                return False

        try:
            print(f"🔊 Synthesizing: '{text}'")

            # Synthesize speech - returns a generator of AudioChunk objects
            audio_generator = self.voice_model.synthesize(text)

            # Collect all audio chunks into a list
            audio_chunks = []
            sample_rate = None

            for chunk in audio_generator:
                # Extract sample rate from the first chunk
                if sample_rate is None:
                    if hasattr(chunk, 'sample_rate'):
                        sample_rate = chunk.sample_rate
                    else:
                        sample_rate = self.voice_model.config.sample_rate

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
                sample_rate = self.voice_model.config.sample_rate
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

    def is_model_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self.voice_model is not None


# Convenience function for quick synthesis
def synthesize_speech(text: str, model_path: str = "en_US-amy-medium.onnx") -> bool:
    """
    Convenience function to synthesize speech

    Args:
        text: Text to synthesize
        model_path: Path to the Piper model file

    Returns:
        True if successful, False otherwise
    """
    tts = TextToSpeech(model_path)
    return tts.synthesize_speech(text)
