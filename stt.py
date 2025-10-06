import os
import tempfile
import scipy.io.wavfile as wavfile
import whisper
from typing import Optional

# STT Configuration
WHISPER_MODEL_NAME = "small"


class SpeechToText:
    """Speech-to-Text functionality using Whisper"""

    def __init__(self, model_name: str = WHISPER_MODEL_NAME):
        """
        Initialize STT with Whisper model

        Args:
            model_name: Name of the Whisper model to use
        """
        self.model_name = model_name
        self.model = None

    def load_model(self) -> None:
        """Load the Whisper model"""
        if self.model is None:
            print(f"Loading Whisper model '{self.model_name}'...")
            self.model = whisper.load_model(self.model_name)
            print("✅ Whisper model loaded")

    def transcribe_audio(self, audio_data: bytes, sample_rate: int = 16000) -> str:
        """
        Transcribes audio data using Whisper

        Args:
            audio_data: Raw audio data as numpy array
            sample_rate: Sample rate of the audio data

        Returns:
            Transcribed text or empty string if no speech detected
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded. Call load_model() first.")

        print("🧠 Transcribing audio with Whisper...")

        try:
            # Save the audio to a temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_wav = temp_file.name

            # Ensure audio data is in the correct format for WAV file
            if hasattr(audio_data, 'astype'):
                audio_data = audio_data.astype('int16')

            wavfile.write(temp_wav, sample_rate, audio_data)

            # Transcribe the audio file
            result = self.model.transcribe(temp_wav, fp16=False)

            # Clean up the temporary file
            try:
                os.unlink(temp_wav)
            except:
                pass

            transcribed_text = result["text"].strip()
            print(f"📝 Transcription: '{transcribed_text}'")
            return transcribed_text

        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""

    def is_model_loaded(self) -> bool:
        """Check if the model is loaded"""
        return self.model is not None


# Convenience function for quick transcription
def transcribe_audio(audio_data: bytes, model_name: str = WHISPER_MODEL_NAME) -> str:
    """
    Convenience function to transcribe audio data

    Args:
        audio_data: Raw audio data as numpy array
        model_name: Name of the Whisper model to use

    Returns:
        Transcribed text
    """
    stt = SpeechToText(model_name)
    stt.load_model()
    return stt.transcribe_audio(audio_data)
