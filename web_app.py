import os
import threading
import time
import json
from flask import Flask, render_template, request, jsonify
from audio import record_audio_stream, SAMPLE_RATE, CHANNELS, ACTIVATION_THRESHOLD, SILENCE_TIMEOUT
from stt import SpeechToText
from tts import TextToSpeech

# Initialize Flask app
app = Flask(__name__)

# Global variables
whisper_model = None
voice_model = None
recording_thread = None
is_recording = False
transcription_text = ""
current_status = "Ready"

# Initialize models
stt_model = SpeechToText()
tts_model = TextToSpeech()

def initialize_models():
    """Initialize models on startup"""
    global whisper_model, voice_model

    try:
        print("Loading Whisper model...")
        stt_model.load_model()
        whisper_model = stt_model.model
        print("✅ Whisper model loaded")

        print("Loading Piper TTS model...")
        if tts_model.load_model():
            voice_model = tts_model.voice_model
            print("✅ Piper TTS model loaded")
        else:
            print("❌ Failed to load Piper TTS model")

    except Exception as e:
        print(f"❌ Error loading models: {e}")

def record_audio():
    """Record audio from microphone"""
    global is_recording, transcription_text, current_status
    print("🎤 Recording started...")
    recording = []
    silence_start_time = None
    start_time = time.time()
    chunk_duration = 0.1
    last_update_time = 0
    update_interval = 0.5  # Update every 0.5 seconds

    import sounddevice as sd
    import numpy as np

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16') as stream:
        while is_recording:
            if time.time() - start_time > 10.0:  # Max 10 seconds
                current_status = "Recording: Maximum duration reached"
                break

            audio_chunk, overflowed = stream.read(int(SAMPLE_RATE * chunk_duration))
            if overflowed:
                print("⚠️ Audio buffer overflowed")

            volume = np.linalg.norm(audio_chunk) * 10 / (2**15)

            if volume > ACTIVATION_THRESHOLD:
                if silence_start_time is None:
                    print("🗣️ Speech detected")
                    current_status = "Recording: Speech detected"
                silence_start_time = None

                # Provide real-time feedback during recording
                current_time = time.time()
                if current_time - last_update_time > update_interval:
                    elapsed = current_time - start_time
                    current_status = f"Recording: {elapsed:.1f}s elapsed"
                    last_update_time = current_time
            else:
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > SILENCE_TIMEOUT:
                    print("👂 Silence detected, stopping...")
                    current_status = "Processing audio..."
                    break

            recording.append(audio_chunk)

    if recording:
        print(f"✅ Recorded {len(recording) * chunk_duration:.1f} seconds of audio")
        return np.concatenate(recording, axis=0)
    return None

def transcribe_audio(audio_data):
    """Transcribe audio using Whisper"""
    print("🧠 Transcribing audio...")

    try:
        return stt_model.transcribe_audio(audio_data, SAMPLE_RATE)
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

def synthesize_speech(text):
    """Synthesize speech using Piper"""
    try:
        if not text.strip():
            return False

        print(f"🔊 Synthesizing: '{text}'")

        return tts_model.synthesize_speech(text)

    except Exception as e:
        print(f"Error in speech synthesis: {e}")
        return False

def initialize_models():
    """Initialize models on startup"""
    global whisper_model, voice_model

    try:
        print("Loading Whisper model...")
        stt_model.load_model()
        whisper_model = stt_model.model
        print("✅ Whisper model loaded")

        print("Loading Piper TTS model...")
        if tts_model.load_model():
            voice_model = tts_model.voice_model
            print("✅ Piper TTS model loaded")
        else:
            print("❌ Failed to load Piper TTS model")

    except Exception as e:
        print(f"❌ Error loading models: {e}")

def record_audio():
    """Record audio from microphone"""
    global is_recording, transcription_text
    print("🎤 Recording started...")
    recording = []
    silence_start_time = None
    start_time = time.time()
    chunk_duration = 0.1

    import sounddevice as sd
    import numpy as np

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='int16') as stream:
        while is_recording:
            if time.time() - start_time > 10.0:  # Max 10 seconds
                break

            audio_chunk, overflowed = stream.read(int(SAMPLE_RATE * chunk_duration))
            if overflowed:
                print("⚠️ Audio buffer overflowed")

            volume = np.linalg.norm(audio_chunk) * 10 / (2**15)

            if volume > ACTIVATION_THRESHOLD:
                if silence_start_time is None:
                    print("🗣️ Speech detected")
                silence_start_time = None
            else:
                if silence_start_time is None:
                    silence_start_time = time.time()
                elif time.time() - silence_start_time > SILENCE_TIMEOUT:
                    print("👂 Silence detected, stopping...")
                    break

            recording.append(audio_chunk)

    if recording:
        print(f"✅ Recorded {len(recording) * chunk_duration:.1f} seconds of audio")
        return np.concatenate(recording, axis=0)
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/start_recording', methods=['POST'])
def start_recording():
    global is_recording, recording_thread

    if is_recording:
        return jsonify({"status": "error", "message": "Already recording"})

    is_recording = True
    recording_thread = threading.Thread(target=record_and_process)
    recording_thread.start()

    return jsonify({"status": "recording"})

@app.route('/api/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording
    is_recording = False
    return jsonify({"status": "stopped"})

def transcribe_audio(audio_data):
    """Transcribe audio using Whisper"""
    print("🧠 Transcribing audio...")

    try:
        return stt_model.transcribe_audio(audio_data, SAMPLE_RATE)
    except Exception as e:
        print(f"Error during transcription: {e}")
        return ""

def synthesize_speech(text):
    """Synthesize speech using Piper"""
    try:
        if not text.strip():
            return False

        print(f"🔊 Synthesizing: '{text}'")

        return tts_model.synthesize_speech(text)

    except Exception as e:
        print(f"Error in speech synthesis: {e}")
        return False

@app.route('/api/speak', methods=['POST'])
def speak_text():
    data = request.get_json()
    text = data.get('text', '')

    if not text.strip():
        return jsonify({"status": "error", "message": "No text provided"})

    success = synthesize_speech(text)
    return jsonify({"status": "success" if success else "error"})

@app.route('/api/status')
def get_status():
    return jsonify({
        "recording": is_recording,
        "transcription": transcription_text,
        "status": current_status
    })

def record_and_process():
    """Record audio and process it"""
    global is_recording, transcription_text, current_status

    try:
        current_status = "Recording..."
        audio_data = record_audio()
        is_recording = False

        if audio_data is not None:
            current_status = "Transcribing audio..."
            transcribed_text = transcribe_audio(audio_data)
            transcription_text = transcribed_text

            if transcribed_text:
                print(f"Transcription: {transcribed_text}")
                # Don't auto-speak - user doesn't want TTS after STT
                current_status = "Ready"
            else:
                current_status = "Ready"
        else:
            current_status = "Ready"

    except Exception as e:
        print(f"Error in recording thread: {e}")
        is_recording = False
        current_status = "Ready"

if __name__ == '__main__':
    initialize_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
