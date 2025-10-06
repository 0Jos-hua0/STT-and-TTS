import sys
import os

# --- Import the new modular structure ---
from audio import record_audio_stream, SAMPLE_RATE
from stt import SpeechToText
from tts import TextToSpeech

print("Core libraries imported successfully.")


# --- Now, import PyQt5 ---
print("Importing PyQt5...")
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTextEdit, QLabel, QFrame, QProgressBar)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QFont, QIcon
print("PyQt5 imported successfully.")


class ModelLoaderWorker(QThread):
    """
    Worker thread to load AI models in the background to prevent GUI freezing.
    """
    models_loaded = pyqtSignal(object, object)  # stt_model, tts_model
    error_occurred = pyqtSignal(str)

    def run(self):
        try:
            print("Loading STT model...")
            stt_model = SpeechToText()
            stt_model.load_model()
            print("STT model loaded.")

            print("Loading TTS model...")
            tts_model = TextToSpeech()
            if tts_model.load_model():
                print("TTS model loaded.")
                self.models_loaded.emit(stt_model, tts_model)
            else:
                self.error_occurred.emit("Failed to load TTS model")
        except Exception as e:
            self.error_occurred.emit(f"Failed to load models: {e}")


class VoiceWorker(QThread):
    """
    A single, persistent worker for all voice processing tasks.
    """
    transcription_update = pyqtSignal(str)
    tts_complete = pyqtSignal(bool)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, stt_model, tts_model):
        super().__init__()
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.is_recording = False
        self.text_to_speak = None

    def start_recording(self):
        self.is_recording = True
        self.text_to_speak = None
        self.start()

    def stop_recording(self):
        self.is_recording = False

    def start_tts(self, text):
        self.text_to_speak = text
        self.is_recording = False
        self.start()

    def run(self):
        if self.is_recording:
            self.run_recording_and_transcription()
        elif self.text_to_speak:
            self.run_tts()

    def run_recording_and_transcription(self):
        try:
            self.status_update.emit("🎤 Recording... Speak now!")
            # Use the streaming recorder from audio module
            audio_data = record_audio_stream(stop_event=lambda: not self.is_recording)

            if audio_data is None or len(audio_data) < SAMPLE_RATE:
                self.status_update.emit("Recording stopped. No speech detected.")
                # We still emit an empty string to reset the GUI state
                self.transcription_update.emit("")
                return

            self.status_update.emit("🧠 Transcribing audio...")
            transcribed_text = self.stt_model.transcribe_audio(audio_data, SAMPLE_RATE)
            self.transcription_update.emit(transcribed_text)
        except Exception as e:
            self.error_occurred.emit(f"Recording/Transcription error: {e}")

    def run_tts(self):
        try:
            self.status_update.emit("🔊 Synthesizing speech...")
            success = self.tts_model.synthesize_speech(self.text_to_speak)
            self.tts_complete.emit(success)
        except Exception as e:
            self.error_occurred.emit(f"TTS error: {e}")


class VoiceRepeaterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.stt_model = None
        self.tts_model = None
        self.worker = None

        self.init_ui()
        self.load_styles()
        self.start_model_loading()

    def start_model_loading(self):
        self.status_label.setText("⏳ Loading AI models... Please wait.")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate progress

        self.model_loader = ModelLoaderWorker()
        self.model_loader.models_loaded.connect(self.on_models_loaded)
        self.model_loader.error_occurred.connect(self.on_model_load_error)
        self.model_loader.start()

    def on_models_loaded(self, stt_model, tts_model):
        self.stt_model = stt_model
        self.tts_model = tts_model
        self.status_label.setText("✅ Models loaded successfully. Ready.")
        self.progress_bar.setVisible(False)
        self.record_button.setEnabled(True)
        self.speak_button.setEnabled(True)

        # Initialize the persistent worker thread
        self.worker = VoiceWorker(self.stt_model, self.tts_model)
        self.worker.status_update.connect(self.status_label.setText)
        self.worker.error_occurred.connect(self.on_worker_error)
        self.worker.transcription_update.connect(self.on_transcription_complete)
        self.worker.tts_complete.connect(self.on_tts_complete)

    def on_model_load_error(self, error_message):
        self.status_label.setText(f"❌ {error_message}")
        self.progress_bar.setVisible(False)
        self.record_button.setEnabled(False)
        self.speak_button.setEnabled(False)

    def init_ui(self):
        self.setWindowTitle("Verbal Mirror - Voice Repeater")
        self.setGeometry(100, 100, 800, 600)
        self.setWindowIcon(QIcon()) 

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        title_label = QLabel("🎤 Verbal Mirror 🎤")
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        self.status_label = QLabel("Initializing...")
        self.status_label.setFont(QFont("Arial", 12))
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        text_group = QFrame()
        text_group.setFrameStyle(QFrame.StyledPanel)
        text_layout = QVBoxLayout(text_group)

        input_label = QLabel("Text to Speak:")
        text_layout.addWidget(input_label)
        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Enter text to synthesize...")
        self.input_text.setMaximumHeight(100)
        text_layout.addWidget(self.input_text)

        transcription_label = QLabel("Live Transcription:")
        text_layout.addWidget(transcription_label)
        self.transcription_display = QTextEdit()
        self.transcription_display.setReadOnly(True)
        self.transcription_display.setPlaceholderText("Transcribed text will appear here...")
        text_layout.addWidget(self.transcription_display)
        layout.addWidget(text_group)

        button_layout = QHBoxLayout()
        self.record_button = QPushButton("🎙️ Start Recording")
        self.record_button.clicked.connect(self.start_recording)
        self.record_button.setEnabled(False)
        button_layout.addWidget(self.record_button)

        self.stop_button = QPushButton("⏹️ Stop Recording")
        self.stop_button.clicked.connect(self.stop_recording)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        self.speak_button = QPushButton("🔊 Speak Text")
        self.speak_button.clicked.connect(self.speak_text)
        self.speak_button.setEnabled(False)
        button_layout.addWidget(self.speak_button)
        
        self.clear_button = QPushButton("🗑️ Clear")
        self.clear_button.clicked.connect(self.clear_text)
        button_layout.addWidget(self.clear_button)
        
        layout.addLayout(button_layout)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

    def load_styles(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #2b2b2b; color: #f0f0f0; }
            QLabel { color: #f0f0f0; }
            QTextEdit {
                background-color: #3c3f41; color: #f0f0f0; border: 1px solid #555;
                border-radius: 5px; padding: 8px; font-size: 14px;
            }
            QPushButton {
                color: white; border: none; padding: 12px 24px;
                border-radius: 8px; font-size: 14px; font-weight: bold;
            }
            QPushButton:disabled { background-color: #555; color: #999; }
            #record_button { background-color: #28a745; }
            #record_button:hover { background-color: #218838; }
            #stop_button { background-color: #dc3545; }
            #stop_button:hover { background-color: #c82333; }
            #speak_button { background-color: #007bff; }
            #speak_button:hover { background-color: #0069d9; }
            #clear_button { background-color: #ffc107; color: #111; }
            #clear_button:hover { background-color: #e0a800; }
            QProgressBar {
                border: 2px solid grey; border-radius: 5px; text-align: center;
            }
            QProgressBar::chunk { background-color: #007bff; width: 20px; }
        """)
        # Set object names for specific styling
        self.record_button.setObjectName("record_button")
        self.stop_button.setObjectName("stop_button")
        self.speak_button.setObjectName("speak_button")
        self.clear_button.setObjectName("clear_button")

    def start_recording(self):
        if self.worker.isRunning(): return
        self.record_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.speak_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.worker.start_recording()

    def stop_recording(self):
        # This will signal the worker to stop, but the worker will continue
        # until the transcription and TTS are complete.
        self.worker.stop_recording()
        self.stop_button.setEnabled(False) # Prevent multiple clicks
        self.status_label.setText("⏹️ Stopping recording...")

    def on_transcription_complete(self, text):
        self.transcription_display.setPlainText(text)
        # Automatically speak the result
        if text.strip():
            self.status_label.setText("✅ Transcription complete. Synthesizing response...")
            self.speak_text(text_to_speak=text)
        else:
            self.status_label.setText("✅ No speech detected in recording.")
            # If no speech, just reset the UI state
            self.on_tts_complete(True)

    def speak_text(self, text_to_speak=None):
        if self.worker.isRunning(): return
        
        # Use provided text (from transcription) or text from the input box
        text = text_to_speak if isinstance(text_to_speak, str) else self.input_text.toPlainText().strip()
        if not text:
            self.status_label.setText("⚠️ No text to speak.")
            return

        self.record_button.setEnabled(False)
        self.speak_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.worker.start_tts(text)

    def on_tts_complete(self, success):
        self.status_label.setText("✅ Speech synthesis complete." if success else "❌ Speech synthesis failed.")
        # Reset button states to be ready for the next action
        self.record_button.setEnabled(True)
        self.speak_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.clear_button.setEnabled(True)
    
    def on_worker_error(self, error_message):
        self.status_label.setText(f"❌ Error: {error_message}")
        self.record_button.setEnabled(True)
        self.speak_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.clear_button.setEnabled(True)
        if self.worker.isRunning():
            self.worker.quit()

    def clear_text(self):
        self.input_text.clear()
        self.transcription_display.clear()
        self.status_label.setText("Ready.")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop_recording()
            self.worker.quit()
            self.worker.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceRepeaterGUI()
    window.show()
    sys.exit(app.exec_())

