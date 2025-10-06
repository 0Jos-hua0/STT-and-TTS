# 🗣️ Verbal Mirror - STT & TTS Application

A sophisticated, fully offline speech-to-text (STT) and text-to-speech (TTS) application with both web and desktop interfaces. Uses open-source AI models for completely local processing.

## ✨ Features

- **🎤 Real-time Speech Recognition** - Converts speech to text using Whisper AI
- **🔊 High-quality Text-to-Speech** - Uses Piper TTS with Amy's voice
- **🌐 Web Interface** - Modern web app with real-time status updates
- **🖥️ Desktop GUI** - PyQt5 interface for desktop use
- **⚡ Modular Architecture** - Separate modules for audio, STT, and TTS
- **🔄 Real-time Feedback** - Live status updates during recording and processing
- **💾 Persistent Transcriptions** - Previous transcriptions remain visible

## ⚠️ Important Warnings & Prerequisites

### 🚨 DLL Initialization Issues
**You may encounter DLL initialization failures** when running this application, especially on Windows. This is a common issue with the underlying audio libraries.

**Solutions:**
1. Install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. If issues persist, run the application multiple times - it often resolves after the first few attempts
3. Check that all dependencies are properly installed in your virtual environment

### 📋 Prerequisites

#### 1. External System Dependencies
- **FFmpeg** - Required by Whisper for audio processing
  - **Windows/macOS/Linux:** Install FFmpeg and ensure it's in your system PATH
- **Microsoft Visual C++ Redistributable** - For Windows audio support

#### 2. Python Environment Setup
```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

#### 3. AI Models (Auto-downloaded)
- **Whisper Model (STT)** - Uses 'small' model (downloads automatically ~40MB)
- **Piper Model (TTS)** - Uses 'en_US-amy-medium' voice (downloads automatically ~60MB)

## 🚀 How to Run

### Option 1: Web Interface (Recommended)
```bash
python web_app.py
```
- Open `http://localhost:5000` in your browser
- Click "Start Recording" to begin speech recognition
- View transcriptions in real-time
- Use "Speak" button for text-to-speech

### Option 2: Desktop GUI
```bash
python voice_gui.py
```
- PyQt5 interface for desktop use
- Same functionality as web app in a native window

## 📖 User Manual

### Web Interface Usage
1. **Start Recording:**
   - Click "🎤 Start Recording"
   - Speak clearly into your microphone
   - Status updates show recording progress
   - Click "⏹️ Stop Recording" when finished

2. **View Transcriptions:**
   - Transcriptions appear in the text area during/after recording
   - Previous transcriptions are preserved between recordings

3. **Text-to-Speech:**
   - Type text in the input field
   - Click "Speak" to hear it spoken
   - Uses Amy's natural voice

### Desktop GUI Usage
- Same functionality as web interface
- Native desktop window
- May have better audio device handling

### Status Messages
- **"Recording..."** - Currently recording audio
- **"Recording: X.Xs elapsed"** - Shows recording duration
- **"Recording: Speech detected"** - Audio input detected
- **"Transcribing audio..."** - Processing speech to text
- **"Ready"** - Waiting for input

## 🏗️ Architecture

### Modular Design
- **`audio.py`** - Audio recording and processing
- **`stt.py`** - Speech-to-text using Whisper
- **`tts.py`** - Text-to-speech using Piper
- **`web_app.py`** - Flask web application
- **`voice_gui.py`** - PyQt5 desktop interface

### Key Features
- **Real-time Status Updates** - Live feedback during all operations
- **Persistent Sessions** - Previous transcriptions remain visible
- **STT-Only Mode** - No automatic TTS after transcription
- **Error Handling** - Graceful handling of audio/processing errors

## 🔧 Recent Updates

### Version 2.0 - Major Improvements
- ✅ **Modular Architecture** - Split into separate audio, STT, and TTS modules
- ✅ **Real-time Status Updates** - Live feedback during recording and processing
- ✅ **Persistent Transcriptions** - Previous results don't disappear
- ✅ **STT-Only Mode** - Removed automatic TTS after transcription
- ✅ **Enhanced Web Interface** - Better real-time updates and status display
- ✅ **Improved Error Handling** - Better DLL initialization and audio error handling

### Previous Versions
- **Version 1.0** - Basic voice repeater functionality

## 🛠️ Troubleshooting

### Common Issues

#### DLL Initialization Failures
```
Error: DLL load failed
```
**Solutions:**
1. Install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Try running the application multiple times
3. Ensure you're using the virtual environment

#### Audio Device Issues
```
Error: Audio device not found
```
**Solutions:**
1. Check microphone permissions
2. Verify audio drivers are installed
3. Try different audio devices in system settings

#### Model Download Issues
```
Error: Model download failed
```
**Solutions:**
1. Check internet connection
2. Models download automatically on first run
3. Ensure sufficient disk space (~100MB for models)

#### Transcription Not Displaying
```
Backend shows transcription but web interface doesn't update
```
**Solutions:**
1. Check browser console for JavaScript errors
2. Ensure `/api/status` endpoint is accessible
3. Verify real-time polling is working

### Performance Tips
- Use the 'small' Whisper model for better performance
- Close other applications that use audio/microphone
- Ensure sufficient RAM (at least 4GB recommended)

## 📝 Requirements

See `requirements.txt` for complete dependency list:
- `openai-whisper` - Speech recognition
- `piper-tts` - Text-to-speech
- `sounddevice` - Audio input/output
- `numpy` - Audio processing
- `flask` - Web framework
- `pyqt5` - Desktop GUI

## 🤝 Contributing

This is an open-source project. Feel free to contribute improvements, bug fixes, or new features!

## 📄 License

This project is open-source and available for personal and educational use.
