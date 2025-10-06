# 🗣️ Local Voice Echo (Verbal Mirror)

This is a simple, fully offline project that converts speech-to-text (STT) and then immediately echoes the transcribed text using text-to-speech (TTS). It uses only open-source, locally installed models.

## ⚠️ Prerequisites

You MUST install the following external system tool and download the AI models before running the Python script.

### 1. External System Dependency (FFmpeg)

FFmpeg is required by the Whisper library to handle audio input correctly.

* **Windows/macOS/Linux:** Install FFmpeg and ensure it is available in your system's PATH.

### 2. Python Environment Setup

1.  Create and activate a virtual environment:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    ```
2.  Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

### 3. Download AI Models (STT and TTS)

The project requires a Whisper model and a Piper voice model to run locally.

1.  **Whisper Model (STT):**
    * The script uses the **'tiny'** model by default for fast performance on CPU. Whisper will download this model automatically the first time you run the script.

2.  **Piper Model (TTS):**
    * **Download the Piper Voice:** Go to the [Piper TTS GitHub repository](https://github.com/rhasspy/piper) to find voice models.
    * Download both the **ONNX file** (e.g., `en_US-lessac-medium.onnx`) and its corresponding **JSON configuration file** (e.g., `en_US-lessac-medium.json`).
    * Place **BOTH** of these files directly into your project's main directory.
    * **Important:** If you choose a different voice, update the file paths in `voice_repeater.py`.

## 🚀 How to Run the Project

1.  Ensure all prerequisites and models are set up and in the project folder.
2.  Run the main script:
    ```bash
    python voice_repeater.py
    ```

The application will start listening. Say something, and after a moment of silence, the program will transcribe your speech and immediately speak the text back to you.

To stop the program, simply say the word **"quit"** or press **Ctrl+C**.

## Troubleshooting

- If you get an error about missing DLLs, you may need to install the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).
- If you encounter issues with audio devices, check your system's audio settings and ensure the correct input/output devices are selected.
- For better performance, consider using a more powerful Whisper model (e.g., 'base' or 'small') if your system can handle it.
