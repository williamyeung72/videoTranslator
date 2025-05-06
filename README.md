# Video Translator

A tool for translating videos between different languages with automatic transcription, translation, and voice synthesis.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Language Options](#language-options)
  - [RVC Options](#rvc-options)
- [How It Works](#how-it-works)
- [Models and Voices](#models-and-voices)
  - [Whisper Models](#whisper-models)
  - [Voice Selection](#voice-selection)
- [Supported Languages](#supported-languages)
- [Output Structure](#output-structure)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- 🎤 Automatic video transcription using OpenAI's Whisper
- 🌐 Multi-language translation using M2M100
- 🔊 Text-to-speech with gTTS
- 🎵 Optional RVC voice conversion
- 🌍 Support for multiple languages
- 💾 Progress saving and resuming
- ⏱️ Automatic audio timing synchronization

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed and added to PATH
- Internet connection (for translation and TTS)
- CUDA-capable GPU (recommended for RVC (but u can use it project without RVC))

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-translator.git
cd video-translator
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg (required for audio/video processing):
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH
- Linux: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`

## Usage

Basic usage:
1. Place your video file in the project directory `path/to/video.mp4`
2. Run the script:
```bash
python main.py "path/to/video.mp4"
```

With language options:
```bash
python main.py "path/to/video.mp4" --source-lang en --target-lang ru
```

### RVC Options (u can eneble or disable it)
```bash
# Disable RVC
python main.py "path/to/video.mp4" --no-rvc

# Use specific RVC model
python main.py "path/to/video.mp4" --rvc-model "models/rvc/your_model"
```

## How It Works
Basic usage:

```bash
python main.py "path/to/video.mp4"
```

The script will create an organized output structure:
```
output/
└── video_name/
    ├── tts-chunks/           # Individual TTS audio chunks
    │   ├── video_name_0000.mp3
    │   ├── video_name_0001.mp3
    │   └── ...
    ├── transcript.txt        # Original transcription
    ├── translated.txt        # Translated text
    ├── audio_dubbed.mp3      # Combined dubbed audio
    └── video_name_dubbed.mp4 # Final video with dubs
```
Main process in project:
1. **Transcription**: Uses Whisper to convert speech to text
2. **Translation**: Translates the text using M2M100 model
3. **TTS Generation**: Creates audio using gTTS
4. **Audio Processing**: Adjusts audio timing to match video
5. **Video Creation**: Combines original video with new audio


The script saves progress at each step:
- If `transcript.txt` exists, skips transcription
- If `translated.txt` exists, skips translation
- If TTS chunks exist, skips TTS generation
- If final files exist, skips final processing

To force reprocessing, delete the corresponding files.

## Models and Voices

### Whisper Models

The default model is "base", but you can use different Whisper models for better accuracy:

| Model | Size | RAM | Speed | Quality |
|-------|------|-----|--------|---------|
| tiny | 1GB | ~1GB | Fastest | Basic |
| base | 1GB | ~1GB | Fast | Good |
| small | 2GB | ~2GB | Medium | Better |
| medium | 5GB | ~5GB | Slow | Great |
| large | 10GB | ~10GB | Slowest | Best |

To change the Whisper model:
```python
def transcribe_video(video_path, transcript_path, source_lang='en'):
    print("🔍 Loading Whisper model...")
    # Change "base" to any of: "tiny", "base", "small", "medium", "large"
    model = whisper.load_model("base")
```

### Voice Selection

#### gTTS Voices
- Automatic voice selection based on target language
- Natural-sounding voices for each supported language
- No additional configuration needed

#### RVC Voice Conversion
1. Create a `models/rvc/` directory
2. Add your RVC model files (`.pth` and `.index`)
3. Update the model path:
```python
rvc = RVCConverter("models/rvc/your_model_name")
```

Available RVC models:
- Male voices: Add your male voice model files
- Female voices: Add your female voice model files

## Supported Languages

- English (en)
- Russian (ru)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Japanese (ja)
- Korean (ko)
- Chinese (zh)


## Requirements

- Python 3.8+
- FFmpeg
- CUDA-capable GPU (recommended for RVC)
- See `requirements.txt` for Python dependencies

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Install FFmpeg and add it to your system PATH
   - Verify installation: `ffmpeg -version`

2. **Translation Quality**
   - Try different Whisper models for better transcription
   - Check if the source language is correctly set

3. **Voice Quality**
   - Use RVC for better voice quality
   - Try different RVC models for different voices

4. **GPU Issues**
   - Ensure CUDA is properly installed
   - Check GPU memory usage
   - Try smaller models if out of memory

## License

MIT License

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [gTTS](https://github.com/pndurette/gTTS) for text-to-speech
- [FFmpeg](https://ffmpeg.org/) for video processing
- [M2M100](https://huggingface.co/facebook/m2m100_418M) for translation 