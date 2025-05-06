# Video Translator

A Python tool that automatically translates videos from English to Russian, including speech-to-text transcription, translation, and text-to-speech dubbing.

## Features

- 🎥 Video processing with FFmpeg
- 🎤 Speech-to-text transcription using Whisper
- 🌐 Translation
- 🔊 Text-to-speech generation with gTTS
- ⏱️ Automatic audio timing synchronization
- 📁 Organized output structure
- 🔄 Progress saving and resuming

## Prerequisites

- Python 3.8 or higher
- FFmpeg installed and added to PATH
- Internet connection (for translation and TTS)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/video-translator.git
cd video-translator
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your video file in the project directory
2. Run the script with the video file path as an argument:
```bash
# Windows:
python main.py "path\to\your\video.mp4"

# Unix/MacOS:
python main.py "path/to/your/video.mp4"
```

For example:
```bash
# Windows:
python main.py "Vibe Coding is Getting Out of Hand.webm"

# Unix/MacOS:
python main.py "Vibe\ Coding\ is\ Getting\ Out\ of\ Hand.webm"
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

## How It Works

1. **Transcription**: Uses Whisper to convert speech to text
2. **Translation**: Translates the text
3. **TTS Generation**: Creates Russian audio using gTTS
4. **Audio Processing**: Adjusts audio timing to match video
5. **Video Creation**: Combines original video with new audio

## Progress Saving

The script saves progress at each step:
- If `transcript.txt` exists, skips transcription
- If `translated.txt` exists, skips translation
- If TTS chunks exist, skips TTS generation
- If final files exist, skips final processing

To force reprocessing, delete the corresponding files.

## Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Install FFmpeg and add it to your system PATH
   - Verify installation: `ffmpeg -version`

2. **TTS Connection Errors**
   - Check your internet connection
   - Ensure Google services are accessible
   - Try using a VPN if blocked

3. **Translation Quality**
   - The script uses same NLLB model for high-quality translations
   - Technical terms are preserved better than with basic translation models

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) for speech recognition
- [gTTS](https://github.com/pndurette/gTTS) for text-to-speech
- [FFmpeg](https://ffmpeg.org/) for video processing 