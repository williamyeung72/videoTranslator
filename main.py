import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
from gtts import gTTS
from transformers import pipeline
import os
import tempfile
import shutil
import subprocess
from pathlib import Path
import re
import mutagen
import argparse

# === Clean filename from special characters ===
def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

# === 0. Check for ffmpeg ===
def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("❌ ffmpeg not found. Install FFmpeg and add it to PATH.")

# === Get audio duration ===
def get_audio_duration(path):
    from mutagen.mp3 import MP3
    return MP3(path).info.length

# === Adjust audio speed to target duration ===
def adjust_audio_speed(input_path, output_path, target_duration):
    actual_duration = get_audio_duration(input_path)
    if actual_duration > target_duration:
        speed = actual_duration / target_duration
        if speed > 2.0:
            print(f"⚠️ Audio too long: {actual_duration:.2f}s > {target_duration:.2f}s, limited to 2x")
            speed = 2.0
        command = [
            "ffmpeg", "-y", "-i", input_path,
            "-filter:a", f"atempo={speed:.3f}",
            output_path
        ]
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    else:
        shutil.copy(input_path, output_path)
        return False

# === 1. Video transcription with Whisper ===
def transcribe_video(video_path, transcript_path):
    print("🔍 Loading Whisper model...")
    model = whisper.load_model("base")
    print("📼 Starting transcription...")
    result = model.transcribe(video_path)
    with open(transcript_path, "w", encoding="utf-8") as f:
        for seg in result['segments']:
            f.write(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}\n")
    print(f"✅ Transcribed {len(result['segments'])} segments")
    return result['segments']

# === 2. Text translation ===
def translate_text(texts, output_path):
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        raise ImportError("❌ Required packages not found. Install them: pip install transformers sentencepiece")

    print("🌐 Loading translation model...")
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    print("🔁 Translating text...")
    translations = []
    for i, text in enumerate(texts):
        try:
            # Tokenize the text
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            # Translate from English (eng_Latn) to Russian (rus_Cyrl)
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id["rus_Cyrl"],
                max_length=512
            )
            
            # Decode the translation
            translated = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            
            print(f"Original: {text[:50]}...")
            print(f"Translated: {translated[:50]}...")
            
        except Exception as e:
            print(f"⚠️ Translation error for segment {i}: {e}")
            translated = text
        translations.append(translated)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in translations:
            f.write(line + "\n")

    print(f"✅ Translated {len(translations)} segments")
    return translations

# === 3. Generate TTS with timing ===
def generate_tts_with_timing(texts, output_dir, base_name, segments):
    audio_files = []

    for i, (translated_text, seg) in enumerate(zip(texts, segments)):
        idx = f"{i:04d}"
        temp_raw = output_dir / f"{base_name}_{idx}_raw.mp3"
        temp_path = output_dir / f"{base_name}_{idx}.mp3"

        # Generate TTS
        generate_tts_audio(translated_text, str(temp_raw))

        # Calculate duration of current video segment
        start, end = float(seg['start']), float(seg['end'])
        target_duration = end - start

        # Adjust audio speed to sync with video
        adjust_audio_speed(str(temp_raw), str(temp_path), target_duration)
        
        audio_files.append(str(temp_path))

    return audio_files

# === 4. TTS with gTTS ===
def generate_tts_audio(text, output_path, lang='ru'):
    try:
        tts = gTTS(text, lang=lang)
        tts.save(output_path)
        return True
    except Exception as e:
        print(f"⚠️ TTS error for \"{text[:30]}...\": {e}")
        return False

# === 5. Combine mp3 files ===
def combine_audio_segments(audio_dir, final_audio_dir, output_path):
    # Get list of mp3 files
    print("try search in audio_dir=", audio_dir)
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(".mp3") and not f.endswith("_raw.mp3")])

    # Create list file for concatenation
    list_path = os.path.join(audio_dir, "list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for file in audio_files:
            file_path = os.path.join(file)
            f.write(f"file '{file_path}'\n")

    # Convert final file path to absolute path
    print("Converting final file path to absolute path")
    print(final_audio_dir, "/", output_path)
    combined_path = os.path.join(final_audio_dir, output_path)
    os.makedirs(final_audio_dir, exist_ok=True)

    command = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy", combined_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ Combined audio file saved as {combined_path}")

# === 6. Replace audio track in video ===
def replace_audio(video_path, audio_path, output_path):
    print("🎬 Adding audio to video...")

    # ✅ Convert paths to POSIX (with forward slashes)
    video_path = Path(video_path).as_posix()
    audio_path = Path(audio_path).as_posix()
    output_path = Path(output_path).as_posix()

    command = f'ffmpeg -y -i "{video_path}" -i "{audio_path}" -c:v copy -map 0:v:0 -map 1:a:0 -shortest "{output_path}"'
    os.system(command)

# === Main function ===
def main(video_path):
    check_ffmpeg()

    input_path = Path(video_path)
    base_name = sanitize_filename(input_path.stem)
    
    # Create output directory structure
    output_dir = Path("output") / base_name
    tts_chunks_dir = output_dir / "tts-chunks"
    output_dir.mkdir(parents=True, exist_ok=True)
    tts_chunks_dir.mkdir(parents=True, exist_ok=True)

    # Define all output file paths
    transcript_file = output_dir / "transcript.txt"
    translated_file = output_dir / "translated.txt"
    dubbed_audio = output_dir / "audio_dubbed.mp3"
    final_output = output_dir / f"{base_name}_dubbed.mp4"

    # === Transcription ===
    if transcript_file.exists():
        print(f"⏩ Skipping transcription — found file {transcript_file}")
        segments = []
        with open(transcript_file, "r", encoding="utf-8") as f:
            for line in f:
                if "] " in line:
                    # Parse timing and text from line like "[0.00 - 2.50] Some text"
                    timing, text = line.split("] ", 1)
                    start, end = timing.strip("[]").split(" - ")
                    segments.append({
                        "text": text.strip(),
                        "start": float(start),
                        "end": float(end)
                    })
    else:
        print("🔍 Transcribing video...")
        segments = transcribe_video(video_path, transcript_file)

    texts = [seg['text'] for seg in segments]

    # === Translation ===
    if translated_file.exists():
        print(f"⏩ Skipping translation — found file {translated_file}")
        with open(translated_file, "r", encoding="utf-8") as f:
            translations = [line.strip() for line in f.readlines()]
    else:
        print("🌐 Translating text...")
        translations = translate_text(texts, translated_file)

    # === TTS with timing ===
    expected_audio_files = [str(tts_chunks_dir / f"{base_name}_{i:04d}.mp3") for i in range(len(translations))]
    if all(os.path.exists(f) and os.path.getsize(f) > 0 for f in expected_audio_files):
        print("⏩ Skipping TTS audio generation — all audio files found in folder.")
        temp_audio_files = expected_audio_files
    else:
        print("🔊 Generating TTS audio with timing...")
        temp_audio_files = generate_tts_with_timing(translations, tts_chunks_dir, base_name, segments)
        print(f"✅ Generated {len(temp_audio_files)} audio files")

    # === Combine audio ===
    if not dubbed_audio.exists():
        print("🎧 Combining audio segments...")
        combine_audio_segments(tts_chunks_dir, output_dir, "audio_dubbed.mp3")
    else:
        print(f"⏩ Skipping combination — found file {dubbed_audio}")

    # === Build final video ===
    if not final_output.exists():
        print("🎬 Creating final video...")
        replace_audio(video_path, dubbed_audio, final_output)
    else:
        print(f"⏩ Skipping build — found file {final_output}")

    print(f"🎉 Done! Result: {final_output}")

# === Entry point ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Translate video from English to Russian')
    parser.add_argument('video_path', help='Path to the video file to translate')
    args = parser.parse_args()
    
    main(args.video_path)
