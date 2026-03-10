from faster_whisper import WhisperModel
import sys

audio_file = sys.argv[1]

print("Loading Whisper model...")

model = WhisperModel("base", device="cpu")

print("Transcribing audio...")

segments, info = model.transcribe(audio_file)

result = ""

for segment in segments:
    result += segment.text + " "

print("\nTranscription Result:")
print(result)