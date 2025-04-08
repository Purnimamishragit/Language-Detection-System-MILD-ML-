import assemblyai as aai
from constants import ASSEMBLY_AI_API_KEY

# Set the API key
aai.settings.api_key = ASSEMBLY_AI_API_KEY

# Use a raw string or escape the backslashes in the file path
audio_url = "audio1.mp3"

# Configure transcription with language detection enabled
config = aai.TranscriptionConfig(language_detection=True)

# Create a transcriber instance
transcriber = aai.Transcription(config=config)

# Transcribe the audio file
transcript = transcriber.transcribe(audio_url)

# Print the transcription text
print(transcript.text)
