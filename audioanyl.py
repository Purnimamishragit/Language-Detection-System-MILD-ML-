import speech_recognition as sr
from enum import Enum

class Language(Enum):
    ENGLISH = 'en-US'
    FRENCH = 'fr-FR'
    GERMAN = 'de-DE'
    ITALIAN = 'it-IT'
    SPANISH = 'es-ES'
    PORTUGUESE = 'pt-BR'
    KOREAN = "ko-KR"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    JAPANESE = "ja-JP"
    RUSSIAN = "ru-RU"
    POLISH = "pl-PL"
    UKRAINIAN = "uk-UA"
    BULGARIAN = "bg-BG"
    BENGALI = "bn-BD"
    TURKISH = "tr-TR"
    ARABIC = "ar-SA"
    INDONESIAN = "id-ID"
    THAI = "th-TH"
    VIETNAMESE = "vi-VN"
    MALAY = "ms-MY"
    HINDI = "hi-IN"
    PUNJABI = "pa-IN"
    TELUGU = "te-IN"
    GUJARATI = "gu-IN"
    ORIYA = "or-IN"
    MARATHI = "mr-IN"
    SINDHI = "sd-IN"
    TAMIL = "ta-IN"
    KANNADA = "kn-IN"
    MALAYALAM = "ml-IN"
    ASSAMESE = "as-IN"
    ODIA = "or-IN"
    SANSKRIT = "sa-IN"

class SpeechToText:
    @staticmethod
    def print_mic_device_index():
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print("{1}, device_index={0}".format(index, name))

    @staticmethod
    def speech_to_text(device_index, language=Language.ENGLISH):
        r = sr.Recognizer()
        with sr.Microphone(device_index=device_index) as source:
            print("Recording...")
            audio = r.listen(source)
            print("Recording Complete...")

            try:
                # Direct recognition in specified language
                text = r.recognize_google(audio, language=language.value)
                print(f"Transcribed Text ({language.name}):", text)
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))

def check_mic_device_index():
    SpeechToText.print_mic_device_index()

def run_speech_to_text_in_specified_language(device_index=1, language=Language.ENGLISH):
    SpeechToText.speech_to_text(device_index, language)

if __name__ == "__main__":
    check_mic_device_index()
    # Specify a device index and language here
    device_index = 1  # For example, set device index to 1
    language = Language.JAPANESE# Set to desired language, e.g., Spanish
    run_speech_to_text_in_specified_language(device_index=device_index, language=language)

