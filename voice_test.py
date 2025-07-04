import pyttsx3

engine = pyttsx3.init()

# Test if the TTS engine works by saying a simple text
engine.say("Hello there")
engine.runAndWait()
