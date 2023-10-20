import socket
import speech_recognition as sr
import tempfile
import os

# Initialize a socket client to connect to the main program
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 12345))

def get_hearing_sender():
    recognizer = sr.Recognizer()

    while True:
        with sr.Microphone() as source:
            try:
                audio = recognizer.listen(source)  # Listen for audio

                # Save audio to a temporary WAV file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
                    temp_wav.write(audio.get_wav_data())

                # Send the path to the WAV file to the main program
                client_socket.send(temp_wav.name.encode())

            except sr.UnknownValueError:
                pass
            except sr.RequestError:
                pass
            except sr.WaitTimeoutError:
                pass

get_hearing_sender()
