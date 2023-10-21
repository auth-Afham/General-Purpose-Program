import socket
import speech_recognition as sr
import audioop
from textblob import TextBlob

# Initialize a socket client to connect to the main program
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('localhost', 12345))

def check_amplitude(audio_data):
    # Calculate the amplitude of the audio data
    rms = audioop.rms(audio_data, 2)  # 2 for format=PCM
    return rms

def get_hearing_sender():
    hearing_amplitude = 0
    feedback = 0
    WORD_COUNT = 9
    hearing_words = [0] * WORD_COUNT

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        try:
            audio = recognizer.listen(source)  # Listen for up to 5 seconds

            # Calculate the amplitude of the audio
            hearing_amplitude = check_amplitude(audio.frame_data)

            text = recognizer.recognize_google(audio)
            print("You said: " + text)

            blob = TextBlob(text)
            feedback = blob.sentiment.polarity * 100

            # Tokenize the text into words
            words = text.lower().split()  # Convert to lowercase

            heard_words = words[-WORD_COUNT:]

            # Send the data to the main program
            data_to_send = f"{hearing_amplitude},{feedback},{','.join(map(str, heard_words))}"
            client_socket.send(data_to_send.encode())

        except sr.UnknownValueError:
            pass
        except sr.RequestError:
            pass
        except sr.WaitTimeoutError:
            pass

if __name__ == '__main__':
    try:
        while True:
            # Send the data to the main program
            get_hearing_sender()
    except KeyboardInterrupt:
        print("Program ended due to Ctrl+C in the terminal.")
        client_socket.close()
        exit(0)