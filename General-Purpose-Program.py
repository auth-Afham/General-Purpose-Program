import os
import pyautogui
import time
import math
from PIL import ImageGrab
import joblib
from sklearn.linear_model import SGDRegressor
import numpy as np
from pynput.keyboard import Key, Listener as KeyboardListener
from pynput.mouse import Listener as MouseListener, Button, Controller
import csv
import speech_recognition as sr
import audioop
from textblob import TextBlob
from collections import deque

hearing_filename = "hearing.csv"
dataset_filename = "dataset.csv"
model_filename = "model.pkl"

touch_keyboard = 0
touch_mouse = 0
touch_cursor_x = 0
touch_cursor_y = 0

sight_average_red = 0
sight_average_green = 0
sight_average_blue = 0

hearing_amplitude = 0

feedback = 0

command = 0
next_command = 0

argument_1st = 0
argument_2nd = 0

counter = 0
action = 0

word_indices = [0] * 10
with_action_data = [touch_keyboard, touch_mouse, touch_cursor_x, touch_cursor_y, sight_average_red, sight_average_green, sight_average_blue, hearing_amplitude, feedback, command, counter, argument_1st, argument_2nd, action]

# Insert the additional_list between elements 3 and 4 of the original_list
index_to_insert = 8
combined_with_action_data = with_action_data[:index_to_insert] + word_indices + with_action_data[index_to_insert:]

previous_controlled_action_flag = False
controlled_action_flag = False

# Initialize a list to store the last 10 heard words
heard_words = deque(maxlen=10)

# List of variable names
variable_names = [
    "touch_keyboard", "touch_mouse", "touch_cursor_x", "touch_cursor_y",
    "sight_average_red", "sight_average_green", "sight_average_blue",
    "hearing_amplitude",
    "1st_recent_word", "2nd_recent_word", "3rd_recent_word", "4th_recent_word", "5th_recent_word", "6th_recent_word", "7th_recent_word", "8th_recent_word", "9th_recent_word", "10th_recent_word",
    "feedback", "command", "argument_1st", "argument_2nd", "counter", "action"
]

negative_keys_to_check = ['backspace', 'del', 'esc', 'ctrl+c', 'ctrl+z', 'f1', 'f4', 'f7', '-', '/', '!', 'capslock']

# Define a list of keys you want to check
positive_keys_to_check = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'up', 'down', 'left', 'right',  # Arrow keys
    'space', 'enter', 'shift', 'ctrl', 'alt',  # Other keys
]

def floor_positive_absolute(x):
    # Calculate the floor value of the absolute (positive) value of x
    return math.floor(abs(x))

def floor_positive_ceil_negative(x):
    if x >= 0:
        return math.floor(x)
    else:
        return math.ceil(x)

# Function to format the key for display
def format_key(key):
    if hasattr(key, 'name'):
        return key.name
    else:
        return str(key).strip("'")

# Callback function for key presses
def on_key_press(key):
    global touch_keyboard, feedback, next_command, action, negative_keys_to_check, positive_keys_to_check
    
    # Format the key for display
    formatted_key = format_key(str(key))

    if formatted_key in negative_keys_to_check:
        touch_keyboard = negative_keys_to_check.index(formatted_key) + 1
        feedback -= 1
        print(f"'{formatted_key}' is pressed at index {touch_keyboard}!")

    if formatted_key in positive_keys_to_check:
        touch_keyboard = len(negative_keys_to_check) + positive_keys_to_check.index(formatted_key) + 1
        feedback += 1
        print(f"'{formatted_key}' is pressed at index {touch_keyboard}!")

# Callback function for key releases
def on_key_release(key):
    global touch_keyboard

    # Do something when a key is released
    touch_keyboard = 0

# Callback function for mouse clicks
def on_click(x, y, button, pressed):
    global touch_mouse, feedback, negative_keys_to_check, positive_keys_to_check

    if pressed:
        if button == Button.left:
            touch_mouse = 2 # You can assign any unique value you like
            feedback += 1  # You can assign the appropriate feedback value
            print(f"Left mouse button clicked at index {touch_mouse}!")
        elif button == Button.right:
            touch_mouse = 3 # You can assign any unique value you like
            feedback += 1  # You can assign the appropriate feedback value
            print(f"Right mouse button clicked at index {touch_mouse}!")

def get_touch_values():
    global touch_cursor_x, touch_cursor_y
    
    # Initialize the mouse controller
    mouse = Controller()

    # Get the current cursor position
    touch_cursor_x, touch_cursor_y = mouse.position

def check_uniform_color(pixel_data):
    global feedback, next_command, action

    # Get the RGB value of the first pixel
    first_pixel_color = pixel_data[0]

    # Check if all pixels have the same color as the first pixel
    if np.all(pixel_data == first_pixel_color):
        feedback -= 1
        sight_uniform_color = True
        print("The screenshot is a uniform color.")
    else:
        sight_uniform_color = False
        # print("The screenshot has multiple colors.")

def get_sight_values():
    global sight_average_red, sight_average_green, sight_average_blue

    # Capture a screenshot
    screenshot = ImageGrab.grab()

    # Convert the screenshot to RGB mode (if it's not already)
    screenshot = screenshot.convert("RGB")

    # Compress the screenshot to 1x1 pixel
    screenshot = screenshot.resize((1, 1))

    # Get pixel data
    pixel_data = np.array(screenshot)

    # Get the RGB value of the compressed pixel
    pixel_color = pixel_data[0, 0]

    # Separate the RGB values into separate variables
    sight_average_red, sight_average_green, sight_average_blue = pixel_color[0], pixel_color[1], pixel_color[2]

    check_uniform_color(pixel_data)

def check_amplitude(audio_data):
    # Calculate the amplitude of the audio data
    rms = audioop.rms(audio_data, 2)  # 2 for format=PCM
    return rms

def get_word_index_in_csv(word, existing_words):
    # Check if the word exists in the CSV file
    if word in existing_words:
        return existing_words.index(word) + 1  # Return the index (1-based)
    return 0

def get_hearing_values():
    global hearing_amplitude, feedback, word_indices, heard_words

    hearing_amplitude = 0
    word_indices = [0] * 10

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say something:")
        try:
            audio = recognizer.listen(source, timeout=0.1)  # Listen for up to 5 seconds

            # Calculate the amplitude of the audio
            hearing_amplitude = check_amplitude(audio.frame_data)

            text = recognizer.recognize_google(audio)
            print("You said: " + text)

            blob = TextBlob(text)
            feedback += floor_positive_ceil_negative(blob.sentiment.polarity * 10)

            # Tokenize the text into words
            words = text.lower().split()  # Convert to lowercase

            heard_words = words[-10:]
            
            # Write the list of unique words to a CSV file
            write_heard_words_to_csv()

        except sr.UnknownValueError:
            print("Sorry, I could not understand your speech.")
            pass
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            pass
        except sr.WaitTimeoutError:
            print("Listening timed out. No speech detected.")
            pass

def write_heard_words_to_csv():
    global hearing_filename, word_indices, heard_words

    # Read the existing list of unique words from the CSV file
    existing_words = read_existing_words_from_csv()
    
    # Add new unique words to the existing list
    for word in heard_words:
        if word not in existing_words:
            existing_words.append(word)
    
    # Write the updated list of unique words to the CSV file
    with open(hearing_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for word in existing_words:
            writer.writerow([word])

    # Get the index of each word in the CSV or 0 if it doesn't exist
    word_indices = [get_word_index_in_csv(word, existing_words) for word in heard_words]

    # Add spaces to the list as needed
    while len(word_indices) < 10:
        word_indices.append(0)

def read_existing_words_from_csv():
    global hearing_filename

    existing_words = []
    
    # Check if the CSV file already exists
    if os.path.exists(hearing_filename):
        with open(hearing_filename, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                existing_words.append(row[0])
    
    return existing_words

def get_feedback_values():
    global feedback

    if feedback > 0:
        feedback -= 1
    elif feedback < 0:
        feedback += 1

# Function to calculate speed and angle between two cursor positions
def get_angle_and_speed():
    # Get the current cursor position
    x1, y1 = pyautogui.position()

    # Wait for a short period to get the next cursor position
    time.sleep(0.1)

    # Get the new cursor position
    x2, y2 = pyautogui.position()

    # Calculate the angle between the two points (in radians)
    angle = math.atan2(y2 - y1, x2 - x1)

    # Convert the angle to degrees and normalize it to the range [0, 360)
    angle_degrees = (math.degrees(angle) + 360) % 360

    # Calculate the distance between the two points (speed)
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return angle_degrees, dist

def move_cursor_smoothly(angle_degrees, speed, steps=100):
    # Initialize the mouse controller
    mouse = Controller()

    # Convert the angle from degrees to radians
    angle_rad = math.radians(angle_degrees)

    # Calculate the change in x and y coordinates based on speed and angle
    delta_x = speed * math.cos(angle_rad)
    delta_y = speed * math.sin(angle_rad)

    # Get the current cursor position
    current_x, current_y = mouse.position

    # Calculate the step size for x and y coordinates
    step_x = delta_x / steps
    step_y = delta_y / steps

    # Get the screen size using pyautogui
    screen_width, screen_height = pyautogui.size()

    # Move the cursor smoothly
    for _ in range(steps):
        current_x += step_x
        current_y += step_y

        # Ensure the cursor stays within the screen boundaries
        current_x = max(0, min(current_x, screen_width))
        current_y = max(0, min(current_y, screen_height))

        # Move the cursor to the new position
        mouse.position = (current_x, current_y)

        time.sleep(0.01)

def check_controlled_action():
    global touch_mouse
    global command
    global next_command  
    global argument_1st
    global argument_2nd
    global counter
    global action
    global controlled_action_flag
    global negative_keys_to_check
    global positive_keys_to_check

    angle, speed = get_angle_and_speed()

    if angle != 0 and speed != 0 and controlled_action_flag == False:
        next_command = 1
        argument_1st = 0
        argument_2nd = 0
        counter = 0
        action = 1
        controlled_action_flag = True
        touch_mouse = 1

def get_predicted_action_values(combined_without_action_data):
    global touch_mouse
    global command
    global next_command  
    global argument_1st
    global argument_2nd
    global counter
    global action
    global controlled_action_flag
    global negative_keys_to_check
    global positive_keys_to_check

    touch_mouse = 0

    if controlled_action_flag == False:
        # Load the existing model
        existing_model = joblib.load(model_filename)

        predicted_action = existing_model.predict(np.array(combined_without_action_data).reshape(1, -1))[0]

    if next_command == 0: # Nothing / Predict command
        command = 0
        if controlled_action_flag == False:
            next_command = floor_positive_absolute(predicted_action % 2)
        else:
            next_command = 0
        argument_1st = 0
        argument_2nd = 0
        action = next_command
        counter = 0
        check_controlled_action()
    elif next_command == 1: # Move Cursor command
        command = 1
        if controlled_action_flag:
            if counter == 0: # Angle
                angle, speed = get_angle_and_speed()
                action = angle
                argument_1st = action
                counter += 1
            elif counter == 1: # Speed & Execute
                angle, speed = get_angle_and_speed()
                action = speed
                argument_2nd = action
                next_command = 0
                counter += 1
                controlled_action_flag = False
            touch_mouse = 1
        else:
            if counter == 0: # Angle
                action = abs(predicted_action % 360)
                argument_1st = action
                counter += 1
            elif counter == 1: # Speed & Execute
                next_command = 0
                action = abs(predicted_action % 1080)
                argument_2nd = action
                counter += 1   

                print("Moving the cursor predictively...")
                move_cursor_smoothly(argument_1st, argument_2nd)

            check_controlled_action()      

def initialize_model(data, model_filename=model_filename):
    # Extract features (everything except the last element) and target (last element)
    X = np.array(data[:-1]).reshape(1, -1)
    y = np.array(data[-1])

    # Initialize the SGDRegressor with online learning using 'squared_error' loss
    model = SGDRegressor(loss='squared_error', penalty='l2', max_iter=1000, tol=1e-3, learning_rate='invscaling', eta0=0.01)

    # Online learning loop (just one iteration for your single data point)
    model.partial_fit(X, [y])

    # If the model file doesn't exist, save the current model
    joblib.dump(model, model_filename)

def train_sgd_regressor_online(data, model_filename=model_filename):
    # Extract features (everything except the last element) and target (last element)
    X = np.array(data[:-1]).reshape(1, -1)
    y = np.array(data[-1])

    # Check if the model file already exists
    if os.path.exists(model_filename):
        # Load the existing model
        existing_model = joblib.load(model_filename)
        
        # Update the existing model with the new data point
        existing_model.partial_fit(X, [y])
        
        # Save the updated model back to the same file, overwriting the previous model
        joblib.dump(existing_model, model_filename)
        print(f"Model updated and saved to {model_filename}")
    else:
        # Initialize the SGDRegressor with online learning using 'squared_error' loss
        model = SGDRegressor(loss='squared_error', penalty='l2', max_iter=1000, tol=1e-3, learning_rate='invscaling', eta0=0.01)

        # Online learning loop (just one iteration for your single data point)
        model.partial_fit(X, [y])

        # If the model file doesn't exist, save the current model
        joblib.dump(model, model_filename)
        print(f"Model saved to {model_filename}")

def get_features_to_list():
    global dataset_filename
    global model_filename

    global touch_keyboard
    global touch_mouse
    global touch_cursor_x
    global touch_cursor_y

    global sight_average_red
    global sight_average_green
    global sight_average_blue

    global hearing_amplitude

    global feedback
    global command
    global argument_1st
    global argument_2nd
    global counter
    global action

    global index_to_insert
    global word_indices

    get_touch_values()
    get_sight_values()
    get_hearing_values()
    get_feedback_values()

    without_action_data = [touch_keyboard, touch_mouse, touch_cursor_x, touch_cursor_y, sight_average_red, sight_average_green, sight_average_blue, hearing_amplitude, feedback, command, counter, argument_1st, argument_2nd]
    combined_without_action_data = without_action_data[:index_to_insert] + word_indices + without_action_data[index_to_insert:]
    get_predicted_action_values(combined_without_action_data) # get user Contolled actions

    # Define example data (input features and output)
    with_action_data = [touch_keyboard, touch_mouse, touch_cursor_x, touch_cursor_y, sight_average_red, sight_average_green, sight_average_blue, hearing_amplitude, feedback, command, counter, argument_1st, argument_2nd, action]
    combined_with_action_data = with_action_data[:index_to_insert] + word_indices + with_action_data[index_to_insert:]

    with open(dataset_filename, 'a', newline='') as file:
        writer = csv.writer(file)
    
        # Write example data to the CSV file
        writer.writerows([combined_with_action_data])

    if previous_controlled_action_flag == True and controlled_action_flag == False:
        # Create an empty list to store the data
        data = []

        # Open the CSV file for reading
        with open(dataset_filename, 'r') as file:
            # Create a CSV reader object
            csv_reader = csv.reader(file)

            next(csv_reader)  # Skip the header row
            
            # Iterate over each row in the CSV file
            for row in csv_reader:
                # Convert each element in the row to the appropriate data type if needed
                # For example, you can convert string values to float or int
                # Assuming all values in the CSV are floats, you can convert them like this:
                row = [float(value) for value in row]
                
                # Append the row to the data list
                data.append(row)

        train_sgd_regressor_online(data[0])
    elif controlled_action_flag == False:
        # Train the model with the sample data
        train_sgd_regressor_online(combined_with_action_data)

if __name__ == '__main__':
    if not os.path.exists(model_filename):
        initialize_model(combined_with_action_data)

    if not os.path.exists(dataset_filename):
        with open(dataset_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([variable_names])

    with KeyboardListener(on_press=on_key_press, on_release=on_key_release) as keyboard_listener:
        with MouseListener(on_click=on_click) as mouse_listener:
            try:
                while True:
                    get_features_to_list()
            except KeyboardInterrupt:
                print("KeyboardInterrupt: Stopping background processes...")
                keyboard_listener.stop()
                mouse_listener.stop()
