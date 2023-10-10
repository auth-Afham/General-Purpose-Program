import os
import pyautogui
import time
import math
from PIL import ImageGrab
import joblib
from sklearn.linear_model import SGDRegressor
import numpy as np
import random
import pyaudio
import keyboard  # Import the keyboard library
from pynput.keyboard import Key, Listener
from pynput.mouse import Controller, Listener, Button
import csv
import speech_recognition as sr
import audioop
from textblob import TextBlob
import threading

model_filename = "model.pkl"
csv_filename = "spreadsheet.csv"

touch_key = 0
touch_cursor_x = 0
touch_cursor_y = 0

sight_average_red = 0
sight_average_green = 0
sight_average_blue = 0
sight_uniform_color = False

hearing_amplitude = 0

feedback = 0

command = 0
next_command = 0

argument_list = []
arguments = 0

counter = 0
action = 0

previous_controlled_action_flag = False
controlled_action_flag = False

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

# Callback function for mouse clicks
def on_click(x, y, button, pressed):
    global touch_key
    global feedback
    global negative_keys_to_check
    global positive_keys_to_check

    touch_key = 0

    if pressed:
        if button == Button.left:
            touch_key = len(negative_keys_to_check) + len(positive_keys_to_check) + 1 # You can assign any unique value you like
            feedback += 1  # You can assign the appropriate feedback value
            print("Left mouse button clicked!")
        elif button == Button.right:
            touch_key = len(negative_keys_to_check) + len(positive_keys_to_check) + 2 # You can assign any unique value you like
            feedback += 1  # You can assign the appropriate feedback value
            print("Right mouse button clicked!")

# Create a listener for mouse clicks
mouse_listener = Listener(on_click=on_click)

# Start the mouse listener in the background
mouse_listener.start()

# Function to format the key for display
def format_key(key):
    if hasattr(key, 'name'):
        return key.name
    else:
        return str(key).strip("'")

# Callback function for key presses
def on_key_press(key):
    global touch_key
    global feedback
    global next_command    
    global action
    global negative_keys_to_check
    global positive_keys_to_check

    touch_key = 0
    
    # Format the key for display
    formatted_key = format_key(key)
    # Do something when a key is pressed
    print(f'Key {formatted_key} pressed')

    for index, key in enumerate(negative_keys_to_check):
        if keyboard.is_pressed(key):
            touch_key = index + 1
            feedback -= 1
            next_command = 0            
            action = 0
            print(f"'{key}' is pressed at index {touch_key}!")
            break

    for index, key in enumerate(positive_keys_to_check):
        if keyboard.is_pressed(key):
            touch_key = len(negative_keys_to_check) + index + 1
            feedback += 1
            print(f"'{key}' is pressed at index {touch_key}!")
            break

# Callback function for key releases
def on_key_release(key):
    # Do something when a key is released
    formatted_key = format_key(key)
    print(f'Key {formatted_key} released')

    touch_key = 0

# Function to check for keypresses
def keyboard_listener():
    # Create a listener for keyboard events
    with Listener(on_press=on_key_press, on_release=on_key_release) as listener:
        # Start the keyboard listener in the background
        listener.join()

# Create a thread for the keyboard listener
keyboard_thread = threading.Thread(target=keyboard_listener)

# Start the keyboard thread
keyboard_thread.start()

def get_touch_values():
    global touch_cursor_x
    global touch_cursor_y
    
    # Initialize the mouse controller
    mouse = Controller()

    # Get the current cursor position
    touch_cursor_x, touch_cursor_y = mouse.position

def check_uniform_color(pixel_data):
    global sight_uniform_color
    global feedback
    global next_command
    global action

    # Get the RGB value of the first pixel
    first_pixel_color = pixel_data[0]

    # Check if all pixels have the same color as the first pixel
    if np.all(pixel_data == first_pixel_color):
        feedback -= 1
        next_command = 0
        action = 0
        sight_uniform_color = True
        print("The screenshot is a uniform color.")
    else:
        sight_uniform_color = False
        print("The screenshot has multiple colors.")

def get_sight_values():
    global sight_average_red
    global sight_average_green
    global sight_average_blue

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

    # check_uniform_color(pixel_data)

def check_amplitude(audio_data):
    # Calculate the amplitude of the audio data
    rms = audioop.rms(audio_data, 2)  # 2 for format=PCM
    return rms

def get_hearing_values():
    global hearing_amplitude
    global feedback
    global next_command
    global action

    hearing_amplitude = 0

    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say something:")
        try:
            audio = recognizer.listen(source, timeout=0.1)  # Listen for up to 5 seconds
            print("Listening...")

            # Calculate the amplitude of the audio
            hearing_amplitude = check_amplitude(audio.frame_data)
            print(f"Amplitude: {hearing_amplitude}")

            text = recognizer.recognize_google(audio)
            print("You said: " + text)

            blob = TextBlob(text)
            feedback += floor_positive_ceil_negative(blob.sentiment.polarity * 10)

            if feedback < 0:
                next_command = 0
                action = 0

            print(f"Polarity Score: {blob.sentiment.polarity}")

        except sr.UnknownValueError:
            print("Sorry, I could not understand your speech.")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        except sr.WaitTimeoutError:
            print("Listening timed out. No speech detected.")

def get_feedback_values():
    global touch_key
    global sight_uniform_color

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

    print(f"Angle: {angle_degrees} degrees, Speed: {dist} pixels/second")

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
    global touch_key
    global command
    global next_command    
    global argument_list
    global arguments
    global counter
    global action
    global controlled_action_flag
    global negative_keys_to_check
    global positive_keys_to_check

    angle, speed = get_angle_and_speed()

    if angle != 0 and speed != 0 and controlled_action_flag == False:
        next_command = 1
        counter = 0
        argument_list = []
        arguments = 0
        action = 1
        controlled_action_flag = True
        touch_key = len(negative_keys_to_check) + len(positive_keys_to_check) + 3

def get_predicted_action_values(without_action_data):
    global touch_key
    global command
    global next_command    
    global argument_list
    global arguments
    global counter
    global action
    global controlled_action_flag
    global negative_keys_to_check
    global positive_keys_to_check

    touch_key = 0

    if controlled_action_flag == False:
        # Load the existing model
        existing_model = joblib.load(model_filename)

        predicted_action = existing_model.predict(np.array(without_action_data).reshape(1, -1))[0]

    if next_command == 0: # Nothing / Predict command
        command = 0
        if controlled_action_flag == False:
            next_command = floor_positive_absolute(predicted_action % 2)
        else:
            next_command = 0
        argument_list = []
        arguments = 0
        action = next_command
        counter = 0
        check_controlled_action()
    elif next_command == 1: # Move Cursor command
        command = 1
        if controlled_action_flag:
            if counter == 0: # Angle
                angle, speed = get_angle_and_speed()
                action = angle
                argument_list.append(action)

                # Convert any negative data in argument_list into positive data
                argument_list_positive = [abs(x) for x in argument_list]

                # Floor the data in argument_list
                argument_list_floor = [math.floor(x) for x in argument_list_positive]

                # Convert argument_list to string and zfill each data by 3 with 0
                argument_list_string = [str(x).zfill(4) for x in argument_list_floor]

                # F-string together data in argument_list into one string
                argument_string = ''.join(argument_list_string)

                # Convert the fstring to integer
                arguments = int(argument_string)

                counter += 1
            elif counter == 1: # Speed & Execute
                angle, speed = get_angle_and_speed()
                action = speed
                argument_list.append(action)
                
                # Convert any negative data in argument_list into positive data
                argument_list_positive = [abs(x) for x in argument_list]

                # Floor the data in argument_list
                argument_list_floor = [math.floor(x) for x in argument_list_positive]

                # Convert argument_list to string and zfill each data by 3 with 0
                argument_list_string = [str(x).zfill(4) for x in argument_list_floor]

                # F-string together data in argument_list into one string
                argument_string = ''.join(argument_list_string)

                # Convert the fstring to integer
                arguments = int(argument_string)

                next_command = 0
                counter += 1
                controlled_action_flag = False
            touch_key = len(negative_keys_to_check) + len(positive_keys_to_check) + 3
        else:
            if counter == 0: # Angle
                action = abs(predicted_action % 360)
                argument_list.append(action)

                # Convert any negative data in argument_list into positive data
                argument_list_positive = [abs(x) for x in argument_list]

                # Floor the data in argument_list
                argument_list_floor = [math.floor(x) for x in argument_list_positive]

                # Convert argument_list to string and zfill each data by 3 with 0
                argument_list_string = [str(x).zfill(4) for x in argument_list_floor]

                # F-string together data in argument_list into one string
                argument_string = ''.join(argument_list_string)

                # Convert the fstring to integer
                arguments = int(argument_string)

                counter += 1
            elif counter == 1: # Speed & Execute
                next_command = 0

                action = abs(predicted_action % 1080)
                argument_list.append(action)
                
                # Convert any negative data in argument_list into positive data
                argument_list_positive = [abs(x) for x in argument_list]

                # Floor the data in argument_list
                argument_list_floor = [math.floor(x) for x in argument_list_positive]

                # Convert argument_list to string and zfill each data by 3 with 0
                argument_list_string = [str(x).zfill(4) for x in argument_list_floor]

                # F-string together data in argument_list into one string
                argument_string = ''.join(argument_list_string)

                # Convert the fstring to integer
                arguments = int(argument_string)

                counter += 1   

                print("Moving the cursor predictively...")
                move_cursor_smoothly(argument_list[0], argument_list[1])

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
    global touch_key
    global touch_cursor_x
    global touch_cursor_y

    global sight_average_red
    global sight_average_green
    global sight_average_blue

    global hearing_amplitude

    global feedback
    global command
    global arguments
    global counter
    global action

    global model_filename
    global csv_filename

    get_touch_values()
    get_sight_values()
    get_hearing_values()
    get_feedback_values()

    without_action_data = [touch_key, touch_cursor_x, touch_cursor_y, sight_average_red, sight_average_green, sight_average_blue, hearing_amplitude, feedback, command, arguments, counter]
    get_predicted_action_values(without_action_data) # get user Contolled actions

    with open(csv_filename, 'a', newline='') as file:
        writer = csv.writer(file)
    
        # Define example data (input features and output)
        with_action_data = [[touch_key, touch_cursor_x, touch_cursor_y, sight_average_red, sight_average_green, sight_average_blue, hearing_amplitude, feedback, command, arguments, counter, action]]
    
        # Write example data to the CSV file
        writer.writerows(with_action_data)

    if previous_controlled_action_flag == True and controlled_action_flag == False:
        # Create an empty list to store the data
        data = []

        # Open the CSV file for reading
        with open(csv_filename, 'r') as file:
            # Create a CSV reader object
            csv_reader = csv.reader(file)
            
            # Iterate over each row in the CSV file
            for row in csv_reader:
                # Convert each element in the row to the appropriate data type if needed
                # For example, you can convert string values to float or int
                # Assuming all values in the CSV are floats, you can convert them like this:
                row = [float(value) for value in row]
                
                # Append the row to the data list
                data.append(row)

        # Now, 'data' contains your CSV data as a list of lists
        # Each inner list represents a row of data

        train_sgd_regressor_online(data[0])
    elif controlled_action_flag == False:
        # Train the model with the sample data
        train_sgd_regressor_online(with_action_data[0])

initialize_data = [touch_key, touch_cursor_x, touch_cursor_y, sight_average_red, sight_average_green, sight_average_blue, hearing_amplitude, feedback, command, arguments, counter, action]

initialize_model(initialize_data)

with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)

while True:
    previous_controlled_action_flag = controlled_action_flag

    print("---")
    get_features_to_list()