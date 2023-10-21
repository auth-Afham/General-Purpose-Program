import os
import pyautogui
import time
import math
from PIL import ImageGrab, Image
import joblib
from sklearn.linear_model import SGDRegressor
import numpy as np
from pynput.keyboard import Key, Listener as KeyboardListener
from pynput.mouse import Listener as MouseListener, Button, Controller
import csv
from collections import deque
import socket
import pyttsx3

HEARING_FILENAME = "hearing.csv"
DATASET_FILENAME = "dataset.csv"
MODEL_FILENAME = "model.pkl"

RESET_STARTING_FILES = False
RESET_TRAINING_FILES = True

# Sort by probability:

MAX_COMMAND_COUNT = 5

NOTHING_PREDICT_COMMAND = 0

# DESTROY_SPACE_COMMAND =  
# CREATE_SPACE_COMMAND =

# HEARING_FOCUS_COMMAND =
# SIGHT_FOCUS_COMMAND = 
# TOUCH_FOCUS_COMMAND = 

TEXT_TO_SPEECH_COMMAND = 1

# KEY_RELEASE_COMMAND = 
BUTTON_PRESS_COMMAND = 2

# MOUSE_RELEASE_COMMAND =
CURSOR_CLICK_COMMAND = 3
MOVE_CURSOR_COMMAND = 4

action = 0

touch_button = 0
touch_cursor = 0
touch_cursor_x = 0
touch_cursor_y = 0

hearing_amplitude = 0

feedback = 0

command = 0
next_command = 0
counter = 0

argument_1 = 0
argument_2 = 0

previous_angle = 0
previous_speed = 0
previous_touch_cursor = 0
previous_touch_button = 0

PIXEL_COUNT = 432 # = 16 * 9 * 3 = height * width * rgb
WORD_COUNT = 9 

sight_pixels = [0] * PIXEL_COUNT 
hearing_words = [0] * WORD_COUNT
with_action_data = [action, touch_button, touch_cursor, touch_cursor_x, touch_cursor_y, hearing_amplitude, feedback, command, counter, argument_1, argument_2]

# Insert the additional_list between elements 3 and 4 of the original_list
combined_with_action_data = with_action_data + sight_pixels + hearing_words

user_action_flag = False
user_mouse_click_flag = False
user_button_press_flag = False

# 16:9 == 1920:1080
# Desired width
DESIRED_WIDTH = 16  # Change this to your desired width

# Calculate the height to maintain aspect ratio
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
ASPECT_RATIO = SCREEN_WIDTH / SCREEN_HEIGHT
DESIRED_HEIGHT = int(DESIRED_WIDTH / ASPECT_RATIO)

# Initialize a list to store the last 10 heard words
heard_words = deque(maxlen=WORD_COUNT)

# List of variable names
variable_names = [
    "action",
    "touch_button", "touch_cursor", "touch_cursor_x", "touch_cursor_y",
    "hearing_amplitude",
    "feedback", "command", "counter", "argument_1", "argument_2",
]

sight_pixel_headers = [f"s{i}" for i in range(1, PIXEL_COUNT + 1)]
hearing_word_headers = [f"h{i}" for i in range(1, WORD_COUNT + 1)]

combined_variable_names = variable_names + sight_pixel_headers + hearing_word_headers

negative_keys_to_check = ['backspace', 'del', 'esc', 'ctr2Cl+c', 'ctrl+z', 'f1', 'f4', 'f7', '-', '/', '!', 'capslock']

# Define a list of keys you want to check
positive_keys_to_check = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'up', 'down', 'left', 'right',  # Arrow keys
    'space', 'enter', 'shift', 'ctrl', 'alt',  # Other keys
]

combined_keys_to_check = negative_keys_to_check + positive_keys_to_check

# Function to format the key for display
def format_key(key):
    if hasattr(key, 'name'):
        return key.name
    else:
        return str(key).strip("'")

# Callback function for key presses
def on_key_press(key):
    global touch_button, feedback
    global previous_touch_button
    global user_button_press_flag
    global negative_keys_to_check, positive_keys_to_check
    
    # Format the key for display
    formatted_key = format_key(str(key))

    if formatted_key in negative_keys_to_check:
        touch_button = negative_keys_to_check.index(formatted_key) + 1
        feedback -= 1
        print(f"'{formatted_key}' is pressed at index {touch_button}!")
        previous_touch_button = touch_button
        user_button_press_flag = True

    if formatted_key in positive_keys_to_check:
        touch_button = len(negative_keys_to_check) + positive_keys_to_check.index(formatted_key) + 1
        feedback += 1
        print(f"'{formatted_key}' is pressed at index {touch_button}!")
        previous_touch_button = touch_button
        user_button_press_flag = True

# Callback function for key releases
def on_key_release(key):
    global touch_button

    # Do something when a key is released
    touch_button = 0

# Callback function for mouse clicks
def on_click(x, y, button, pressed):
    global touch_cursor, feedback
    global previous_touch_cursor
    global user_mouse_click_flag
    global negative_keys_to_check, positive_keys_to_check

    if pressed:
        if button == Button.left:
            touch_cursor = 2 # You can assign any unique value you like
            print(f"Left mouse button clicked at index {touch_cursor}!")
        elif button == Button.right:
            touch_cursor = 3 # You can assign any unique value you like
            print(f"Right mouse button clicked at index {touch_cursor}!")
        feedback += 1  # You can assign the appropriate feedback value
        previous_touch_cursor = touch_cursor
        user_mouse_click_flag = True

def get_touch_values():
    global touch_cursor_x, touch_cursor_y
    
    # Initialize the mouse controller
    mouse = Controller()

    # Get the current cursor position
    touch_cursor_x, touch_cursor_y = mouse.position

def check_uniform_color(pixel_data):
    global feedback

    # Get the RGB value of the first pixel
    first_pixel_color = pixel_data[0]

    # Check if all pixels have the same color as the first pixel
    if np.all(pixel_data == first_pixel_color):
        feedback -= 1

def get_sight_values():
    global sight_pixels, DESIRED_WIDTH, DESIRED_HEIGHT

    # Capture a screenshot of the entire screen
    screenshot = ImageGrab.grab()

    # Resize the screenshot to the desired dimensions
    screenshot = screenshot.resize((DESIRED_WIDTH, DESIRED_HEIGHT), Image.Resampling.LANCZOS)

    # Convert the screenshot to RGB mode (if it's not already)
    screenshot = screenshot.convert("RGB")

    # Get pixel data
    pixel_data = np.array(screenshot)

    sight_pixels = []
    # Iterate through pixel_data and write each pixel's data to the CSV
    for row in range(DESIRED_HEIGHT):
        for col in range(DESIRED_WIDTH):
            for rgb in range(3):
                sight_pixels.append(pixel_data[row, col, rgb])

    check_uniform_color(pixel_data)

def read_existing_words_from_csv():
    global HEARING_FILENAME

    existing_words = []
    
    # Check if the CSV file already exists
    if os.path.exists(HEARING_FILENAME):
        with open(HEARING_FILENAME, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                existing_words.append(row[0])
    
    return existing_words

def get_word_index_in_csv(word, existing_words):
    # Check if the word exists in the CSV file
    if word in existing_words:
        return existing_words.index(word) + 1  # Return the index (1-based)
    return 0

def write_heard_words_to_csv():
    global HEARING_FILENAME, WORD_COUNT, hearing_words, heard_words

    # Read the existing list of unique words from the CSV file
    existing_words = read_existing_words_from_csv()
    
    # Add new unique words to the existing list
    for word in heard_words:
        if word not in existing_words:
            existing_words.append(word)
    
    # Write the updated list of unique words to the CSV file
    with open(HEARING_FILENAME, 'w', newline='') as file:
        writer = csv.writer(file)
        for word in existing_words:
            writer.writerow([word])

    # Get the index of each word in the CSV or 0 if it doesn't exist
    hearing_words = [get_word_index_in_csv(word, existing_words) for word in heard_words]

    # Add spaces to the list as needed
    while len(hearing_words) < WORD_COUNT:
        hearing_words.append(0)

def get_hearing_receiver():
    global hearing_values_socket, hearing_amplitude, feedback, WORD_COUNT, hearing_words, heard_words

    hearing_amplitude = 0
    hearing_words = [0] * WORD_COUNT

    # Receive hearing values from the hearing values script
    hearing_values_socket.settimeout(0.1)  # Set a timeout of 0.1 seconds
    try:
        # Receive the data from the client script
        data = hearing_values_socket.recv(1024).decode()

        # Split the received data into individual values
        values = data.split(',')
        
        # Extract the values
        hearing_amplitude = float(values[0])
        feedback = float(values[1])
        # {','.join(map(str, heard_words))}
        heard_words = values[2:]
        # print(heard_words)

        # You can now use hearing_amplitude, feedback, and heard_words as needed

        # Write the list of unique words to a CSV file
        write_heard_words_to_csv()
    except socket.timeout:
        # No data received in 0.1 seconds, continue processing or do nothing
        pass

def get_feedback_values():
    global feedback

    if feedback > 0:
        feedback -= 1
    elif feedback < 0:
        feedback += 1

# Function to calculate speed and angle between two cursor positions
def get_angle_and_speed():
    global previous_angle, previous_speed

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

    previous_angle, previous_speed = angle_degrees, dist

    return angle_degrees, dist

def check_controlled_action(bot_action_flag):
    global NOTHING_PREDICT_COMMAND, CURSOR_CLICK_COMMAND, MOVE_CURSOR_COMMAND
    global action
    global touch_cursor, touch_button
    global command, next_command
    global counter
    global argument_1, argument_2
    global previous_angle, previous_speed, previous_touch_cursor
    global user_action_flag, user_mouse_click_flag, user_button_press_flag
    global negative_keys_to_check, positive_keys_to_check

    angle, speed = get_angle_and_speed()

    if not user_action_flag and not bot_action_flag:
        if user_button_press_flag:
            action = BUTTON_PRESS_COMMAND
            command = NOTHING_PREDICT_COMMAND
            next_command = BUTTON_PRESS_COMMAND
            counter = 0
            argument_1 = 0
            argument_2 = 0
            user_action_flag = True
        elif user_mouse_click_flag:
            action = CURSOR_CLICK_COMMAND
            command = NOTHING_PREDICT_COMMAND
            next_command = CURSOR_CLICK_COMMAND
            counter = 0
            argument_1 = 0
            argument_2 = 0
            user_action_flag = True
        elif angle != 0 and speed != 0:
            action = CURSOR_CLICK_COMMAND
            command = NOTHING_PREDICT_COMMAND
            next_command = MOVE_CURSOR_COMMAND
            counter = 0
            argument_1 = 0
            argument_2 = 0
            user_action_flag = True

def simulate_text_to_speech(text):
    # Initialize the TTS engine
    tts_engine = pyttsx3.init()

    # Use the TTS engine to speak the text
    tts_engine.say(text)
    tts_engine.runAndWait()

def simulate_button_press(touch_button):
    global positive_keys_to_check, combined_keys_to_check
    # Simulate key presses based on the touch_button value
    # You can map touch_button values to specific key presses here
    # For example:
    print("simulate_button_press = ", positive_keys_to_check[touch_button % len(positive_keys_to_check)])
    pyautogui.press(combined_keys_to_check[touch_button])
    # Add more cases for other keys you want to simulate

def simulate_mouse_click(touch_cursor):
    if touch_cursor == 2:
        pyautogui.click(button='left')
    elif touch_cursor == 3:
        pyautogui.click(button='right')

def simulate_moving_cursor(angle_degrees, speed, steps=100, safety_margin=10):
    global SCREEN_WIDTH, SCREEN_HEIGHT

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

    # Move the cursor smoothly
    for _ in range(steps):
        current_x += step_x
        current_y += step_y

        # Ensure the cursor stays within the screen boundaries
        current_x = max(safety_margin, min(current_x, SCREEN_WIDTH - safety_margin))
        current_y = max(safety_margin, min(current_y, SCREEN_HEIGHT - safety_margin))

        # Move the cursor to the new position
        mouse.position = (current_x, current_y)

        time.sleep(0.01)

def get_predicted_action_values(combined_without_action_data):
    global MAX_COMMAND_COUNT, NOTHING_PREDICT_COMMAND, CURSOR_CLICK_COMMAND, MOVE_CURSOR_COMMAND
    global action
    global touch_cursor
    global command, next_command
    global counter
    global argument_1, argument_2
    global hearing_amplitude, hearing_words
    global previous_angle, previous_speed, previous_touch_cursor, previous_touch_button
    global user_action_flag, user_mouse_click_flag, user_button_press_flag
    global negative_keys_to_check, positive_keys_to_check, combined_keys_to_check

    touch_cursor = 0
    touch_button = 0
    bot_action_flag = False

    if user_action_flag == False:
        # Load the existing model
        existing_model = joblib.load(MODEL_FILENAME)
        predicted_action = existing_model.predict(np.array(combined_without_action_data).reshape(1, -1))[0]

    if next_command == NOTHING_PREDICT_COMMAND: # Nothing / Predict command
        command = NOTHING_PREDICT_COMMAND
        if user_action_flag == False:
            next_command = math.floor(abs(predicted_action)) % MAX_COMMAND_COUNT
            print("next_command = ", next_command)
        else:
            next_command = NOTHING_PREDICT_COMMAND
        if next_command == 1 and len(read_existing_words_from_csv()) == 0:
            next_command = NOTHING_PREDICT_COMMAND
        action = next_command
        counter = 0
        argument_1 = argument_2 = 0
        check_controlled_action(bot_action_flag)
    elif next_command == TEXT_TO_SPEECH_COMMAND: # Text-to-Speech command
        hearing_amplitude = 0
        hearing_words = [0] * WORD_COUNT
        command = TEXT_TO_SPEECH_COMMAND
        action = math.floor(abs(predicted_action)) % len(read_existing_words_from_csv())
        print("action = ", action)
        counter += 1
        argument_1 = action
        hearing_words[0] = argument_1
        next_command = NOTHING_PREDICT_COMMAND
        simulate_text_to_speech(read_existing_words_from_csv()[argument_1])
    elif next_command == BUTTON_PRESS_COMMAND: # Key Press command
        command = BUTTON_PRESS_COMMAND
        if user_action_flag:
            touch_button = previous_touch_button
            action = touch_button
            counter += 1
            argument_1 = action
            next_command = NOTHING_PREDICT_COMMAND
            user_button_press_flag = False
            user_action_flag = False
        else:
            action = int(math.floor(abs(predicted_action)) % len(combined_keys_to_check))
            print("action = ", action)
            touch_button = action + 1
            counter += 1
            argument_1 = action
            next_command = NOTHING_PREDICT_COMMAND
            simulate_button_press(argument_1)
            # bot_action_flag = True
            # check_controlled_action(bot_action_flag)
    elif next_command == CURSOR_CLICK_COMMAND: # Mouse Click command
        command = CURSOR_CLICK_COMMAND
        if user_action_flag:
            touch_cursor = previous_touch_cursor
            action = touch_cursor
            counter += 1
            argument_1 = action
            next_command = NOTHING_PREDICT_COMMAND
            user_mouse_click_flag = False
            user_action_flag = False
        else:
            action = math.floor(abs(predicted_action)) % 2 + 2
            print("action = ", action)
            touch_cursor = action
            counter += 1
            argument_1 = action
            next_command = NOTHING_PREDICT_COMMAND
            simulate_mouse_click(argument_1)
            # bot_action_flag = True
            # check_controlled_action(bot_action_flag)    
    elif next_command == MOVE_CURSOR_COMMAND: # Move Cursor command
        command = MOVE_CURSOR_COMMAND
        if user_action_flag:
            if counter == 0: # Angle
                action = previous_angle
                counter += 1
                argument_1 = action
            elif counter == 1: # Speed & Execute
                action = previous_speed
                touch_cursor = 1
                counter += 1
                argument_2 = action
                next_command = NOTHING_PREDICT_COMMAND
                user_action_flag = False
        else:
            if counter == 0: # Angle
                action = abs(predicted_action) % 360
                counter += 1
                argument_1 = action
            elif counter == 1: # Speed & Execute
                action = abs(predicted_action) % 1080
                touch_cursor = 1
                counter += 1
                argument_2 = action  
                next_command = NOTHING_PREDICT_COMMAND
                simulate_moving_cursor(argument_1, argument_2)
                bot_action_flag = True
            print("action = ", action)
            check_controlled_action(bot_action_flag)    

def initialize_model(data, model_filename=MODEL_FILENAME):
    # Extract features (everything except the last element) and target (last element)
    X = np.array(data[1:]).reshape(1, -1)
    y = np.array(data[0])

    # Initialize the SGDRegressor with online learning using 'squared_error' loss
    model = SGDRegressor(loss='squared_error', penalty='l2', max_iter=1000, tol=1e-3, learning_rate='invscaling', eta0=0.01)

    # Online learning loop (just one iteration for your single data point)
    model.partial_fit(X, [y])

    # If the model file doesn't exist, save the current model
    joblib.dump(model, MODEL_FILENAME)

    print(f"Model saved to {MODEL_FILENAME}")

def train_sgd_regressor_online(data, model_filename=MODEL_FILENAME):
    # Extract features (everything except the last element) and target (last element)
    X = np.array([row[1:] for row in data])  # Extract features for all rows
    y = np.array([row[0] for row in data])   # Extract targets for all rows

    # Load the existing model
    existing_model = joblib.load(MODEL_FILENAME)
    
    # Update the existing model with the new data points
    # Use ravel() to convert y to a 1D array
    existing_model.partial_fit(X, y)
    
    # Save the updated model back to the same file, overwriting the previous model
    joblib.dump(existing_model, MODEL_FILENAME)

def get_features_to_list():
    global DATASET_FILENAME, MODEL_FILENAME
    global RESET_STARTING_FILES, RESET_TRAINING_FILES
    global action
    global touch_button, touch_cursor
    global touch_cursor_x, touch_cursor_y
    global hearing_amplitude
    global feedback
    global command
    global counter
    global argument_1, argument_2
    global sight_pixels, hearing_words
    global user_action_flag

    get_touch_values()
    get_sight_values()
    get_hearing_receiver()
    get_feedback_values()

    # print(hearing_words)

    without_action_data = [touch_button, touch_cursor, touch_cursor_x, touch_cursor_y, hearing_amplitude, feedback, command, counter, argument_1, argument_2]
    combined_without_action_data = without_action_data + sight_pixels + hearing_words
    get_predicted_action_values(combined_without_action_data) # get user Contolled actions

    # Define example data (input features and output)
    with_action_data = [action, touch_button, touch_cursor, touch_cursor_x, touch_cursor_y, hearing_amplitude, feedback, command, counter, argument_1, argument_2]
    combined_with_action_data = with_action_data + sight_pixels + hearing_words

    with open(DATASET_FILENAME, 'a', newline='') as file:
        writer = csv.writer(file)
    
        # Write example data to the CSV file
        writer.writerows([combined_with_action_data])

    if not user_action_flag:
        # Create an empty list to store the data
        data = []

        # Open the CSV file for reading
        with open(DATASET_FILENAME, 'r') as file:
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

        train_sgd_regressor_online(data)
 
        if RESET_TRAINING_FILES:
            with open(DATASET_FILENAME, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows([combined_variable_names])

if __name__ == '__main__':
    # Initialize a socket server to listen for hearing values
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(('localhost', 12345))  # Bind to a specific address and port
    server_socket.listen(1)  # Listen for incoming connections

    print("Waiting for a connection...")

    # Accept a connection from the hearing values script
    hearing_values_socket, _ = server_socket.accept()
    print("Connected to the hearing values script.")

    if RESET_STARTING_FILES:
        initialize_model(combined_with_action_data)

        with open(DATASET_FILENAME, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([combined_variable_names])

    with KeyboardListener(on_press=on_key_press, on_release=on_key_release) as keyboard_listener:
        with MouseListener(on_click=on_click) as mouse_listener:
            try:
                while True:
                    print("---")
                    get_features_to_list()
            except KeyboardInterrupt:
                print("Program ended due to Ctrl+C in the terminal.")
                # Clean up and close sockets
                keyboard_listener.stop()
                mouse_listener.stop()
                hearing_values_socket.close()
                server_socket.close()
                exit(0)