# Touch-and-Talk System

Here's an image of my project:

![Project Image](https://pbs.twimg.com/media/GBJw4trWIAASN_w?format=jpg&name=medium)

This is a Python script for a Touch-and-Talk system that allows users to interact with their computer through touch and voice commands. The system uses machine learning to predict and execute user actions based on touch and hearing inputs. The script captures touch events (keyboard and mouse) and hearing inputs, processes the data, and trains an online machine learning model for predicting user actions.

## Features

- **Touch Inputs:** Capture touch events such as keyboard presses and mouse clicks.
- **Voice Inputs:** Receive hearing values from a separate script, allowing voice commands.
- **Machine Learning:** Utilize an online SGDRegressor model to predict and execute user actions.
- **Text-to-Speech:** Simulate text-to-speech functionality for voice feedback.

## Prerequisites

Before running the script, make sure you have the following Python packages installed:

```bash
pip install pyautogui Pillow joblib numpy pynput pyttsx3
```

## Usage

1. **Configure Parameters:** Modify the script's parameters at the beginning according to your preferences.

```python
# Parameters
HEARING_FILENAME = "hearing.csv"
DATASET_FILENAME = "dataset.csv"
MODEL_FILENAME = "model.pkl"

RESET_STARTING_FILES = False
RESET_TRAINING_FILES = True
# ... (other parameters)
```

2. **Run the Script:** Execute the script using the Python interpreter.

```bash
python main_receiver.py
```

3. **Interact with the System:** Once the script is running, you can interact with your computer through touch (keyboard and mouse) and voice commands. The script will continuously capture and process inputs to predict and execute user actions.

4. **Voice Commands:** To use voice commands, you need a separate script that sends hearing values to the specified socket (localhost:12345). The system includes functionality for text-to-speech and can simulate button presses and mouse clicks based on predictions.

## Notes

- The script uses an online SGDRegressor model for online learning, allowing continuous updates based on user interactions.
- Ensure that the required libraries are installed, and the script has the necessary permissions to capture touch and hearing inputs.

Feel free to explore and customize the script based on your specific use case and requirements.
