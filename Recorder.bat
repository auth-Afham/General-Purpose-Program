@echo off

:: Start the main_receiver script in the background
start /B cmd /C "python 3_main_receiver.py"

:: Add a delay of 2 seconds (you can adjust the time as needed)
timeout /t 2

:: Start the hearing_sender script in the background
start /B cmd /C "python 3_hearing_sender.py"

:: Wait for user input to end the program
echo Press Ctrl+C to terminate the program.
pause

:: Terminate the background processes (main_receiver and hearing_sender) when the user presses Ctrl+C
taskkill /IM python.exe /F
