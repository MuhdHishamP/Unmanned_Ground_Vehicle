import RPi.GPIO as GPIO
import time

# Set GPIO mode to BCM
GPIO.setmode(GPIO.BOARD)

# Set up GPIO pins for trigger and echo
trigger_pin = 12
echo_pin = 16
GPIO.setup(trigger_pin, GPIO.OUT)
GPIO.setup(echo_pin, GPIO.IN)

# Function to measure distance with HC-SR04
def distance():
    # Send a 10us pulse to trigger pin
    GPIO.output(trigger_pin, True)
    time.sleep(0.00001)
    GPIO.output(trigger_pin, False)

    # Wait for echo pin to go high
    while GPIO.input(echo_pin) == 0:
        pulse_start = time.time()

    # Wait for echo pin to go low
    while GPIO.input(echo_pin) == 1:
        pulse_end = time.time()

    # Calculate pulse duration and convert to distance
    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    distance = round(distance, 2)
    
    return distance

# Print distance every second
while True:
    dist = distance()
    print("Distance:", dist, "cm")
    time.sleep(1)
