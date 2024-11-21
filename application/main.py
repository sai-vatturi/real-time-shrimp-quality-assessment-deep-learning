import requests
from picamera2 import Picamera2
from PIL import Image
from io import BytesIO
import RPi.GPIO as GPIO
from time import sleep
from RPLCD.gpio import CharLCD

# Server URL
API_URL = "http://34.234.97.27:8080/detect/"

# GPIO Pins for LCD
lcd = CharLCD(
    numbering_mode=GPIO.BCM,
    pin_rs=20,       # RS pin to GPIO 20 (Pin 38)
    pin_e=21,        # E pin to GPIO 21 (Pin 40)
    pins_data=[16, 12, 13, 19],  # Data pins to GPIO 16, 12, 13, 19 (Pins 36, 32, 33, 35)
    cols=16,         # Number of columns on your LCD
    rows=2,          # Number of rows on your LCD
    dotsize=8        # Dotsize (usually 8 for 16x2 LCDs)
)

# GPIO Pin for Buzzer
BUZZER_PIN = 18  # GPIO 18 (Pin 12)

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

# Initialize the camera
picam2 = Picamera2()

# Classification categories
BAD_SHRIMP_CLASSES = [
    "BrokenTail",
    "BlackTail",
    "Vain",
    "Hanging Meat",
    "ImproperPeeling",
    "Vain"
]

GOOD_SHRIMP_CLASSES = [
    "GoodEZPL",
    "GoodHEADLESS",
    "GoodPD",
    "GoodPDTO"
]


def beep_buzzer(times=1):
    """Beep the buzzer a specified number of times."""
    for _ in range(times):
        GPIO.output(BUZZER_PIN, GPIO.HIGH)  # Turn buzzer on
        sleep(0.2)  # Beep duration
        GPIO.output(BUZZER_PIN, GPIO.LOW)  # Turn buzzer off
        sleep(0.2)  # Pause between beeps


def capture_image():
    """Capture an image, resize it to 640x640 while maintaining aspect ratio, and return it as a JPEG buffer."""
    print("Capturing image...")
    # Configure the camera for still image capture
    picam2.configure(picam2.create_still_configuration(main={"size": (640, 480)}))  # Original resolution
    picam2.start()

    # Capture the image as a NumPy array
    img_array = picam2.capture_array()
    picam2.stop()

    # Convert NumPy array to JPEG using Pillow
    img = Image.fromarray(img_array)
    # Scale down the image to fit 640x640 while keeping the aspect ratio
    img.thumbnail((640, 640))

    # Save the image to a BytesIO buffer
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)

    print("Image captured and resized.")
    return buffer


def send_image_to_server(image_buffer):
    """Send the image to the server and return the response."""
    print("Sending image to server...")
    try:
        # Send the image as a POST request
        files = {"file_list": ("image.jpg", image_buffer, "image/jpeg")}
        response = requests.post(API_URL, files=files)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error sending image to server: {e}")
        return None


def process_results(results):
    """Process the API results to determine if the shrimp is good or bad."""
    for detection in results[0]:  # Iterate over detections
        class_name = detection.get("class_name", "")
        if class_name in BAD_SHRIMP_CLASSES:
            return "BAD", class_name
        elif class_name in GOOD_SHRIMP_CLASSES:
            return "GOOD", class_name
    return "UNKNOWN", "Unknown Shrimp"


def display_status(status, class_name):
    """Display the status and class name on the LCD."""
    lcd.clear()
    if status == "GOOD":
        lcd.write_string("Status: GOOD")
        lcd.crlf()
        lcd.write_string(f"Class: {class_name}")
    elif status == "BAD":
        lcd.write_string("Status: BAD")
        lcd.crlf()
        lcd.write_string(f"Class: {class_name}")
    else:
        lcd.write_string("Status: UNKNOWN")
        lcd.crlf()
        lcd.write_string("Check Image!")


if __name__ == "__main__":
    try:
        while True:
            # Step 1: Capture Image
            image_buffer = capture_image()

            # Step 2: Send the Image to the API
            json_results = send_image_to_server(image_buffer)

            if not json_results:
                print("Failed to get a valid response from the server.")
                lcd.clear()
                lcd.write_string("Server Error!")
                sleep(2)
                continue

            # Step 3: Process Results
            status, class_name = process_results(json_results)

            # Step 4: Display Results on LCD
            display_status(status, class_name)

            # Step 5: Beep the Buzzer
            if status == "GOOD":
                beep_buzzer(times=1)  # Single beep for good shrimp
            elif status == "BAD":
                beep_buzzer(times=3)  # Three beeps for bad shrimp

            # Wait 5 seconds before the next cycle
            sleep(5)

    except KeyboardInterrupt:
        print("Exiting program.")
    finally:
        # Cleanup GPIO and LCD
        lcd.clear()
        GPIO.cleanup()
