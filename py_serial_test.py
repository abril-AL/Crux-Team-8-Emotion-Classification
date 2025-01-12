# Serial Signal (python to Arduino UNO)
import serial
import time

# Define the serial port and baud rate
arduino_port = "COM6"  # Change this to your Arduino's port
baud_rate = 9600

# Establish a serial connection
try:
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    print(f"Connected to {arduino_port}")
except serial.SerialException as e:
    print(f"Error connecting to {arduino_port}: {e}")
    exit() #Error connecting to port, Arduino might be in diff COM

# Allow Arduino time to reset
time.sleep(2)


# Function for sending byte
def send_byte(data_byte):
    """Sends a single byte to the Arduino."""
    try:
        # serial write
        ser.write(bytes([data_byte]))
        print(f"Sent: {data_byte}")

        # Optional: Read response from Arduino
        time.sleep(0.1)  # Give Arduino time to respond
        if ser.in_waiting > 0:
            response = ser.readline().decode().strip()
            print(f"Arduino response: {response}")
    except Exception as e:
        print(f"Error sending data: {e}")


# Testing - loop for sending signal, 1 byte at a time
try:
    while True:
        # Enter the byte to send (0 or 1 in this example)
        user_input = input("Enter a byte to send, or 'q' to quit: ").strip()
        if user_input.lower() == 'q':
            break

        # Validate input
        if user_input.isdigit() and 0 <= int(user_input) <= 255:
            send_byte(int(user_input))
        else:
            print("Please enter a valid byte (0-255).")

except KeyboardInterrupt:
    print("\nExiting program.")

finally:
    ser.close()
    print("Serial connection closed.")
