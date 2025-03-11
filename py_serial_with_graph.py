import serial
import time
import matplotlib.pyplot as plt

# Define the serial port and baud rate
arduino_port = "COM6"  # Change this to your Arduino's port
baud_rate = 9600

# Establish a serial connection
try:
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    print(f"Connected to {arduino_port}")
except serial.SerialException as e:
    print(f"Error connecting to {arduino_port}: {e}")
    exit()

# Allow Arduino time to reset
time.sleep(2)

# Initialize live plotting
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_xlabel("Valence (Unpleasant to Pleasant)")
ax.set_ylabel("Arousal (Low to High)")
ax.set_title("Live Arousal-Valence Emotion Tracking")

# Define quadrant colors
colors = {
    "angry": "#E6CCE6",  # Light purple
    "happy": "#FFF2CC",  # Light yellow
    "sad": "#CCE6FF",  # Light blue
    "calm": "#E6F2CC",  # Light green
    "neutral": "white"
}

# Create background color quadrants
ax.fill_between([-1, 0], 0, 1, color=colors["angry"])  # Top left (Angry)
ax.fill_between([0, 1], 0, 1, color=colors["happy"])  # Top right (Happy)
ax.fill_between([-1, 0], -1, 0, color=colors["sad"])  # Bottom left (Sad)
ax.fill_between([0, 1], -1, 0, color=colors["calm"])  # Bottom right (Calm)

# Plot initialization
point, = ax.plot(0, 0, 'ro', markersize=10)  # Red dot at center

# Function to send a byte to Arduino
def send_byte(data_byte):
    """Sends a single byte to the Arduino and updates the graph."""
    try:
        ser.write(bytes([data_byte]))
        print(f"Sent: {data_byte}")

        # Read response from Arduino (optional)
        time.sleep(0.1)
        if ser.in_waiting > 0:
            response = ser.readline().decode().strip()
            print(f"Arduino response: {response}")

        # Update the graph based on input
        update_graph(data_byte)

    except Exception as e:
        print(f"Error sending data: {e}")

def update_graph(value):
    """Maps input (0-4) to predefined Arousal-Valence coordinates and updates the graph."""
    mapping = {
        0: (0, 0),    # Neutral (White) - Center
        1: (-0.5, 0.5), # Angry (Red) - Top-left
        2: (0.5, 0.5),  # Happy (Green) - Top-right
        3: (-0.5, -0.5),# Sad (Blue) - Bottom-left
        4: (0.5, -0.5)  # Calm (Yellow) - Bottom-right
    }

    valence, arousal = mapping.get(value, (0, 0))  # Default to neutral

    point.set_xdata(valence)
    point.set_ydata(arousal)

    plt.pause(0.1)  # Ensure the plot updates in Spyder

# Main loop for user input
plt.show(block=False)  # Non-blocking display
try:
    while True:
        user_input = input("Enter a byte (0-4) or 'q' to quit: ").strip()
        if user_input.lower() == 'q':
            break

        if user_input.isdigit() and 0 <= int(user_input) <= 4:
            send_byte(int(user_input))
        else:
            print("Please enter a valid byte (0-4).")

except KeyboardInterrupt:
    print("\nExiting program.")

finally:
    ser.close()
    print("Serial connection closed.")
