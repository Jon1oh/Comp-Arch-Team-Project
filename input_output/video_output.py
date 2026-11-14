import serial, time, shared_state # Ensure you have a shared_state.py file to hold the latest_analysis dictionary

# Replace this with your Mac's actual USB port name (find it in Arduino IDE > Tools > Port)
# Typical Mac examples: '/dev/cu.usbserial-1410' or '/dev/cu.usbmodem14101'
USB_PORT = '/dev/cu.usbserial-1410' 
BAUD_RATE = 9600

# Initialize the physical USB connection to the TinyScreen
try:
    tiny_serial = serial.Serial(USB_PORT, BAUD_RATE, timeout=1)
    # Give the Arduino hardware 2 seconds to reset and initialize after opening the port
    time.sleep(2) 
    print(f"Successfully connected to TinyScreen on {USB_PORT}")
except Exception as e:
    print(f"Warning: TinyScreen not connected to {USB_PORT}. Physical display disabled.")
    tiny_serial = None


def broadcast_results(status, ai_output=None):
    """
    Updates the shared state for the iOS app to poll, 
    and sends the paginated data to the TinyScreen via USB.
    """
    
    # Handle non-complete states (like "waiting" or "error")
    if status != "complete" or ai_output is None:
        shared_state.latest_analysis = {
            "status": status,
            "behavior": "",
            "description": ""
        }
        return

    # 1. Extract the data safely from your teammate's AI dictionary
    short_label = ai_output.get('short_label', 'ANALYSIS DONE')
    meaning = ai_output.get('body_language_meaning', 'No description available.')
    confidence = ai_output.get('confidence', 'N/A')

    # 2. Save to shared_state for the iOS App and Web Dashboard to fetch
    # We map 'short_label' to behavior and 'meaning' to description for the iOS UI
    shared_state.latest_analysis = {
        "status": "complete",
        "behavior": short_label,
        "description": meaning
    }