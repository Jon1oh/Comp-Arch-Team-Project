import serial, time, threading
from . import shared_state

USB_PORT = '/dev/cu.usbmodem1401'
BAUD_RATE = 9600

try:
    tiny_serial = serial.Serial(USB_PORT, BAUD_RATE, timeout=1, write_timeout=2)
    time.sleep(2)
    print(f"Successfully connected to TinyScreen on {USB_PORT}")
except Exception as e:
    print(f"Warning: TinyScreen not connected to {USB_PORT}. Reason: {e}")
    tiny_serial = None

def listen_for_tinyscreen_reset():
    """Continuously listens for hardware reset commands from TinyScreen+."""
    global tiny_serial
    while True:
        try:
            if tiny_serial and tiny_serial.in_waiting > 0:
                msg = tiny_serial.readline().decode('utf-8', errors='ignore').strip()
                if msg == "RESET":
                    print("[Hardware] Reset triggered from TinyScreen+! Starting new session...")
                    shared_state.latest_analysis = {
                        "status": "waiting",
                        "behavior": "",
                        "description": ""
                    }
        except Exception as e:
            print(f"Serial read error: {e}")
        time.sleep(0.1)

if tiny_serial:
    threading.Thread(target=listen_for_tinyscreen_reset, daemon=True).start()

def reset_all():
    """Resets both shared session state and the TinyScreen+ display."""
    global tiny_serial
    shared_state.latest_analysis = {"status": "waiting", "behavior": "", "description": ""}
    if tiny_serial:
        try:
            tiny_serial.reset_input_buffer()
            tiny_serial.write(b"RESET\n")
            tiny_serial.flush()
            print("--> Sent RESET command to TinyScreen+")
        except Exception as e:
            print(f"Serial reset write error: {e}")

def broadcast_results(status, ai_output=None):
    global tiny_serial
    if status == "analysing":
        shared_state.latest_analysis = {"status": "analysing", "behavior": "", "description": ""}
        if tiny_serial:
            try:
                tiny_serial.write(b"ANALYSING\n")
                tiny_serial.flush()
            except Exception:
                pass
        return

    if status != "complete" or ai_output is None:
        shared_state.latest_analysis = {"status": status, "behavior": "", "description": ""}
        return

    behavior = "ALERT"

    # Strip any stray '|' characters so Arduino field splitting stays intact
    explanation = str(ai_output.get('explanation', 'Cat is active.')).replace("|", " ")

    obs_raw = ai_output.get('supporting_observations', [])
    observations = ", ".join(obs_raw) if isinstance(obs_raw, list) else str(obs_raw)
    observations = observations.replace("|", " ")

    confidence = ai_output.get('confidence', 'Med')
    quality = ai_output.get('data_quality', 'Ltd')
    reliability = f"Confidence:{confidence}|Data:{quality}"

    shared_state.latest_analysis = {
        "status": "complete",
        "behavior": behavior,
        "description": explanation
    }

    if tiny_serial:
        try:
            payload = f"{behavior}|{explanation}|{observations}|{reliability}\n".encode()
            print(f"--> Firing compact payload to TinyScreen: {payload}")
            tiny_serial.write(payload)
            tiny_serial.flush()
            print("--> Serial payload successfully flushed!")
        except Exception as e:
            print(f"CRITICAL Serial transmission failed: {e}")