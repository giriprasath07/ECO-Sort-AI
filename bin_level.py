# bin_level.py
import lgpio
import time
import requests

# --- Firebase Project (must match UI app) ---
FIREBASE_BASE_URL = "https://mysmartbinproject-8bcad-default-rtdb.asia-southeast1.firebasedatabase.app"
BIN1_URL = f"{FIREBASE_BASE_URL}/bins/1/percentage.json"  # Non-Recyclable
BIN2_URL = f"{FIREBASE_BASE_URL}/bins/2/percentage.json"  # Recyclable

# --- Ultrasonic Sensor Pins ---
RECYCLE_TRIG = 5
RECYCLE_ECHO = 6
NONREC_TRIG = 27
NONREC_ECHO = 22

BIN_HEIGHT_CM = 40  # Adjust to your bin height

chip = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(chip, RECYCLE_TRIG)
lgpio.gpio_claim_input(chip, RECYCLE_ECHO)
lgpio.gpio_claim_output(chip, NONREC_TRIG)
lgpio.gpio_claim_input(chip, NONREC_ECHO)


def read_distance(trigger, echo):
    """Measure distance with ultrasonic sensor."""
    lgpio.gpio_write(chip, trigger, 0)
    time.sleep(0.000002)
    lgpio.gpio_write(chip, trigger, 1)
    time.sleep(0.00001)
    lgpio.gpio_write(chip, trigger, 0)

    start = time.time()
    stop = time.time()

    timeout = start + 0.05
    while lgpio.gpio_read(chip, echo) == 0:
        start = time.time()
        if time.time() > timeout:
            return BIN_HEIGHT_CM * 2

    timeout = start + 0.05
    while lgpio.gpio_read(chip, echo) == 1:
        stop = time.time()
        if time.time() > timeout:
            break

    elapsed = stop - start
    return (elapsed * 34300) / 2


def read_bin_levels():
    """Calculate bin fill levels in percentage."""
    recycle_distance = read_distance(RECYCLE_TRIG, RECYCLE_ECHO)
    non_recycle_distance = read_distance(NONREC_TRIG, NONREC_ECHO)

    recycle_level = max(0, min(100, 100 - (recycle_distance / BIN_HEIGHT_CM) * 100))
    non_recycle_level = max(0, min(100, 100 - (non_recycle_distance / BIN_HEIGHT_CM) * 100))
    return int(round(recycle_level)), int(round(non_recycle_level))


def send_to_firebase(recycle_level, non_recycle_level):
    """Send levels to Firebase in the structure expected by UI."""
    try:
        r1 = requests.put(BIN1_URL, json=non_recycle_level)  # bin1 → Non-Recyclable
        r2 = requests.put(BIN2_URL, json=recycle_level)      # bin2 → Recyclable
        if r1.status_code != 200 or r2.status_code != 200:
            print(f"⚠ Firebase Error: {r1.status_code}/{r2.status_code}")
    except Exception as e:
        print(f"⚠ Failed to send data: {e}")


def start_monitoring(polling_interval=5):
    """Continuously read and push bin levels to Firebase."""
    try:
        while True:
            recycle_level, non_recycle_level = read_bin_levels()
            print(f"♻ Recycle Bin: {recycle_level:.1f}% full")
            print(f"🗑 Non-Recycle Bin: {non_recycle_level:.1f}% full")
            send_to_firebase(recycle_level, non_recycle_level)
            time.sleep(polling_interval)
    except KeyboardInterrupt:
        lgpio.gpiochip_close(chip)
