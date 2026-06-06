import serial
import csv
from datetime import datetime

PORT = '/dev/cu.usbmodem1301'
BAUD_RATE = 9600

filename = f"temperature_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

ser = serial.Serial(PORT, BAUD_RATE, timeout=1)

try:
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        print(f"Saving data to {filename}...")

        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()

            if line:
                print(line)
                writer.writerow(line.split(','))
                file.flush()

except KeyboardInterrupt:
    print("\nStopping logger...")

finally:
    ser.close()
    print("Serial port closed.")