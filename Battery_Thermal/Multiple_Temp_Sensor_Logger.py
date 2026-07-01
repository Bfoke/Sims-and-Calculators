import serial
import csv
from datetime import datetime

PORT = '/dev/cu.usbmodem11301'
BAUD_RATE = 9600

filename = f"temperature_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

ser = serial.Serial(PORT, BAUD_RATE, timeout=1)

try:
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        print(f"Saving data to {filename}...")

        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()

            if not line:
                continue

            print(line)

            parts = line.split(',')

            # Expect:
            # time_ms,temp1_c,temp2_c
            if len(parts) == 3:
                writer.writerow(parts)
                file.flush()

except KeyboardInterrupt:
    print("\nStopping logger...")

finally:
    ser.close()
    print("Serial port closed.")