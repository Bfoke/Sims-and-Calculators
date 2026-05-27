import serial
import csv
from datetime import datetime

# CHANGE THIS to your Arduino port
# Windows example: 'COM3'
# Mac example: '/dev/tty.usbmodem14101'
# Linux example: '/dev/ttyACM0'
PORT = 'COM3'

BAUD_RATE = 9600

filename = f"temperature_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

ser = serial.Serial(PORT, BAUD_RATE)

with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)

    print(f"Saving data to {filename}...")
    
    while True:
        line = ser.readline().decode('utf-8').strip()

        if line:
            print(line)

            # Split CSV data
            data = line.split(',')

            # Write to file
            writer.writerow(data)

            # Save immediately
            file.flush()