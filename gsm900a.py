import serial
import time

def initialize_serial_connection(port='/dev/ttyAMA0', baudrate=9600):
    try:
        ser = serial.Serial(port, baudrate=baudrate, timeout=1)
        time.sleep(2)  # Allow some time for the GSM module to initialize
        return ser
    except serial.SerialException as e:
        print(f"Error opening serial port: {e}")
        return None

def send_sms(ser, phone_number, message):
    try:
        if ser is None:
            print("Serial connection is not initialized.")
            return

        # Set SMS mode to text
        ser.write(b'AT+CMGF=1\r\n')
        time.sleep(1)
        response = ser.read_all().decode('utf-8', errors='replace')
        print(f"Response after setting text mode: {response}")

        # Set recipient phone number
        ser.write(f'AT+CMGS="{phone_number}"\r\n'.encode('utf-8'))
        time.sleep(1)
        response = ser.read_all().decode('utf-8', errors='replace')
        print(f"Response after setting phone number: {response}")

        # Send the message
        ser.write(f'{message}\r\n'.encode('utf-8'))
        ser.write(bytes([26]))  # CTRL+Z to send the message
       # time.sleep(2)

        # Read response from GSM module
        response = ser.read_all().decode('utf-8', errors='replace')
        print(f"Response after sending message: {response}")

    except serial.SerialException as e:
        print(f"Serial Error: {e}")

    except Exception as e:
        print(f"Error: {e}")

def main():
    ser = initialize_serial_connection()
    if ser:
        phone_number = '6309038588'  # Replace with actual phone number
        message = 'Hello from Raspberry Pi 8gb!'
        send_sms(ser, phone_number, message)
        ser.close()


main()
