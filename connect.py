from dronekit import connect, VehicleMode
import time
import argparse

def connectMyCopter():
    parser = argparse.ArgumentParser(description='Commands to connect to a drone')
    parser.add_argument('--connect', help="Connection string (e.g., 'udp:127.0.0.1:14550')")
    args = parser.parse_args()

    connection_string = args.connect
    baud_rate = 57600
    if not connection_string:
        raise ValueError("Connection string not provided. Use the --connect argument.")
    
    print("\nConnecting to vehicle on: %s" % connection_string)
    vehicle = connect(connection_string, baud=baud_rate, wait_ready=True)
    return vehicle




def get_gps_location(vehicle):
    # Get GPS location from the vehicle
    gps_location = vehicle.location.global_frame
    return gps_location

vehicle = connectMyCopter()

while True:
    gps_location = get_gps_location(vehicle)
    print(f"Latitude: {gps_location.lat}, Longitude: {gps_location.lon}, Altitude: {gps_location.alt}")
    time.sleep(1)
    

    