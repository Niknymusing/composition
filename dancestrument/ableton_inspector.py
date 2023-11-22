# Use this script to inspect the OSC messages sent by Ableton Live.
# This script will send a message to Ableton Live, and then print
# the messages received from Ableton Live.
# For example: if you want to see all the parameters available for a device.
# See docs for more: https://github.com/ideoforms/AbletonOSC

from pythonosc import udp_client
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

import threading
import time

ip = "127.0.0.1"
to_ableton = 11000
from_ableton = 11001

def serve():
    print("server started")
    def print_handler(address, *args):
        print("handler running...")
        print(args)
    dispatcher = Dispatcher()
    dispatcher.map("*", print_handler)
    server = BlockingOSCUDPServer((ip, from_ableton), dispatcher)
    time.sleep(0.1)
    server.serve_forever()

def client():
    print("client started")
    client = udp_client.SimpleUDPClient(ip, to_ableton)
    client.send_message("Hi ableton!", None)
    # client_path = "/live/device/get/parameters/name"
    client_path = "/live/device/get/parameters/value"
    client.send_message(client_path, [1, 1, 5]) # frequency
    client_path = "/live/device/get/parameters/name"
    client.send_message(client_path, [1, 1]) # frequency
    time.sleep(0.1)

t1 = threading.Thread(target=serve)
t1.start()
client()