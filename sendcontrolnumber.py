
from pythonosc import udp_client, dispatcher, osc_server
import sys
import argparse

if __name__ == '__main__':
	print(sys.argv[1])

	outparser = argparse.ArgumentParser()
	outparser.add_argument('channel', metavar='N', type=int, nargs='+',
                    help='the integer channel to send to')
	outparser.add_argument("--ip", default='192.168.1.100', help='The ip of the OSC server')
	outparser.add_argument("--port", type=int, default=8000, help="The port the OSC server is listening on")
	args = outparser.parse_args()

	client = udp_client.SimpleUDPClient(args.ip, args.port)
	print(args.channel[0])

	client.send_message(f"/ESN/{sys.argv[1]}", args.channel[0])


