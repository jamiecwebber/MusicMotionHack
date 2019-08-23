from simple_esn import SimpleESN
import matplotlib.pyplot as plt
import numpy as np
import librosa
import soundfile as sf
import scipy.linalg as la
from IPython import display
import time

from pythonosc import udp_client, dispatcher, osc_server
import argparse



def run_network_step(unused_addr, args, *volume):
	# print("[{0}] ~ {1}".format(args[0], volume))
	#print((np.array(volume)*2)-1)
	
		update = esn.step((np.array(volume)*2)-1)
		print(update)
		count = 0
	#client.send_message("/wek/nothing", update)
		


if __name__ == '__main__':

	esn = SimpleESN(n_readout=7, n_components=7, n_inputs=7, input_gain=1, input_sparcity=0.6, damping=0.1, weight_scaling=1.25, sparcity=0.5)

	print('ESN online')
	print('Spectral radius: {}'.format(np.max(np.abs(la.eig(esn.weights_)[0]))))


	# for outgoing messages
	outparser = argparse.ArgumentParser()
	outparser.add_argument("--ip", default='10.0.1.6', help='The ip of the OSC server')
	outparser.add_argument("--port", type=int, default=12000, help="The port the OSC server is listening on")
	args = outparser.parse_args()

	# for incoming messages
	inparser = argparse.ArgumentParser()
	inparser.add_argument("--ip", default='10.0.1.4', help='the ip to listen on')
	inparser.add_argument("--port", type=int, default=12000, help='The port to liste on')
	inargs = inparser.parse_args()

	client = udp_client.SimpleUDPClient(args.ip, args.port)


	dispatcher = dispatcher.Dispatcher()
	dispatcher.map("/wek/outputs", run_network_step, "Echo State Network")

	server = osc_server.ThreadingOSCUDPServer(
		(inargs.ip, inargs.port), dispatcher)
	print("serving on {}".format(server.server_address))
	server.serve_forever()