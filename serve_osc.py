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
	print("[{0}] ~ {1}".format(args[0], volume))
	print(np.array(volume))
	
	update = esn.step(np.array(volume))
	print(update)
	count = 0
	client.send_message("/ESN", update)
		


if __name__ == '__main__':

	esn = SimpleESN(n_readout=7, n_components=7, n_inputs=1, input_gain=1, input_sparcity=0.6, damping=0.1, weight_scaling=1.15, sparcity=0.5)

	print('ESN online')
	print('Spectral radius: {}'.format(np.max(np.abs(la.eig(esn.weights_)[0]))))


	y, sr = librosa.load('VPRYNIMVMT2.wav')
	y = y/max(y)
	print('soundfile loaded')

	# for outgoing messages
	outparser = argparse.ArgumentParser()
	outparser.add_argument("--ip", default='127.0.0.1', help='The ip of the OSC server')
	outparser.add_argument("--port", type=int, default=9000, help="The port the OSC server is listening on")
	args = outparser.parse_args()

	client = udp_client.SimpleUDPClient(args.ip, args.port)


	# for incoming messages
	inparser = argparse.ArgumentParser()
	inparser.add_argument("--ip", default='127.0.0.1', help='the ip to listen on')
	inparser.add_argument("--port", type=int, default=12000, help='The port to listen on')
	inargs = inparser.parse_args()

	dispatcher = dispatcher.Dispatcher()
	dispatcher.map("/outputs", run_network_step, "Echo State Network")

	# run the ESN
	for i in range(200):
		print(i)
		run_network_step('blah','ESN',y[100000+i])
		time.sleep(0.5)


	server = osc_server.ThreadingOSCUDPServer(
		(inargs.ip, inargs.port), dispatcher)
	print("serving on {}".format(server.server_address))
	server.serve_forever()

	

