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
	#print("[{0}] ~ {1}".format(args[0], volume))
	#print(np.array(volume))
	update = esn.step(np.array(volume))
	for index, i in enumerate(update):
		client.send_message(f"/ESN/{index}", ((i+1)/2))

	#client.send_message(f'/ESN/6', update[6])

		


if __name__ == '__main__':

	esn = SimpleESN(n_readout=20, 
                n_components=20, 
                n_inputs=1, 
                input_gain=3, 
                input_sparcity=1, 
                damping=0.9, 
                weight_scaling=1.5, 
                sparcity=1.0,
                random_state=31337)

	print('ESN online')
	print('Spectral radius: {}'.format(np.max(np.abs(la.eig(esn.weights_)[0]))))


	y, sr = librosa.load('VPRYNIMVMT2.wav')
	y = y/max(y)
	print('soundfile loaded')

	# for outgoing messages
	outparser = argparse.ArgumentParser()
	outparser.add_argument("--ip", default='172.16.42.2', help='The ip of the OSC server')
	outparser.add_argument("--port", type=int, default=8000, help="The port the OSC server is listening on")
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
	for i in range(20000):
		print(i)
		run_network_step('blah','ESN',y[400000+i])
		time.sleep(0.3)

	for i in range(100):
		run_network_step('blah','ESN',0)
		time.sleep(0.3)


	server = osc_server.ThreadingOSCUDPServer(
		(inargs.ip, inargs.port), dispatcher)
	print("serving on {}".format(server.server_address))
	server.serve_forever()

	

