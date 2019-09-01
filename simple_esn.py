"""Simple Echo State Network
"""

# Copyright (C) 2015 Sylvain Chevallier <sylvain.chevallier@uvsq.fr>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# TODO: add n_readout = -1 for n_readout = n_components

from __future__ import print_function
import sys
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_random_state, check_array
from numpy import zeros, ones, concatenate, array, tanh, vstack, arange
import numpy as np
import scipy.linalg as la

class SimpleESN(BaseEstimator, TransformerMixin):
    """Simple Echo State Network (ESN)

    Neuron reservoir of sigmoid units, with recurrent connection and random
    weights. Forget factor (or damping) ensure echoes in the network. No
    learning takes place in the reservoir, readout is left at the user's
    convience. The input processed by these ESN should be normalized in [-1, 1]

    Parameters
    ----------
    n_readout : int
        Number of readout neurons, chosen randomly in the reservoir. Determines
        the dimension of the ESN output.
    
    n_components : int, optional
        Number of neurons in the reservoir, 100 by default.

    damping : float, optional
        Damping (forget) factor for echoes, strong impact on the dynamic of the
        reservoir. Possible values between 0 and 1, default is 0.5

    weight_scaling : float, optional
        Spectral radius of the reservoir, i.e. maximum eigenvalue of the weight
        matrix, also strong impact on the dynamical properties of the reservoir.
        Classical regimes involve values around 1, default is 0.9

    discard_steps : int, optional
        Discard first steps of the timeserie, to allow initialisation of the
        network dynamics.

    random_state : integer or numpy.RandomState, optional
        Random number generator instance. If integer, fixes the seed.
        
    Attributes
    ----------
    input_weights_ : array_like, shape (n_features,)
        Weight of the input units

    weights_ : array_Like, shape (n_components, n_components)
        Weight matrix for the reservoir

    components_ : array_like, shape (n_samples, 1+n_features+n_components)
        Activation of the n_components reservoir neurons, including the
        n_features input neurons and the bias neuron, which has a constant
        activation.

    readout_idx_ : array_like, shape (n_readout,)
        Index of the randomly selected readout neurons

    Example
    -------

    >>> from simple_esn import SimpleESN
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> X = np.random.randn(n_samples, n_features)
    >>> esn =SimpleESN(n_readout = 2)
    >>> echoes = esn.fit_transform(X)
    """


    def __init__(self, n_readout, n_inputs=1, input_sparcity=0.1, n_components=100, damping=0.5, input_gain=1.0,
                 weight_scaling=0.9, discard_steps=0, random_state=None, sparcity=0.05, relu=False):
        self.n_readout = n_readout
        self.n_components = n_components
        self.damping = damping
        self.weight_scaling = weight_scaling
        self.discard_steps = discard_steps
        self.random_state = check_random_state(random_state)
        self.sparcity = sparcity;
        self.input_sparcity = input_sparcity;
        self.input_gain = input_gain;
        self.input_weights_ = self.random_state.rand(self.n_components,
                                                         n_inputs)*2-1
        self.weights_ = self.random_state.rand(self.n_components, self.n_components)*2-1
        self.relu = relu
        

        for index, i in enumerate(self.input_weights_):
            for indexj, j in enumerate(i):
                if self.random_state.rand() > self.input_sparcity:
                    self.input_weights_[index][indexj] = 0

        for index, i in enumerate(self.weights_):
            for indexj, j in enumerate(i):
                if self.random_state.rand() > self.sparcity:
                    self.weights_[index][indexj] = 0
        self.state_ = zeros(self.n_components)

        spectral_radius = np.max(np.abs(la.eig(self.weights_)[0]))
        self.weights_ *=  self.weight_scaling / spectral_radius
        
    def step(self, X):
        curr_ = self.state_
        curr_ = (1-self.damping)*curr_ + self.damping*tanh(
            self.input_weights_.dot(X)*self.input_gain + self.weights_.dot(curr_))
        self.state_ = curr_
        if self.relu:
            self.state_ = np.array([np.max([x, 0]) for x in curr_])
        return self.state_


