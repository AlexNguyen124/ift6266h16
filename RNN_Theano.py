"""
#Classic Vanilla RNN
Adaptation of https://github.com/dennybritz/rnn-tutorial-rnnlm/blob/master/rnn_theano.py
"""

import numpy as np
import theano as theano
import theano.tensor as T
from utils import *
import operator

class RNNTheano:
    
    def __init__(self, seq_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.seq_dim = seq_dim #Dimension of a sequence of X
        self.hidden_dim = hidden_dim
        #self.bptt_truncate = bptt_truncate #To use for timeshift?
        
        # Randomly initialize the network parameters
        U = np.random.uniform(-np.sqrt(1./seq_dim), np.sqrt(1./seq_dim), (hidden_dim, seq_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (seq_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

        # Theano: Created shared variables
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))      
        
        # We store the Theano graph here
        self.theano = {}
        self.__theano_build__()
        
    def __theano_build__(self):
        U, V, W = self.U, self.V, self.W
        x = T.ivector('x')
        y = T.ivector('y')
        def forward_prop_step(x_t, s_t_prev, U, V, W):
            s_t = T.tanh(U[:,x_t] + W.dot(s_t_prev)) #How big is s_t_prev? t=1 or t=inf?
            o_t = T.nnet.softmax(V.dot(s_t))
            return [o_t[0], s_t]
        [o,s], updates = theano.scan(
            forward_prop_step,
            sequences=x,
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[U, V, W],
            truncate_gradient=self.bptt_truncate,
            strict=True)
        
        