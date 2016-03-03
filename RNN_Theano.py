"""
#Classic Vanilla RNN
Adaptation of https://github.com/dennybritz/rnn-tutorial-rnnlm/blob/master/rnn_theano.py
"""

import numpy as np
import theano as theano
import theano.tensor as T


input_dim=10
hidden_dim=100

class RNNTheano:
    #initialize everything!
    def __init__(self, input, target, input_dim, hidden_dim, output_dim, lr):
        self.input  = input
        self.target = target
        self.lr     = lr
        #Weight Matrixes        
        U           = np.random.uniform(size=(hidden_dim, input_dim),  low=-np.sqrt(1./input_dim),  high=np.sqrt(1./input_dim),  dtype=theano.config.floatX)  # input-to-hidden weight matrix
        W           = np.random.uniform(size=(hidden_dim, hidden_dim), low=-np.sqrt(1./hidden_dim), high=np.sqrt(1./hidden_dim), dtype=theano.config.floatX) # Hidden to hidden recurrent connections weight matrix
        V           = np.random.uniform(size=(input_dim, hidden_dim),  low=-np.sqrt(1./hidden_dim), high=np.sqrt(1./hidden_dim), dtype=theano.config.floatX) # hidden-to-output weight matrix
        # Biases        
        b           = np.zeros((hidden_dim,), dtype=theano.config.floatX) # Hidden bias
        c           = np.zeros((output_dim,), dtype=theano.config.floatX) # Output bias
        h0          = np.zeros((hidden_dim,), dtype=theano.config.floatX) # Initial hidden State of RNN

        
        # Theano: Created shared variables
        self.U      = theano.shared(name='U', value=U)
        self.W      = theano.shared(name='W', value=W)
        self.V      = theano.shared(name='V', value=V)                                 
        self.b      = theano.shared(name='b', value=b)
        self.c      = theano.shared(name='c', value=c)
        self.h0     = theano.shared(name='h0', value=h0)


        self.params = [self.W, self.U, self.V, self.h0, self.b, self.c]
        
        # Recurrent activation function        
        def step(x_t, h_tm1):
            h_t     = T.tanh(T.dot(x_t, self.U) + T.dot(h_tm1, self.W) + self.b)
            y_t     = T.dot(h_t, V)
            return h_t, y_t
        # the hidden state `h` for the entire sequence, and the output for the

        [self.h, self.y], _ = theano.scan(step,
                        sequences       =self.input,
                        outputs_info    =[self.h0, None],
                        non_sequences   =[W, U, V])
        
        #Error function
        self.loss = T.mean((self.y - self.target) ** 2)
        # gradients on the weights using BPTT
        gW, gU, gV = T.grad(self.loss, [W, U, V])
        # training function, that computes the error and updates the weights using
        # SGD.
        fn = theano.function([h0, self.input, self.target, self.lr],
                         self.loss,
                         updates={W: W - self.lr * gW,
                                  U: U - self.lr * gU,
                                  V: V - self.lr * gV})
