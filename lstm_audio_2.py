from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne

import matplotlib.pyplot as plt

#Lasagne Seed for Reproducibility
lasagne.random.set_rng(np.random.RandomState(1))


SEQ_LENGTH      = 60    # Sequence Length
N_HIDDEN        = 512   # Number of units in the two hidden (LSTM) layers
LEARNING_RATE   = .01   # Optimization learning rate
GRAD_CLIP       = 100   # All gradients above this will be clipped
PRINT_FREQ      = 1000  # How often should we check the output?
NUM_EPOCHS      = 50    # Number of epochs to train the net
BATCH_SIZE      = 64    # Batch Size
num_inputs      = 16000 #Number of inputs per sequence

train_track = np.load("train_track.npy")
valid_track = np.load("valid_track.npy")
test_track  = np.load("test_track.npy")

def main(num_epochs=NUM_EPOCHS):
    print("Building network ...")
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, SEQ_LENGTH, num_inputs)
    
    #The network model
    
    l_in            = lasagne.layers.InputLayer(shape=(BATCH_SIZE, SEQ_LENGTH, num_inputs))
    l_forward_1     = lasagne.layers.LSTMLayer(l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,nonlinearity=lasagne.nonlinearities.tanh)
    l_forward_2     = lasagne.layers.LSTMLayer(l_forward_1, N_HIDDEN, grad_clipping=GRAD_CLIP,nonlinearity=lasagne.nonlinearities.tanh)
    l_shp           = lasagne.layers.ReshapeLayer(l_forward_2, (-1, N_HIDDEN))
    l_dense         = lasagne.layers.DenseLayer(l_shp, num_units=num_inputs, lasagne.nonlinearity=linear)
    l_out           = lasagne.layers.ReshapeLayer(l_dense, (-1, SEQ_LENGTH, num_inputs))
    
    # create output out of input in order to save memory?
    network_output  = lasagne.layers.get_output(l_out)
    cost            = lasagne.objectives.squared_error(network_output,target_values).mean()
    all_params      = lasagne.layers.get_all_params(l_out,trainable=True)
    updates         = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)


    input_values    = T.ivector('target_output')
    target_values   = T.ivector('target_output')

    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train           = theano.function([l_in.input_var, target_values], cost, updates=updates, allow_input_downcast=True)
    compute_cost    = theano.function([l_in.input_var, target_values], cost, allow_input_downcast=True)
