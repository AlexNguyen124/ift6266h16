import theano
import theano.tensor as T
import numpy as np
from lasagne.layers import *
from lasagne.init import *
from lasagne.nonlinearities import *
from lasagne.objectives import *
from lasagne.updates import *
import scipy.io.wavfile
from time import time
import matplotlib.pyplot as plt


print "...loading data"
X_train         = np.load("train_track.npy")
X_valid         = np.load("valid_track.npy")
X_test          = np.load("test_track.npy")

#Hyper-Parameters
examples        = X_train.shape[0]
seq_len         = X_train.shape[1]
num_inputs      = X_train.shape[2]
num_units       = 800
batch_size      = 16
num_epochs      = 100
learn_rate      = 0.01
length          = 60
forget_gate     = Gate(b=Constant(1.0))    # initialize the constant of the forget gate


trial = str(examples)+str(seq_len)+str(num_inputs)+str(num_units)


print "...building model"
X           = T.tensor3('X')

l_input     = InputLayer((None, seq_len, num_inputs))
l_lstm1     = LSTMLayer(l_input, num_units, nonlinearity=tanh, forgetgate=forget_gate)
l_lstm2     = LSTMLayer(l_lstm1, num_units, nonlinearity=tanh, forgetgate=forget_gate)
l_lstm3     = LSTMLayer(l_lstm2, num_units, nonlinearity=tanh, forgetgate=forget_gate)
l_shp       = ReshapeLayer(l_lstm3, (-1, num_units))
l_dense     = DenseLayer(l_shp, num_units=num_inputs, nonlinearity=tanh)
l_out       = ReshapeLayer(l_dense, (-1, seq_len, num_inputs))
net_out     = get_output(l_out, X)
    
    
Y           = net_out[:, 0: seq_len - 1, :]
target      = X[:, 1:, :]
loss        = T.mean((Y - target)**2)
loss_fn     = theano.function([X], loss)
out_fn      = theano.function([X], net_out)

params      = get_all_params(l_out, trainable=True)
grads       = T.grad(loss, params)
updates     = adam(grads, params, learn_rate)
train_fn    = theano.function([X], loss, updates=updates)
val_fn      = theano.function([X], loss)

print "...training model"
TRAINLOSS   = []
VALIDLOSS   = []
epoch       = 0
t0          = time()
best_val_loss = float('inf')
best_model  = None
num_batches = int(len(X_train) / batch_size) - 1

while epoch < num_epochs:
    epoch += 1
    train_losses = []
    for i in range(num_batches):
        X_train_batch = X_train[i*batch_size: (i + 1)*batch_size]
        train_losses = train_fn(X_train_batch)
    
    train_loss = train_losses.mean()
    this_val_loss = val_fn(X_valid)
    if this_val_loss < best_val_loss:
        best_val_loss = this_val_loss
        best_model = get_all_param_values(l_out)
    t = time() - t0
    TRAINLOSS.append(train_loss)
    VALIDLOSS.append(this_val_loss)
    print "Epoch " + str(epoch) \
        + "  Train loss " + str(train_loss) \
        + "  Valid loss " + str(this_val_loss) \
        + "  Total time: " + str(t)
        
np.save("best_model_"+trial, best_model)

t = np.arange(0, len(TRAINLOSS))
# red dashes, blue squares and green triangles
plt.plot(t,np.array(TRAINLOSS),t,np.array(VALIDLOSS),'r--', label="LOSS")
plt.savefig('Cost_Epoch'+trial+'.png')

def generate(X_test, model, l_out, out_fn, length):
    set_all_param_values(l_out, model)
    seed = X_test[0:1]
    generated_seq = []
    prev_input = seed
    for x in range(0, length):
        next_input = out_fn(prev_input)
        generated_seq.append(next_input.flatten()[0:num_inputs])
        #do like melvin. : prev_input = prev_input[:-8000] next_input[-8000:]
        prev_input = next_input
    return generated_seq

generated_seq = generate(X_test, best_model, l_out, out_fn, length)
#np.max(generated_seq), np.min(generated_seq), np.mean(generated_seq)
generated_seq = np.array(generated_seq) * 32768.
generated_seq = generated_seq.reshape(length*num_inputs).astype(np.int16)
#generated_seq.shape
plt.plot(generated_seq)
#plt.plot(X_valid[0].reshape(seq_len*num_inputs))
scipy.io.wavfile.write('generated_seq'+trial+'.wav', 16000, generated_seq)
