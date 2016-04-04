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
import h5py
import matplotlib.pyplot as plt


#Hyper-Parameters
sample_rate     = 16000                     #sample rate of this track is 16khz
seq_len         = 20                        #number of steps in a sequence
num_inputs      = 8000                      #number of quantized bits per steps
num_units       = 500                       #number of hidden units
batch_size      = 1 + 8                     # +1 because target leads input by 1
num_epochs      = 50
learn_rate      = 0.01
length          = 60                        #length of the generated sequence
forget_gate     = Gate(b=Constant(1.0))     # initialize the constant of the forget gate
amp             = 10000.                    #this is the approximate amplitude of the train set
example         = seq_len*num_inputs

#Everything that will be saved (cost, params, plot,...) will bear this suffix
trial = str(seq_len)+str(num_inputs)+str(num_units)+"4L-semi_red"

print "seq_len:     ", seq_len
print "num_inputs:  ", num_inputs
print "hidden units:", num_units
print "...building model"
# Recurrent layers expect input of shape
# (batch_size, seq_len, num_inputs)
l_input     = InputLayer((None, seq_len, num_inputs))
l_lstm1     = LSTMLayer(l_input, num_units, nonlinearity=tanh, forgetgate=forget_gate)
l_lstm2     = LSTMLayer(l_lstm1, num_units, nonlinearity=tanh, forgetgate=forget_gate)
l_lstm3     = LSTMLayer(l_lstm2, num_units, nonlinearity=tanh, forgetgate=forget_gate)
l_lstm4     = LSTMLayer(l_lstm3, num_units, nonlinearity=tanh, forgetgate=forget_gate)
# l_lstm outputs tensor of dimension (batch_size, seq_len, num_inputs)
# Since we are only interested in the final prediction,
# we isolate that quantity and feed it to the next layer. 
# The output of the sliced layer will then be of size (batch_size, num_inputs)
l_slice     = SliceLayer(l_lstm4, -1, 1)
#l_shp       = ReshapeLayer(l_lstm4, (-1, num_units))
l_out       = DenseLayer(l_slice, num_units=num_inputs, nonlinearity=tanh)
#l_out       = ReshapeLayer(l_dense, (-1, seq_len, num_inputs))

X           = T.tensor3('X')            #input data
Y           = X[:,-1,:]                 #target
Y_hat       = get_output(l_out, X)      #network output, fit
    
loss        = T.mean((Y_hat - Y)**2)
loss_fn     = theano.function([X], loss)
out_fn      = theano.function([X], Y_hat)

params      = get_all_params(l_out, trainable=True)
grads       = T.grad(loss, params)
updates     = adam(grads, params, learn_rate)
train_fn    = theano.function([X], loss, updates=updates)
val_fn      = theano.function([X], loss)

print "...preparing input data"
datastream = h5py.File('XqaJ2Ol5cC4.hdf5', 'r')
"""
#to extract array from HDF5
arr = datastream['features'][0,:60000]
plt.plot(arr)
"""
#Setting appropriate length to account for the size of an example
total_len   = datastream['features'][0].shape[0]
train_len   = total_len*8/10/example*example
valid_len   = (total_len-train_len)/2/example*example
test_len    = (total_len-train_len)/2/example*example
unused      = total_len - train_len - valid_len - test_len
print train_len, valid_len, test_len, unused
#num_batches = int(len(X_train) / batch_size) - 1 #old 
num_batches = train_len/num_inputs-seq_len-batch_size+1 # new


#generating training batch on the fly during training
#the data generated is semi redondant:
#batch_size, seq_len, num_inputs = 2,3,5
#n=0:   [[ 0,  1,  2,  3,  4],
#        [ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14]],
#n=1:   [[ 5,  6,  7,  8,  9],
#        [10, 11, 12, 13, 14],
#        [15, 16, 17, 18, 19]]
def gen_data(datastream, batch, batch_size, seq_len, num_inputs):
    Input=[]
    for i in range(batch_size):
        #one sequence of x and y
        x_beg = batch*num_inputs + i*num_inputs
        x_end = x_beg + seq_len * num_inputs
        xi = datastream['features'][0,x_beg:x_end].reshape(seq_len, num_inputs)
        Input.append(xi)
    return np.array(Input).astype(np.float32)/amp

#X_train is created on the fly
valid_size = valid_len/num_inputs-seq_len-batch_size+1 #total validation set size when semi-redondant
X_valid = gen_data(datastream, num_batches+batch_size+seq_len-1, valid_size+batch_size, seq_len, num_inputs)

print "...training model"
TRAINLOSS   = []
VALIDLOSS   = []
epoch       = 0
t0          = time()
best_val_loss = float('inf')
best_model  = None

while epoch < num_epochs:
    epoch += 1
    train_losses = []
    # NEW
    for batch in range(num_batches):
        X_train_batch = gen_data(datastream, batch, batch_size, seq_len, num_inputs)        
        train_losses = train_fn(X_train_batch)
        print batch,
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
plt.plot(t,np.array(TRAINLOSS),t,np.array(VALIDLOSS),'r--', label="LOSS")
plt.savefig('Cost_Epoch'+trial+'.png')


def generate(X_test, model, l_out, out_fn, length):
    set_all_param_values(l_out, model)
    seed = X_test[0:1]
    generated_seq = []
    prev_input = seed
    for x in range(0, length):
        next_input = out_fn(prev_input)
        new_prev_input = np.concatenate((prev_input[:,1:seq_len], next_input[0,0].reshape(1,1,num_inputs)),axis=1)        
        prev_input = new_prev_input
        generated_seq.append(next_input[0,0])
    return generated_seq
    
#scipy.io.wavfile.write('newsample.wav', 16000, (next_input*10000).reshape(seq_len*num_inputs).astype(np.int16))

generated_seq = generate(X_test, best_model, l_out, out_fn, length)
#np.max(generated_seq), np.min(generated_seq), np.mean(generated_seq)
generated_seq = np.array(generated_seq) * amp
generated_seq = generated_seq.reshape(length*num_inputs).astype(np.int16)
#generated_seq.shape
plt.plot(generated_seq)
plt.savefig('gen_seq_plot'+trial+'.png')
#plt.plot(X_valid[0].reshape(seq_len*num_inputs))
scipy.io.wavfile.write('generated_seq'+trial+'.wav', 16000, generated_seq)

"""
fig = plt.figure()

ax1 = fig.add_subplot(221)
ax1.plot(generated_seq[700:900])
ax2 = fig.add_subplot(222)
ax2.plot(generated_seq[1500:1700])
ax3 = fig.add_subplot(223)
ax3.plot(generated_seq[3100:3300])
ax4 = fig.add_subplot(224)
ax4.plot(generated_seq[27900:28100])

plt.show()
"""