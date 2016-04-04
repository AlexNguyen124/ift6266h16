import h5py
import matplotlib.pyplot as plt
import numpy as np

f = h5py.File('XqaJ2Ol5cC4.hdf5', 'r')
arr = f['features'][0,:60000]
plt.plot(arr)


def gen_data(datastream, batch, batch_size, seq_len, num_inputs):
    Input=[]
    for i in range(batch_size):
        #one sequence of x and y
        x_beg = batch*num_inputs + i*num_inputs
        x_end = x_beg + seq_len * num_inputs
        xi = datastream[x_beg:x_end].reshape(seq_len, num_inputs)
        Input.append(xi)
        #arr = f['features'][0,:60000]
    return np.array(Input)

batch_size,seq_len,num_inputs = 12,31,14
example                       = seq_len*num_inputs
datastream=np.arange(100*seq_len*num_inputs)

#Setting appropriate length to account for the size of an example
total_len   = datastream.shape[0]
train_len   = total_len*8/10/example*example
valid_len   = (total_len-train_len)/2/example*example
test_len    = (total_len-train_len)/2/example*example
unused      = total_len - train_len - valid_len - test_len
print train_len, valid_len, test_len, unused

num_batch = train_len/num_inputs-seq_len-batch_size+1
valid_size = valid_len/num_inputs-seq_len-batch_size+1

X_train_batch   = gen_data(datastream, num_batch, batch_size, seq_len, num_inputs)
X_valid         = gen_data(datastream, num_batch+batch_size+seq_len-1, valid_size+batch_size, seq_len, num_inputs)
print "X_train_batch"
print  X_train_batch 
print "X_valid[0]"
print  X_valid[0]
print "X_valid[-1]"
print  X_valid[-1]
print datastream[train_len+valid_len]
