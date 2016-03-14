from fuel.datasets.youtube_audio import YouTubeAudio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from random import randint




data        = YouTubeAudio('XqaJ2Ol5cC4')
stream      = data.get_example_stream()
it          = stream.get_epoch_iterator()
track       = next(it)
track       = track[0].reshape(-1)
sample_rate = scipy.io.wavfile.read("/home/alex/fuel_data/XqaJ2Ol5cC4.wav")[0]
total_len   = track.shape[0]

#input shape is expected to be 
#(batch_size, sequence_length, num_inputs)
secs        = 1
num_inputs  = sample_rate*secs
seq_len     = 60
example     = seq_len*sample_rate

#Setting appropriate length to account for the size of an example
train_len   = total_len*8/10/example*example
valid_len   = (total_len-train_len)/2/example*example
test_len    = (total_len-train_len)/2/example*example
unused      = total_len - train_len - valid_len - test_len
print train_len, valid_len, test_len, unused


train_track  = track[:train_len                          ].astype(np.float32)
valid_track  = track[train_len:train_len+valid_len       ].astype(np.float32)
test_track   = track[train_len+valid_len:total_len-unused].astype(np.float32)


#PreProcessing
meanstd             = train_track.std()
train_track_mean    = train_track.mean()
train_track_std     = train_track.std()
train_track        -= train_track_mean
train_track        /= train_track_std + 0.1 * meanstd
valid_track        -= train_track_mean
valid_track        /= train_track_std + 0.1 * meanstd
test_track         -= train_track_mean
test_track         /= train_track_std + 0.1 * meanstd


#Chopping track in "secs" seconds intervals
# and making input matrix for lasagne lstm
# by reshaping with proper dimensions
# x = np.arange(3*5*8).reshape(3, 20, 8)
train_track  = train_track.astype(np.float32).reshape(train_len/example, seq_len, num_inputs)
valid_track  = valid_track.astype(np.float32).reshape(valid_len/example, seq_len, num_inputs)
test_track   =  test_track.astype(np.float32).reshape(test_len/example,  seq_len, num_inputs)


"""
#Plotting a piece of the sample
plt.plot(track[160000:240000])
scipy.io.wavfile.write("newsample.wav", 16000, track[160000:240000])
print train_track.max(), train_track.min(), train_track.mean()
"""

def pca(data, dimstokeep):
    """ principal components analysis of data (columnwise in array data), retaining as many components as required to retain var_fraction of the variance 
    """
    from numpy.linalg import eigh
    u, v = eigh(np.cov(data, rowvar=0, bias=1))
    v = v[:, np.argsort(u)[::-1]]
    backward_mapping = v[:,:dimstokeep].T
    forward_mapping = v[:,:dimstokeep]
    return backward_mapping.astype("float32"), forward_mapping.astype("float32"), np.dot(v[:,:dimstokeep].astype("float32"), backward_mapping), np.dot(forward_mapping, v[:,:dimstokeep].T.astype("float32"))

#pca_backward, pca_forward, zca_backward, zca_forward = pca(train_track, dimstokeep=2000)
#pca_backward, pca_forward, zca_backward, zca_forward = pca(train_track, var_fraction=0.9)


np.save("train_track", train_track)
np.save("valid_track", valid_track)
np.save("test_track",  test_track)


track_len = track[0].shape[0]
track_time = track_len / sample_rate
#Length of interval
inter_len = sample_rate * secs
num_interval = track_time / secs
#X = []
for i in range(num_interval):
    X[i] = track[0][i*inter_len:(i+1)*inter_len].T

"""
#Random subsample stacked
newsample2 = X[0].astype(np.int16)
for i in range(15):
    newsample2 = np.hstack((newsample2,X[randint(0,num_interval)])).astype(np.int16)  
scipy.io.wavfile.write("newsample2.wav", 16000, newsample2)
"""
