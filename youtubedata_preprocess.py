
from fuel.datasets.youtube_audio import YouTubeAudio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from random import randint




data = YouTubeAudio('XqaJ2Ol5cC4')
stream = data.get_example_stream()
it = stream.get_epoch_iterator()
sequence = next(it)
sequence = sequence[0].reshape(-1)

total_time=3*60*60+1*60+29.0
print total_time
total_len = sequence.shape[0]
print total_len
len_of_1sec_sample = total_len/total_time
print len_of_1sec_sample


train_len = 1000000
valid_len = (total_len - train_len) / 2
test_len  = total_len - valid_len - train_len


train_sequence  = sequence[:train_len]
valid_sequence  = sequence[train_len:train_len + valid_len]
test_sequence   = sequence[valid_len + test_len:]

#PreProcessing
meanstd             = train_sequence.std()
train_sequence_mean = train_sequence.mean()
train_sequence_std  = train_sequence.std()
train_sequence     -= train_sequence_mean
train_sequence     /= train_sequence_std + 0.1 * meanstd
valid_sequence     -= train_sequence_mean
valid_sequence     /= train_sequence_std + 0.1 * meanstd
test_sequence      -= train_sequence_mean
test_sequence      /= train_sequence_std + 0.1 * meanstd

def pca(data, dimstokeep):
    """ principal components analysis of data (columnwise in array data), retaining as many components as required to retain var_fraction of the variance 
    """
    from np.linalg import eigh
    u, v = eigh(np.cov(data, rowvar=0, bias=1))
    v = v[:, np.argsort(u)[::-1]]
    backward_mapping = v[:,:dimstokeep].T
    forward_mapping = v[:,:dimstokeep]
    return backward_mapping.astype("float32"), forward_mapping.astype("float32"), np.dot(v[:,:dimstokeep].astype("float32"), backward_mapping), np.dot(forward_mapping, v[:,:dimstokeep].T.astype("float32"))


pca_backward, pca_forward, zca_backward, zca_forward = pca(train_sequence, dimstokeep=2000)
#pca_backward, pca_forward, zca_backward, zca_forward = pca(train_sequence, var_fraction=0.9)


"""
#Plotting a piece of the sample
newsample = sequence[0][160000:240000]
plt.plot(newsample)
plt.ylabel('some numbers')
plt.show()
scipy.io.wavfile.write("newsample.wav", 16000, newsample)
"""



#Chopping track in "secs" seconds intervals
secs = 5
track_len = sequence[0].shape[0]
sample_rate = scipy.io.wavfile.read("/home/alex/fuel_data/XqaJ2Ol5cC4.wav")[0]
track_time = track_len / sample_rate
#Length of interval
inter_len = sample_rate * secs
num_interval = track_time / secs
X = np.zeros((num_interval, inter_len))
#X = []
for i in range(num_interval):
    X[i] = sequence[0][i*inter_len:(i+1)*inter_len].T

"""
#Random subsample stacked
newsample2 = X[0].astype(np.int16)
for i in range(15):
    newsample2 = np.hstack((newsample2,X[randint(0,num_interval)])).astype(np.int16)  
scipy.io.wavfile.write("newsample2.wav", 16000, newsample2)
"""
