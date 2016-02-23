from fuel.datasets.youtube_audio import YouTubeAudio
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from random import randint



data = YouTubeAudio('XqaJ2Ol5cC4')
stream = data.get_example_stream()
it = stream.get_epoch_iterator()
sequence = next(it)

"""
#Plotting a piece of the sample
newsample = sequence[0][160000:240000]
plt.plot(newsample)
plt.ylabel('some numbers')
plt.show()
scipy.io.wavfile.write("newsample.wav", 16000, newsample)
"""

total_time=3*60*60+1*60+29.0
print total_time
total_len = sequence[0].shape[0]
print total_len
len_of_1sec_sample = total_len/total_time
print len_of_1sec_sample

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


