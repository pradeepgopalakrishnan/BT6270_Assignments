import matplotlib.pyplot as plt
import numpy as np

f=open('data.txt','r')
lines=f.readlines()
current=[]
freq=[]
for i,line in enumerate(lines):
    line=float(line)
    if(i%2):
        freq.append(line)
    else:
        current.append(line)

#This is to correct for the time scale of the simulation. The result gives time in seconds.
freq=[100*i for i in freq]

plt.plot(current, freq)
plt.title("Plot of Spiking Frequency vs External Current in a Neuron")
plt.ylabel("Freq. in Hz")
plt.xlabel("External Current in microAmpere")
plt.show()

