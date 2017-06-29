import h5py
import os
import ast
import numpy as np
import matplotlib.pyplot as plt
import regex as re
from D_gmtools.gm_tools import *
from novainstrumentation import filter, smooth


def openH5(filename):
	print(filename)

	f = h5py.File(filename, 'r')

	Macs = [key for key in f.keys()][0]

	data_group = f[Macs + "/raw"]

	fs = f[Macs].attrs["sampling rate"] * 1.0

	data1 = data_group["channel_4"][:, 0]
	data2 = data_group["channel_5"][:, 0]


	return data1, data2, fs


bvp1, bvp2, fs = openH5("Exercise11/ctrx_0007803B4686_2017-04-11_12-16-22.h5")

plt.plot(bvp2)
plt.show()

bvp2 = bvp2[:100000]
bvp22 = filter.lowpass(bvp2, f=20, fs= fs, order=2)
bvp22 = bvp22 - smooth(bvp22, window_len=2000)
rising, falling, r,f, pks = R_F_Amp(bvp22, 0.3*max(bvp22), 0.2*max(bvp22))


# Quantization of the derivative
ds1 = np.diff(bvp22)
ds1 = np.clip(np.around(ds1 * 2), -1, 1).astype(int)
ds1 = np.insert(ds1, 0, 0)
x = np.empty(len(ds1), dtype='str')
x[find(ds1 == -1)] = '-'
x[find(ds1 == 0)] = '_'
x[find(ds1 == 1)] = '+'
# ds1 = medfilt(ds1, win_size)
bvp22 = (bvp22-mean(bvp22))/max(bvp22)
plt.plot(bvp22)
plt.plot(rising)
plt.show()
Matrix = [rising, x]

string = mergeChars(Matrix)

plt.plot( (bvp22-mean(bvp22))/max(bvp22))
pos1=[]
pos2=[]
pos3=[]
pos4=[]
pos5=[]

for i in re.finditer(r'(1\+).+?(1-).+?(0[\+_-]).+?(0-).+?', string):
	pos1.append(i.span()[0]//2)
	pos4.append(i.span()[1] // 2)
	print(re.search(i.group(2), i.group()).start())
	print(re.search(i.group(3), i.group()).start())
	pos2.append(i.span()[0]//2 + re.search(i.group(2), i.group()).start()//2)
	pos3.append(i.span()[0]//2 + re.search(i.group(3), i.group()).start()//2)

	axvspan(i.span()[0]//2, i.span()[1]//2, facecolor='0.5', alpha=0.5)

plt.vlines(pos1, -1, 1)
plt.vlines(pos2, -1, 1)
plt.vlines(pos3, -1, 1)
plt.vlines(pos4, -1, 1)
plt.show()
