import regex as re
import pandas as pd
from gm_tools import *

vm = np.array(pd.read_csv('Exercise8/coord_demo3.txt', header=None))
x, y = vm[:, 0], vm[:, 1]
plt.plot(x)
plt.plot(y)
dx1 = normalization(np.diff(x), -1, 1)
dx1 = np.clip(np.around(dx1 * 2), -1, 1)
dy1 = normalization(np.diff(y), -1, 1)
dy1 = np.clip(np.around(dy1 * 2), -1, 1)
x1 = np.empty(len(dx1) + 1, dtype='str')
x1[find(dx1 == -1)] = 'l'
x1[find(dx1 == 0)] = 'f'
x1[find(dx1 == 1)] = 'r'
y1 = np.empty(len(dy1) + 1, dtype='str')
y1[find(dy1 == -1)] = 'd'
y1[find(dy1 == 0)] = 's'
y1[find(dy1 == 1)] = 'u'

StrM = [x1, y1]
str = mergeChars(StrM)

#Left
plt.plot(x)


for i in re.finditer(r'(fu)+', str):
	print(i.span()[0])
	axvspan(i.span()[0]//2, i.span()[1]//2, facecolor='0.5', alpha=0.5)
plt.show()

print()

