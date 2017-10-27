import numpy as np
data = np.genfromtxt('data.txt', delimiter=',', names=True)
print data

print data['col1']    # array([   1.,   10.,  100.])
print data['col2']    # array([   2.,   20.,  200.])
print data['col3']    # array([   3.,   30.,  300.])


labels = np.genfromtxt('data.txt', delimiter=',', usecols=0, dtype=str)

raw_data = np.genfromtxt('data.txt', delimiter=',')[:,1:]
data = {label: row for label, row in zip(labels, raw_data)}
#
print data
# print '\n', data['row1']    # array([   1.,   10.,  100.])
# print data['row2']    # array([   2.,   20.,  200.])
# print data['row3']    # array([   3.,   30.,  300.])
