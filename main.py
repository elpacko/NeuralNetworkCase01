import numpy as np


def column(matrix, i):
    return [row[i] for row in matrix]


def calculateE(yr, yn):
    e = []
    for k in range(len(yr)):
        e.append(yr[k]-yn[k])
    return e


def calculateYN(w,x):
    yn=[]
    for k in range(len(x)):
        yn.append(w*np.transpose(x[k]))
    return yn


errorStop = 0.0004
epochStop = 100

trainingData = np.matrix([
    [7, 0.4, 4.0],
    [9, 0.75, 5.0],
    [11.5, 1.5, 7.0],
    [14, 2.5, 9.0],
    [18, 4.5, 10.0],
    [25, 7.5, 14.0],
    [35.5, 10.5, 20.0]
    ]
)


randomLowerBound = 0
randomHigherBound = 100
xr = trainingData[:, [0, 1]]
x0 = np.ones((xr.shape[0], 1))
x = np.hstack((x0, xr))
# w = [(random.randint(randomLowerBound, randomHigherBound)) for k in range(3)]
w = np.random.uniform(size=(1, 3))
yr = trainingData[:, [2]]

for currentEpoch in range(epochStop):
    yn = calculateYN(w, x)
    e = calculateE(yr, yn)
    print yn
    print e
    if 1 < errorStop:
        break
