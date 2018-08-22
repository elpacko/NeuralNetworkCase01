import numpy as np


def column(_matrix, _i):
    return [row[_i] for row in _matrix]


def calculateE(_yr, _yn):
    _e = []
    for k in range(len(_yr)):
        _e.append((_yr[k] - _yn[k]).item())
    return _e


def calculateYN(_w, _x):
    _yn = []
    for k in range(len(_x)):
        _yn.append((_w * np.transpose(_x[k])).item())
    return _yn


def calculateW(_x, wprev, _e, mu):
    _w = (wprev + ((np.array(_e) * _x) * mu))
    return _w


def square(_list):
    return [i ** 2 for i in _list]


def calculateError(_e):
    return sum(square(_e))/2


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


xr = trainingData[:, [0, 1]]
x0 = np.ones((xr.shape[0], 1))
x = np.hstack((x0, xr))
# w = [(random.randint(randomLowerBound, randomHigherBound)) for k in range(3)]
w = np.random.uniform(size=(1, 3))
yr = trainingData[:, [2]]

for currentEpoch in range(epochStop):
    yn = calculateYN(w, x)
    e = calculateE(yr, yn)
    error = calculateError(e)
    print str(currentEpoch) + ':' + str(error)
    if error < errorStop:
        break
    w = calculateW(x, w, e, 0.0001)
xt = np.array([1, 60, 15]).T  # generamos cada vector de x en cada sample k
dosis = 0
w = np.squeeze(np.asarray(w))
for i in range(len(w) - 1):
    dosis = dosis + w[i] * xt[i]
print dosis
