import numpy as np
from numpy.random import randn, rand
import matplotlib.pyplot as plt

# Cubic Neuronal Unit


mu = 0.000000001  # learning rate
n = 3  # 2 x values + Bias
nw = (n ** 3 + n) / 3  # Number of Weights
colw = rand(nw).T / 3  # Weight Column
colDw = (np.zeros(nw)).T  # column with Deltas of Weights
rawx = np.zeros(nw)  # row of correlated  x values

epochs = 50000

x1 = np.array([7, 9, 11.5, 14, 18, 25, 35.5], dtype=float).T  # weight of child
x2 = np.array([0.4, 0.75, 1.5, 2.5, 4.5, 7.5, 10.5], dtype=float).T  # age of child
yr = np.array([4.0, 5.0, 7.0, 9.0, 10.0, 14.0, 20.0], dtype=float).T

yn = np.zeros(np.size(yr)).T  # neuronal output for each real output
e = np.zeros(np.size(yr)).T  # column vector that holds the error

SSE = np.zeros(epochs)  # objective function (sum of squared errors)

for epoch in range(0, epochs):
    for k in range(0, np.size(yr)):  # loop to cycle trough discrete positions of yr
        x = np.array([1, x1[k], x2[k]]).T   # 1 = bias, x1[k] = weight of child, x2[k] age of child
        pom = 0  # counter
        i = 0
        for i in range(0, n-1):
            j = i
            for j in range(0, n - 1):
                m = j
                for m in range(0, n-1):
                    rawx[pom] = x[i] * x[j] * x[m]
                    pom = pom + 1
        yn[k] = sum(rawx * colw)  # calculation of neural output based on weights
        e[k] = yr[k] - yn[k]

        # begin to calculate weight increase
        pom = 0
        i = 0
        for i in range(0, n-1):
            j = i
            for j in range(0, n - 1):
                m = j
                for m in range(0, n - 1):
                    colDw[pom] = mu * e[k] * (x[i] * x[j] * x[m])
                    pom = pom + 1
        colw = colDw + colw
    SSE[epoch] = sum(e * e)
    if SSE[epoch] < 0.0000000001:
        break
    if np.isnan(SSE[epoch]):
        break
    if epoch % 1000 == 0 or epoch < 10:
        print(SSE[epoch])

# plt.figure()
# plt.plot(range(1, epochs + 1), SSE, 'k'), plt.title(
#     'Training performance, SSE at every training epoch'), plt.xlabel('epoch'), plt.ylabel('SSE'), plt.grid()
# plt.figure()
# plt.plot(yr, 'b'), plt.title('Real Value (Blue) vs Model (Green) output'), plt.xlabel('k'), plt.ylabel('ml')
# plt.plot(yn, 'g'), plt.grid()
# plt.show()

print(colw)

respuesta = ""
while respuesta != "q":
    pesoBB = input("Peso del BB en kg:")
    edadBB = input("Edad BB:")
    xt = np.array([1, float(pesoBB), float(edadBB)]).T  # generamos cada vector de x en cada sample k
    dosis = 0
    pom = 0  # counter
    rawx_test = np.zeros(nw)  # row of correlated  x values
    i = 0
    for i in range(0, n-1):
        j = i
        for j in range(0, n - 1):
            m = j
            for m in range(0, n-1):
                rawx_test[pom] = xt[i] * xt[j] * xt[m]
                pom = pom + 1
    dosis = sum(rawx_test * colw)  # calculation of neural output based on weights
    print ("Dosis:" + str(dosis))
