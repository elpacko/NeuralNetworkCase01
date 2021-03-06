import numpy as np
from numpy.random import randn, rand
import matplotlib.pyplot as plt

plt.clf()
plt.cla()
plt.close()

mu = 0.0001
n = 3
w = rand(n)
x = np.zeros(n)
epochs = 300

x1 = np.array([7, 9, 11.5, 14, 18, 25, 35.5], dtype=float).T #peso
x2 = np.array([0.4, 0.75, 1.5, 2.5, 4.5, 7.5, 10.5], dtype=float).T #promedio edad
yr = np.array([4.0, 5.0, 7.0, 9.0, 10.0, 14.0, 20.0], dtype=float).T #dosis
yn = np.zeros(np.size(yr)).T
e = np.zeros(np.size(yr)).T

# sum of square error (suma del error cuadratico)
SSE = np.zeros(epochs)

for epoch in range(0, epochs):
    for k in range(0, np.size(yr)):  # ciclo para cada sample k de los datos de entrenamiento hasta el (tamano de yr)-1
        x = np.array([1, x1[k], x2[k]]).T  # generamos cada vector de x en cada sample k
        yn[k] = 0
        for i in range(len(w) - 1):
            yn[k] = yn[k] + w[i] * x[i]
        e[k] = yr[k] - yn[k]
        # w = mu * e[k] * x + w  # actualizacion de los pesos sample by sample (u online learning)
        w = mu * e[k] * x.T + w  # actualizacion de los pesos sample by sample (u online learning)
    SSE[epoch] = sum(e * e)
    if epoch % 10 == 0 or epoch < 10:
        print str(epoch) + ":" + str(SSE[epoch])

print "Training Finished"
print "Found weights " + str(w)

xt = np.array([1, 50, 16]).T  # generamos cada vector de x en cada sample k
dosis = 0
for i in range(len(w) - 1):
    dosis = dosis + w[i] * xt[i]
print dosis


plt.figure()
plt.plot(range(1, epochs + 1), SSE, 'k'), plt.title(
    'Training performance, SSE at every training epoch'), plt.xlabel('epoch'), plt.ylabel('SSE'), plt.grid()
plt.figure()
plt.plot(yr, 'b'), plt.title('Real Value (Blue) vs Model (Green) output'), plt.xlabel('k'), plt.ylabel('ml')
plt.plot(yn, 'g'), plt.grid()
plt.show()
