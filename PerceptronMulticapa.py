# Sigmoid NN

import numpy as np
from numpy.random import randn, rand
import matplotlib.pyplot as plt

plt.clf()
plt.cla()
plt.close()

mu = 0.01
n = 3

x = np.zeros(n)
#capa oculta
n0 = 3  #numero de neuronas en la capa oculta
W = np.random.rand([n0, n])  # matriz de pesos w inicializando de manera aleatorea
DW = np.zeros(n0, n)

#capa de salida
wout = rand(n0)
Dwout = np.zeros(n0)
yk = np.zeros(n0).T



epochs = 3000
dw = np.zeros(n)
x1 = np.array([7, 9, 11.5, 14, 18, 25, 35.5], dtype=float).T #peso
x2 = np.array([0.4, 0.75, 1.5, 2.5, 4.5, 7.5, 10.5], dtype=float).T #promedio edad
yr = np.array([4.0, 5.0, 7.0, 9.0, 10.0, 14.0, 20.0], dtype=float).T #dosis

# normalizar z-score
x1mean = np.mean(x1)
x1std = float(np.std(x1)*2.0)
x2mean = np.mean(x2)
x2std = float(np.std(x2)*2.0)
yrmean = np.mean(yr)
yrstd = float(np.std(yr)*2.0)

zx1 = (x1-x1mean)/x1std
zx2 = (x2-x2mean)/x2std
zyr = (yr-yrmean)/yrstd

#zx1 = (x1-np.mean(x1))/float(np.std(x1)*2.0)
#zx2 = (x2-np.mean(x2))/float(np.std(x2)*2.0)
#zyr = (yr-np.mean(yr))/float(np.std(yr)*2.0)



x1 = zx1
x2 = zx2
yr = zyr

yn = np.zeros(np.size(yr)).T
e = np.zeros(np.size(yr)).T

# sum of square error (suma del error cuadratico)
SSE = np.zeros(epochs)

for epoch in range(0, epochs):
    for k in range(0, np.size(yr)):  # ciclo para cada sample k de los datos de entrenamiento hasta el (tamano de yr)-1
        x = np.array([1, x1[k], x2[k]]).T  # generamos cada vector de x en cada sample k
        yn[k] = 0
        #for i in range(len(w) - 1):
        #    yn[k] = yn[k] + w[i] * x[i]
        z = sum(W*x)
        #yn[k] = (2.0000/(1.00000 + np.exp(-float(z))))-1.00000
        yk = (2.0000 / (1.00000 + np.exp(-float(z)))) - 1.00000
        yn[k] = wout * yk  #salida neuronal
        e[k] = yr[k] - yn[k]
        # w = mu * e[k] * x + w  # actualizacion de los pesos sample by sample (u online learning)
        DW = mu * e[k] * x * wout * ((2*x.T*np.exp(-z))/((1+np.exp(-z))**2))  # + w  # actualizacion de los pesos sample by sample (u online learning), aqui va el resultado de la derivada
        Dwout = mu * e[k] * yk  #incremento de los pesos de salida
        w = DW + w
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


respuesta = ""
while respuesta != "q":
    pesoBB = input("Peso del BB en kg:")
    edadBB = input("Edad BB:")
    #xt = np.array([1, float(pesoBB), float(edadBB)]).T  # generamos cada vector de x en cada sample k
    zx1 = (pesoBB - x1mean) / x1std
    zx2 = (edadBB - x2mean) / x2std
    xt = np.array([1, float(zx1), float(zx2)]).T  # generamos cada vector de x en cada sample k
    dosis = 0
    z = sum(w * xt)
    dosis =  (2.0000 / (1.00000 + np.exp(-float(z)))) - 1.00000

    zyr = (dosis  * yrstd ) + yrmean
    print ("Dosis:" + str(zyr))
