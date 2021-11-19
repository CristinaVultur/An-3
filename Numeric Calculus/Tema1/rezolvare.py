import numpy as np
import matplotlib.pyplot as plt
import metode_numerice_ecuatii_algebrice as mnea
from math import e

def f(x):
    y = (np.sin(x) - e**(-x))
    return y

(a, b) = (0, 10)

eps = 10 ** -5


#identificam intervalele pe care funcția f admite o sol unică

interval = mnea.cauta_intervale(f, a, b, 4)
print(interval)
dim = np.shape(interval)[1]

#Desen Grafice
x_grafic = np.linspace(0, 10, 100)
y_grafic = f(x_grafic)

plt.plot(x_grafic, y_grafic, linewidth = 3)
plt.axvline(0, color = 'black')
plt.axhline(0, color = 'black')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('V5 Grafic')
plt.grid(True)

for i in range(dim):
    plt.plot(interval[:, i], np.zeros((2, 1)), color = 'red', linewidth = 10)
plt.show()

#aflam toate rădăcinile conform metodei secantei

r = np.zeros(dim)
for i in range(dim):
    r[i], N = mnea.MetSecantei(f, interval[0,i], interval[1, i], eps)
    print("Metoda Secantei")
    print("Ecuația sinx − e^−x =0")
    print("Intervalul [{:.3f}, {:.3f}]".format(interval[0,i], interval[1,i]))
    print("Solutia Numerica: x *= {:.3f}".format(r[i]))
    print("-----------------------------------")

plt.plot(r, f(r), 'o-', color = 'green', markersize = 10)
plt.show()


r = np.zeros(dim)
for i in range(dim):
    r[i], N = mnea.MetPozFalse(f, interval[0,i], interval[1, i], eps)
    print("Metoda Pozitiei False")
    print("Ecuația sinx − e^−x =0")
    print("Intervalul [{:.3f}, {:.3f}]".format(interval[0,i], interval[1,i]))
    print("Solutia Numerica: x *= {:.3f}".format(r[i]))
    print("-----------------------------------")

plt.plot(r, f(r), 'o-', color = 'green', markersize = 10)
plt.show()