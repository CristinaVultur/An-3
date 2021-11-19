import numpy as np
import proceduri as proc
import scipy.sparse.linalg as sp
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
print("Exercitiul 1, varianta 5")
print("Punctul 1")
n = 6
#1
a = np.zeros((n,1))
for i in range (1, n+1):
    a[i-1] = pow(3,n-i+1)

print(a)
A = np.zeros((n,n))

for i in range (0, n):
    for j in range (0,n):
        A[i][j] = a[abs(i-j)]

print(A)


b = np.zeros((n,1))
for i in range (1, n+1):
    b[i-1] = i*i
print(b)
#2
print("--------Punctul 2---------")
if proc.Sylvester(A):
    print ("Matricea este pozitiv definita")
else:
    print("Matricea nu este pozitiv definita")
#3

#4

print("--------Punctul 4---------")

x0 = np.zeros((len(b),1))

x = proc.GradConjugat(A, b ,10 ** -10, x0)

print("Solutia sistemului")
print(x)
x = sp.cg(A,b)
print(x)


#Ex4 a
print("Exercitiul 4, varianta 5")

X = np.array([0, 1, 3, 6])
Y = np.array([18, 10, (-18), 90])

y = proc.MetNDD(X, Y, x0)
print(y)


yn = proc.MetNewton(X, Y, x0)
print(yn)

print("Exercitiul 3, varianta 5")

def f(x,y):
    return 2*(x**2) + 3*x*y + x + 2*(y**2) - y

a = -4
b = 3
c = -3
d = 4
print("--------Punctul 1---------")

x = np.linspace(a, b, 50)
y = np.linspace(c, d, 50)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')

#surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0)
#fig.colorbar(surf)
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');
#plt.show()


print("--------Punctul 2---------")

x0 = np.array([[a],[c]])
#x0[0] = a
#x0[1] = c
b = np.array([[-1],[1]])
#b = np.zeros((2,1))
#b[0] = -1
#b[1] = 1
A = np.array([[4,3],[3,4]])

r = b - A.dot(x0)
v = r.copy()
Av = A.dot(v)
alpha = - np.dot(np.squeeze(v), np.squeeze(r)) / np.dot(np.squeeze(v), np.squeeze(Av))
x1 = x0 - alpha * v
r = b - A.dot(x1)
beta = np.dot(np.squeeze(r), np.squeeze(Av)) / np.dot(np.squeeze(v) , np.squeeze(Av))
v = r + beta * v

print(x1)

Av = A.dot(v)
alpha = - np.dot(np.squeeze(v), np.squeeze(r)) / np.dot(np.squeeze(v), np.squeeze(Av))
x2 = x1 - alpha * v
r = b - A.dot(x2)
beta = np.dot(np.squeeze(r), np.squeeze(Av)) / np.dot(np.squeeze(v) , np.squeeze(Av))
v = r + beta * v

print(x2)

print("--------Punctul 3---------")

x = sp.cg(A,b)
print(x)

xp = [x[0][0]]
y = [x[0][1]]
z= [f(x[0][0], x[0][1])]
sc = ax.scatter(xp,y,z)
ax.text(x[0][0],  x[0][1], f(x[0][0], x[0][1]), "punct minim", color='red')
plt.show()

