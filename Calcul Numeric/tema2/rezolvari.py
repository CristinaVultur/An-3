import numpy as np
import proceduri as proc
import time

#Exercitiul 2

#construim matricea sistemului
print("Exercitiul 2, varianta 5")

d = 21.0
f = -7.0
c = -7.0
n = 20

A = np.zeros((n,n))
A[0][0] = d  #d*x1 + f*x2 = 2
A[0][1] = f
for i in range(1, n-1): #c*x[1,..,n-2] + d*x[2,...,n-1]+ f*x[1,..,n]
    A[i][i-1] = c
    A[i][i] = d
    A[i][i+1] = f

A[n-1][n-2] = c #c*xn−1 + d*xn = 2
A[n-1][n-1] = d

b = np.zeros((n,1))
b[0] = 2
for i in range (1, n-1):
    b[i] = 1
b[n-1] = 2

tol = 10 ** (-10)

L, R, x = proc.FactLR(A, b, tol)

print(f'Soluția Sistemului: \n{x}')

print(A@x)


print('-----------------------')

print("Exercitiul 3, varianta 5")

A = np.array([[12., 9., 17.],
              [4., 2., 5.],
              [20., 22., 38.]])
b = np.array([[31.], [12.], [50.]])

det, Inv = proc.InvDet(A)

print("Determinantul")
print(det)
print("Inversa")
print(Inv)

x3 = Inv @ b

print(f'Soluția Sistemului: \n{x3}')

print(A@x3)

print('-----------------------')
print("Exercitiul 4, varianta 5")
print("Punctul 1")
n = 6

b = np.zeros((n,1))
for i in range (1, n+1):
    b[i-1] = i*i
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
#2
print("--------Punctul 2---------")

if proc.Sylvester(A):
    print ("Matricea este pozitiv definita")
else:
    print("Matricea nu este pozitiv definita")

#3
print("--------Punctul 4,5,6---------")
print("--------Matricea L---------")

L = proc.FactCholesky(A,b)

print(L)

x = proc.MetCholesky(A,b)
print("--------Solutia Sistemului---------")
print(x)
#print(A@x)