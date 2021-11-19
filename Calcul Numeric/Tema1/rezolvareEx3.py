import MetodeNumericePentruEcuatiiLiniare as sel
import numpy as np

print("Exercitiul 3, varianta 5")
tol = 10 ** -16
A = np.array([[12,9,17], [4,2,5], [20,22,38]])
b = np.array([[31], [12], [50]])
x = sel.gaussPp(A, b, tol)
print("Solutia:")
print(x)
print("--------------------------------------------------------")
#ex 2 V5
print(b[0])
d = 21
f = -7
c = -7
n = 20

#construim matricea sistemului
print("Exercitiul 2, varianta 5")

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

x = sel.gaussPt(A, b, tol)
print("Vectorul solutiilor:")
print(x)
