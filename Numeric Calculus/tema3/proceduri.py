import numpy as np
from math import sqrt


def Minor(A, i, j):
    """
    :param A: matricea
    :param i: randul
    :param j: coloana
    :return: minorul lui a din care am eliminat randul i si coloana j
    """
    sub_m = np.copy(A)
    sub_m = np.delete(sub_m, i, 0)  # stergem randul i
    sub_m = np.delete(sub_m, j, 1)  # stergem coloana j
    return sub_m


def Deternminant(A):
    """
    :param A: matrice
    :return: determinantul ei
    """
    # cazul 2x2

    if len(A) == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    determinant = 0
    # calculam determinantul recursiv
    for c in range(len(A)):
        determinant += ((-1) ** c) * A[0][c] * Deternminant(Minor(A, 0, c))
    return determinant


def Sylvester(A):
    """
    :param A: matricea a
    :return: daca este pozitiv definita
    """

    # folosim criteriul lui Sylvester pt a dertermina daca matricea este pozitiv definita
    n = len(A)
    if A[0][0] <= 0:
        return False
    for i in range(1, n):  # toate matricele din stanga sus (1x1,2x2,3x3) trebue sa aiba determinntul pozitiv
        A_c = np.copy(A)
        A_c = np.delete(A_c, slice(i + 1, n), 0)
        A_c = np.delete(A_c, slice(i + 1, n), 1)
        det = Deternminant(A_c)
        print(A_c)
        print(det)
        if det <= 0:
            return False
    return True


def GradConjugat(A, b, tol, x=None):
    """
    A : matrice
        matrice simetrica poz definita
    b : vector

    x : valoarea de start

    return: solutia sistemului
    """

    r = b - A.dot(x)
    v = r.copy()
    for i in range(len(b)):
        Av = A.dot(v)
        alpha = - np.dot(np.squeeze(v), np.squeeze(r)) / np.dot(np.squeeze(v), np.squeeze(Av))
        x = x - alpha * v
        r = b - A.dot(x)
        if np.sqrt(np.sum((r ** 2))) < tol:
            print('Itr:', i)
            break
        else:
            beta = np.dot(np.squeeze(r), np.squeeze(Av)) / np.dot(np.squeeze(v) , np.squeeze(Av))
            v = r + beta * v
    return x


def MetNDD(X, Y, x):
    n = len(X);  # gradul polinomului Pn
    Q = np.zeros((n, n))
    for i in range(0, n):
        Q[i][1] = Y[i]
    for i in range(1, n):
        for j in range(1, i):
            Q[i][j] = (Q[i][j - 1] - Q[i - 1][j - 1]) / (X[i] - X[i - j + 1])
    Pn = Q[1][1]
    for k in range(1, n):
        p = 1
        for j in range(0, k):
            p = p * (x - X[j])
        Pn = Pn + Q[k][k] * p

    y = Pn
    return y


def metSubAsc(A, b, tol):
    """

    Parameters
    ----------
    A : matrice inferior triunghiulară.
    b : vectorul termenilor liberi.
    tol : toleranța.

    Returns
    -------
    soluția.

    """

    # Verificăm dacă matricea este pătratică
    m, n = np.shape(A)
    if m != n:
        print("Matricea nu este pătratică. Introduceți altă matrice.")
        x = None
        return x

    # Verificăm dacă matricea este superior triunghiulară
    for i in range(m):
        for j in range(i):
            if abs(A[j][i]) > tol:
                print("Matricea nu este inferior triunghiulară.")
                print(A)
                x = None
                return x

    # Verificam dacă toate elementele de pe diagonala principală sunt nenule => Si. este compatibil ddeterminat (adică am soluție unică)
    for i in range(n):
        if abs(A[i][i]) <= tol:
            print("Sistemul nu este compatibil determinat.")
            x = None
            return x

    x = np.zeros((m, 1))
    x[0] = b[0] / A[0][0]

    for k in range(1, n):

        sum = 0
        for j in range(k):
            sum += A[k][j] * x[j]

        x[k] = (1 / A[k][k]) * (b[k] - sum)

    return x


def MetNewton(X, Y, x):
    n = len(X)
    a = np.zeros((n, n))
    for i in range(0, n):
        a[i][0] = 1
    for i in range(1, n):
        for j in range(1, i+1):
            a[i][j] = 1
            for k in range(0, j - 1):
                a[i][j] = a[i][j] * (X[i] - X[k])

    c = metSubAsc(a, Y, 10 ** -10);
    Pn = c[0]
    for k in range(1, n):
        p = 1
        for j in range(0, k):
            p = p * (x - X[j])
        Pn = Pn + c[k] * p

    y = Pn
    return y
