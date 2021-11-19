import numpy as np
from math import sqrt

def FactLR(A, b, tol):
    """

     ----------
    A : matrice pătratică.
    b : vectorul termenilor liberi.

    Returns
    -------
    L, R = matrici
    x = vector
    """
    n = np.shape(A)[0]
    L = np.zeros((n, n))
    R = np.zeros((n, n))

    R[0][0] = A[0][0] #r1 = a1
    np.fill_diagonal(L, 1) #lii = 1, i = 1, n

    #determinam matricile R si L
    for i in range(0, n-1):
        L[i+1][i] = A[i+1][i] / R[i][i]
        R[i][i+1] = A[i][i+1]
        R[i+1][i+1] = A[i+1][i+1] - L[i+1][i] * R[i][i+1]

    #Rezolvam Ly = b
    y = np.zeros(n)
    y[0] = b[0]
    for i in range (1, n):
        y[i] = b[i] - L[i][i-1] * y[i-1]

   # print(y)

    #Rezolvam Rx = y
    x = np.zeros(n)
    x[n-1] = y[n-1]/R[n-1][n-1]
    for i in range(n-2, -1, -1):
        x[i] = (y[i] - R[i][i+1]*x[i+1])/R[i][i]

    #print(R @ x)

    return L, R, x

def Minor(A, i, j):

    """
    :param A: matricea
    :param i: randul
    :param j: coloana
    :return: minorul lui a din care am eliminat randul i si coloana j
    """
    sub_m = np.copy(A)
    sub_m = np.delete(sub_m,i,0)   # stergem randul i
    sub_m = np.delete(sub_m,j,1) # stergem coloana j
    return sub_m

def Deternminant(A):

    """
    :param A: matrice
    :return: determinantul ei
    """
    #cazul 2x2

    if len(A) == 2:
        return A[0][0] * A[1][1] - A[0][1] * A[1][0]
    determinant = 0
    #calculam determinantul recursiv
    for c in range(len(A)):
        determinant += ((-1)**c) * A[0][c] * Deternminant(Minor(A, 0, c))
    return determinant


def InvDet(A):
    """

    :param A: matrice
    :return: determinantul si inversa matricei
    """
    determinant = Deternminant(A)

    if determinant == 0: #matricea nu este inversabila
        return None, None

    #caz 2x2:
    if len(A) == 2:
        return determinant, [[A[1][1] / determinant, -1 * A[0][1] / determinant],
                [-1 * A[1][0] / determinant, A[0][0] / determinant]]

    adjuncta = []
    # calculam matricea adjuncta

    for r in range(len(A)):
        rand = []
        for c in range(len(A)):
            minor = Minor(A, r, c)
            rand.append(((-1)**(r+c)) * Deternminant(minor))
        adjuncta.append(rand)

    # Transpunem matricea
    adjuncta = np.transpose(adjuncta)

    #calculam inversa
    for r in range(len(adjuncta)):
        for c in range(len(adjuncta)):
            adjuncta[r][c] = adjuncta[r][c]/determinant
    return determinant, adjuncta

def Sylvester(A):
    """
    :param A: matricea a
    :return: daca este pozitiv definita
    """

    #folosim criteriul lui Sylvester pt a dertermina daca matricea este pozitiv definita
    n = len(A)
    if A[0][0] <= 0:
        return False
    for i in range(1,n): #toate matricele din stanga sus (1x1,2x2,3x3) trebue sa aiba determinntul pozitiv
        A_c = np.copy(A)
        A_c = np.delete(A_c, slice(i+1,n), 0)
        A_c = np.delete(A_c, slice(i+1,n),1)
        det = Deternminant(A_c)
        print(A_c)
        print(det)
        if det <= 0:
            return False
    return True

def metSubDesc(A, b, tol):
    """

    Parameters
    ----------
    A : matrice pătratică, superior triunghiulară, cu toate elementele de pe diagonala principală nenule.
    b : vectorul termenilor liberi.
    tol : toleranță = valoare numerică foarte mică în raport cu care vom compara numerele apropiate de 0.

    Returns
    -------
    x = Soluția Sistemului.

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
            if abs(A[i][j]) > tol:
                print("Matricea nu este superior triunghiulară.")
                x = None
                return x

    # Verificam dacă toate elementele de pe diagonala principală sunt nenule => Si. este compatibil ddeterminat (adică am soluție unică)
    for i in range(n):
        if A[i][i] == 0:
            print("Sistemul nu este compatibil determinat.")
            x = None
            return x

    x = np.zeros((n, 1))
    x[n - 1] = b[n - 1] / A[n - 1][n - 1]

    k = n - 2
    while k >= 0:
        s = 0
        for j in range(k + 1, n):
            s += x[j] * A[k][j]

        x[k] = (1 / A[k][k]) * (b[k] - s)
        k -= 1

    return x


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


def FactCholesky(A,b):
    """

    :param A: matrice
    :param b: termenii liberi
    :return: matricea l
    """
    n = len(A)
    L = np.zeros((n, n))

    for i in range(0,n):
        for k in range(i+1):
            tmp_sum = sum(L[i][s] * L[k][s] for s in range(k))
            if i == k:
                L[i][k] = sqrt(A[i][i] - tmp_sum)
            else:
                L[i][k] = (1.0 / L[k][k] * (A[i][k] - tmp_sum))

    return L

def MetCholesky(A,b):
    """
    :param A: matrice
    :param b: termenii liberi
    :return: solutia sistemului
    """
    L = FactCholesky(A,b)
    y = metSubAsc(L, b, 10 ** -10)
    x = metSubDesc(np.transpose(L), y, 10 ** -10)

    return x
