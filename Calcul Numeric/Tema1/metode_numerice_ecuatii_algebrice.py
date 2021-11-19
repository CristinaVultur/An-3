import numpy as np

"""
Funcție care caută intervalele pe care funcția are o soluție.
f(a) * f(b) < 0 -> EXISTENȚA
"""


def cauta_intervale(f, a, b, n):
    """
    Parameters
    ----------
    f : funcția asociată ecuației f(x)=0.
    a : capătul din stânga interval.
    b : capătul din dreapta interval.
    n : nr de subintervale în care împărțim intervalul global (a, b).

    Returns
    -------
    Matricea 'intervale' cu 2 linii; prima linie -> capăt st interval curent
    si a doua linie -> capat dr
    si un nr de coloane = nr radacini
    """

    x = np.linspace(a, b, n + 1)  # returnează n+1 numere, situate la distanțe egale, din cadrul intervalului [a, b]
    for i in range(len(x)):  # range: for i = 0, len(x); i++
        if (f(x[
                  i]) == 0):  # capetele intervalelor mele nu au voie să fie 0; tb să avem soluțiile în intervale, nu la capete
            print("Schimba nr de Intervale")
            exit(0)

    matrice = np.zeros(
        (2, 1000))  # returnează un nou vector plin de 0; pt că am (2, 1000) -> matrice cu 2 rânduri și 1000 coloane
    z = 0
    for i in range(n):
        if f(x[i]) * f(x[i + 1]) < 0:  # existență soluție
            matrice[0][z] = x[i]
            matrice[1][z] = x[i + 1]
            z += 1

    matrice_finala = matrice[:, 0:z]  # iei ambele 2 linii și doar coloanele de la 0 la z (numărat mai sus)
    return matrice_finala


"""
Funcție care implementează algoritmul metodei secantei
"""

def MetSecantei(f,x0 , x1, eps):
    """

    :param f:funcția asociată ecuației f(x)=0.
    :param x0:, :param x1:  first 2 guesses
    :param eps: toleranța / eroarea (epsilon).
    :return:
    Soluția aproximativă, dar și numărul de iterații N necesar
     pt a obține soluția cu eroarea eps.
    """

    n = 0
    xm = 0
    xk = 0
    c = 0;

    if(f(x0) * f(x1) < 0):
        while True:
            #calculam val intermediara
            xk = (x0 * f(x1) - x1 * f(x0))/(f(x1) - f(x0))
            #verificam daca xk este sol a ecuatiei
            c = f(x0) * f(xk)

            #update la intervalele de valori

            x0 = x1
            x1 = xk

            #incrementam nr de iteratii
            n += 1

            #daca xk radacina iesim din loop

            if(c==0):
                break;
            xm = ((x0 * f(x1) - x1 * f(x0)) /
                  (f(x1) - f(x0)));

            if (abs(xm - xk) < eps):
                break;

        return xk, n

    else:
        print("Alege alt interval")
        return xk, n

def MetPozFalse(f, a, b, eps):
    """

    :param f: funcția asociată ecuației f(x)=0.
    :param a:capătul din stânga interval.
    b : capătul din dreapta interval.
    :param eps:toleranța / eroarea (epsilon).

    :return: Soluția aproximativă, dar și numărul de iterații N
    necesar pt a obține soluția cu eroarea eps.
    """

    n = 0
    condition = True
    x = (a * f(b) - b * f(a)) / (f(b) - f(a))

    while condition:

        n += 1
        #gaseste pct in care atinge axa x


        #verificam daca este rad

        if f(x) == 0:
            return x, n
        #decidem pe ce parte contiuam

        if f(a) * f(x) < 0:
            b = x
            x = (a * f(b) - b * f(a)) / (f(b) - f(a))
            condition = abs(x - b)/abs(b) >= eps
        else:
            a = x
            x = (a * f(b) - b * f(a)) / (f(b) - f(a))
            condition = abs(x - a) / abs(a) >= eps

    return x,n