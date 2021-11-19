
#Sa se genereze o variabila normala N(4,7)folosind o metoda de compunere-respingere (curs 6).'''

import random
import numpy as np

#generam
#X∼N(0,1)
# Y este o variabil ̆a normala N(m,σ), Y=m+σX.

def compunere_respingere(m, tetha):
    """
    algoritm pentru generarea variabilei X variabila normala N(0,1)

    """
    while True:
        u = random.uniform(0, 1) #generam U~U(0,1)
        y = random.expovariate(1) #Y~Exp(1)
        if u > np.e **(-y**2/2 +y -0.5):
            continue
        else:
            x1 = y
            u = random.uniform(0,1) #U ~ U(0,1)
            if u <= 0.5:
                s = 1
            else:
                s = -1
            x = s*x1
            # vrem N(4,7)
            # X∼N(0,1)
            # Y este o variabila normala N(m,σ), Y=m+σX.
            Y = m + tetha * x
            return Y

num_samples = 10000
m = 4
tetha = 7

print(f'Variabila normala N(4,7): {compunere_respingere(m, tetha)}')

sample = [compunere_respingere(m, tetha) for _ in range(num_samples)]

print("Valori teoretice:")
print("Medie:", m, "-", "Dispersie:", tetha)

print("Eșantioane generate")

m_prim = np.mean(sample)
s_patrat = (np.array([x **2 for x in sample])).sum()/num_samples - m_prim**2
print("Medie:", m_prim, "-", "Dispersie:", np.sqrt(s_patrat))


''' S˘a se genereze variabila hipergeometric˘a cu parametrii citit¸i de la tastatur'''
print()
def hipergeometrica(A, B, n):
    """

    :param A: bile albe
    :param B: bile negre
    :param n: nr de extrageri
    :return: x = nr de bile albe extrase - variablila hipergeometrica
    """

    N = A + B # nr total de bile
    p = A/N
    j = 0 #nr extrageri curente
    x = 0
    while True:
        u = random.uniform(0,1) #generam U~U(0,1)
        if u < p: #s-a extras o bila alba
            x += 1
            s = 1
        else:
            s = 0 #s-a extras o bila neagra
        N += -1 #scadem nr de bile ramase (experiment fara intoarceri)
        A = A - s
        p = A/N
        j += 1

        if j == n: #ajungem la nr total de extrageri
            return x



A = int(input('Alege numarul de bile albe: '))
B = int(input('Alege numarul de bile negre: '))
n = int(input(('Alege numarul de extrageri: ')))

X_h = hipergeometrica(A, B, n)
print(f'Variabila Hipergeometrica {X_h}')