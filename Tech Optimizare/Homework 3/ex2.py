import numpy as np
from scipy.optimize import minimize
from matplotlib.pyplot import *
from numpy import linalg as la

# define function
f = lambda x: (x[0] - 2) ** 4 + (x[0] - 2 * x[1]) ** 2


# Proj = lambda x: np.array([np.min(np.array([1, 2 / la.norm(x)])) * i for i in x])

# proiectia functiei noastre pe multimea Q = {x apartine R^2| ||x||<=2}

def Proj(x):
    if la.norm(x) <= 2:
        return x
    else:
        return (2 / la.norm(x)) * x


def gradient(x):  # gradient of our function
    a11 = 4 * (x[0] - x[1] - 1)
    a22 = -4 * x[0] + 8 * x[1]
    return np.array([a11, a22])


# define epsilon and the x where the algorithm will start
eps = 10e-6
x00 = np.array(70 * np.random.rand(2, 1))

# avem nevoie de un punct din multimea Q deci facem proiectia

x0 = Proj(x00)


# constangerea pentru pb noastra de optimizare
def constraint(x):
    return np.atleast_1d(2 - la.norm(x))


# gasim punctul minim folosind scipy pentru verificare

optim = minimize(f, x0, method="SLSQP", constraints={"fun": constraint, "type": "ineq"})

xoptim = np.array([[optim.x[0]], [optim.x[1]]])


# functia pentru a gasi alpha pt pasul de backtracking

def BacktrackingSearch(x0):
    t = 1
    x = x0
    beta = 0.8
    alpha = 0.3

    # Armijo condition
    while f(x + t * (-gradient(x)))[0] > (f(x) + alpha * t * gradient(x) * (-gradient(x)))[0]:
        t *= beta

    return t


# hessiana functiei noastre
hessian = np.array([[4, -4], [-4, 8]])


# definim functia fradient proiectant
# bkt - false atunci foloseste pas constant
# bkt - true pas ales prin backtracking

def Gradient_Projection(x0, eps, bkt=False):
    curve_y = []
    curve_y2 = []
    iterations = []

    k = 0  # initializeaza

    # in functie de ce pas am ales calculam alpha

    if bkt is False:
        Lips = np.max(la.eigvals(hessian))  ### Constanta Lipschitz a gradientului
        alpha = 1 / Lips
    else:
        alpha = BacktrackingSearch(x0)

    x_old = x0

    grad = gradient(x_old)  # compute the gradient

    x_new = Proj(x_old - alpha * grad)  # calculam x1

    # this is for plotting

    iterations.append(k)
    curve_y.append(f(x_old)[0] - f(xoptim)[0])
    curve_y2.append(la.norm(x_new - x_old))

    # calculam criteriul de oprire ||xk+1 - xk|| <= eps
    criteriu_stop = la.norm(x_new - x_old)
    k += 1

    while criteriu_stop > eps ** 2:

        # doar la backtracking calculam mereu un nou alpha la fiecare iteratie
        if bkt is True:
            alpha = BacktrackingSearch(x_old)

        x_old = x_new

        grad = gradient(x_old)  # compute the gradient

        x_new = Proj(x_old - alpha * grad)  # calculate xk+1

        criteriu_stop = la.norm(x_new - x_old)

        # for plotting

        iterations.append(k)
        curve_y.append(f(x_old)[0] - f(xoptim)[0])
        curve_y2.append(la.norm(x_new - x_old))

        k += 1

    return x_new, iterations, curve_y, curve_y2


# solutiile impreuna cu listele pentru plot

solMGPC, iterations_const, curve_y_const1, curve_y_const2 = Gradient_Projection(x0, eps)

solMGPK, iterationsbkt, curve_y_bkt1, curve_y_bkt2 = Gradient_Projection(x0, eps, bkt=True)

#verify
print(f'Solutia pentru Metoda Gradient Proiectant cu Pas constant: {solMGPC}')
print(f'Solutia pentru Metoda Gradient Proiectant cu ales prin Backtracking: {solMGPK}')
print(f'Solutia data de Scipy: {optim.x}')

plot(iterations_const, curve_y_const1)
plot(iterationsbkt, curve_y_bkt1, color='green')
xlabel('iterations')
ylabel('f(xk)-f*')
legend(['Graph1'])
show()

plot(iterations_const, curve_y_const2)
plot(iterationsbkt, curve_y_bkt2, color='green')
xlabel('iterations')
ylabel('||xk+1 - xk||')
legend(['Graph2'])
show()
