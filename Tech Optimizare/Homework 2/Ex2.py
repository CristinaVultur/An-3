import numpy as np
from scipy.optimize import minimize
from matplotlib.pyplot import *


#define function
f = lambda x: x[0]**4 + x[0]*x[1] + (1+x[1])**2
x0 = np.zeros((2,1))

#search for optim

optim = minimize(f,x0)
print(optim)

xoptim = np.array([[0.69588439], [-1.3479422]])

def gradient(x): #gradient of our function
    a11 = 4*np.power(x[0],3)
    a22 = x[0] + 2*x[1] + 2
    return np.array([a11,a22])

def hessian(x): #hessian
    a = 12 * (x[0][0] ** 2)
    return np.array([[a, 1], [1, 2]])

#x,y for plotting

iterations = []
curve_y_constant1 = []
curve_y_back1 = []


curve_y_constant2 = []
curve_y_back2 = []

def Newton_constant(f, x0, epsilon,max_iteration):

    """

    :param f: function to minimise
    :param x0: first aproximation
    :param epsilon: stopping  criteria ||∇f(xk)|| ≤ epsilon

    :return: xn where f in minimised
    """

    xk = x0

    i = 0 #step count

    while (f(xk)[0]-f(xoptim)[0]) > epsilon:

        i += 1
        iterations.append(i)
        curve_y_constant1.append(f(xk)[0]-f(xoptim)[0])
        curve_y_constant2.append(np.linalg.norm(hessian(xk)))

        grad = gradient(xk)  # compute the gradient, hessian and hessian inverse for the current x
        hess = hessian(xk)
        hess_inv = np.linalg.inv(hess)

        # Newton direction:

        dk = hess_inv @ grad

        #𝑥𝑘+1=𝑥𝑘−𝛼𝑘∇2𝑓𝑥𝑘−1∇𝑓𝑥𝑘

        xnew = xk - dk
        xk = xnew

    print(f'Stopped at iteration {i} with {xnew}')
    return xnew

x0 = np.zeros((2,1))
xn = Newton_constant(f, x0, 1e-4, 10000)

iterationsbkt = []
def BacktrackingSearch(x0):
    t = 1
    x = x0
    beta = 0.8
    alpha = 0.3

    # Armijo condition
    while f(x + t * (-gradient(x)))[0]> (f(x) + alpha * t * gradient(x) *(-gradient(x)))[0]:
        t *= beta

    return t

def Newton_Backtracking(f, x0, epsilon,max_iteration):

    """

    :param f: function to minimise
    :param x0: first aproximation
    :param epsilon: stopping  criteria ||∇f(xk)|| ≤ epsilon

    :return: xn where f in minimised
    """

    xk = x0
    i = 0
    while (f(xk)[0] - f(xoptim)[0]) > epsilon:

        i += 1
        iterationsbkt.append(i)
        curve_y_back1.append(f(xk)[0] - f(xoptim)[0])
        curve_y_back2.append(np.linalg.norm(hessian(xk)))


        alpha = BacktrackingSearch(xk)

        grad = gradient(xk)  # compute the gradient, hessian and hessian inverse for the current x
        hess = hessian(xk)
        hess_inv = np.linalg.inv(hess)

        # Newton direction:

        dk = hess_inv @ grad

        xnew = xk - alpha * dk

        xk = xnew

    print(f'Stopped at iteration {i} with {xnew}')
    return xnew

xb = Newton_Backtracking(f, x0, 1e-4, 10000)
print(xb)

plot(iterations, curve_y_constant1)
plot(iterationsbkt, curve_y_back1, color='green', linestyle='dashed')
xlabel('iterations')
ylabel('f(xk)-f*')
legend(['Graph1'])
show()


plot(iterations, curve_y_constant2)
plot(iterationsbkt, curve_y_back2, color='green', linestyle='dashed')
xlabel('iterations')
ylabel('||∇f(xk)||')
legend(['Graph2'])
show()
