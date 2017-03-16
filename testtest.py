import math
import numpy
import scipy
from scipy import optimize


x_range = numpy.arange(6)
for x in x_range:
    f = math.sin(x)
    f_prime = math.cos(x)

#Newton = scipy.optimize.fmin_ncg(f,22,f_prime)
#,maxiter=10,retall=True)

#x = numpy.arange(start=-5,stop=5)
#variable= numpy.array([10, 9, 4, 9, 4, 1, 2, 3, 4, 5, 3, 3]).reshape(3,4)
numbers = [1,2,3,4,5]
#f_simple = variable*variable
#f_prime_simple = 2*variable
#print(callable(f_simple))


def f_callable(variable):
    for i in range(3):
        return variable[i]*variable[i]

def f_prime_callable(variable):
    for i in range(3):
        return 2*variable[i]

print(callable(f_callable(numbers)))
print(callable(f_prime_callable(numbers)))
Newton = scipy.optimize.fmin_ncg(f_callable,1,f_prime_callable)
#numpy sin

import numpy as np


W_proxy = np.arange(4,31).reshape(3,3,3)
data_proxy = np.arange(20,47).reshape(3,3,3) # can use sphere here instead
convolution_proxy = np.arange(27).reshape(3,3,3)
M_g_proxy = np.arange(2,29).reshape(3,3,3)
chi_proxy = np.arange(50,77).reshape(3,3,3)
d_proxy = np.arange(70,97).reshape(3,3,3)

# convolution of d and chi:
kernel = np.fft.fftn(d_proxy)
chi_fourier = np.fft.fftn(chi_proxy)
a = abs(W_proxy)
norm_part = np.linalg.norm(data_proxy)
print(norm_part)
convolution_calculated = kernel*chi_fourier
lambda_proxy = 10e-04
P_b_proxy = 30

input_for_gauss = 0.5*((np.linalg.norm(data_proxy - convolution_calculated))**2 + lambda_proxy*abs(M_g_proxy)*#laplace of chi_proxy)
