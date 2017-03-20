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

# definition of the place holders for data input for later:

W_proxy = np.arange(4,31).reshape(3,3,3)
data_proxy = np.arange(20,47).reshape(3,3,3) # can use sphere here instead
convolution_proxy = np.arange(27).reshape(3,3,3)
M_g_proxy = np.arange(2,29).reshape(3,3,3)
chi_proxy = np.sin(np.arange(50,77).reshape(3,3,3))
chi_prime_proxy = np.cos(np.arange(50,77).reshape(3,3,3))
d_proxy = np.arange(70,97).reshape(3,3,3)
sin_array = np.asarray([[np.sin, np.sin, np.sin],[np.sin, np.sin, np.sin],[np.sin, np.sin, np.sin]])
cos_array = np.asarray([[np.cos, np.cos, np.cos],[np.cos, np.cos, np.cos],[np.cos, np.cos, np.cos]])
#arr_sin = [np.sin for j in range(27)].reshape(3,3,3)
#arr_cos = np.array([np.cos for j in range(27)].reshape((3,3,3)))
# convolution of d and chi:
kernel = np.fft.fftn(d_proxy)
chi_fourier = np.fft.fftn(chi_proxy)
a = abs(W_proxy)
norm_part = np.linalg.norm(data_proxy)
convolution_calculated = kernel*chi_fourier
lambda_proxy = 10e-04
P_b_proxy = 30

# missing the laplace part! and the correct data
# input does not have any x to solve for!
input_for_gauss = 0.5*((np.linalg.norm(data_proxy - convolution_calculated))**2 + lambda_proxy*np.linalg.norm((M_g_proxy)*chi_proxy))
print(input_for_gauss)
#trial_chi_star = scipy.optimize.fmin_ncg(input_for_gauss,1,chi_prime_proxy)
#testing_stuff = scipy.optimize.fmin_ncg(f=sin_array, x0=np.ones((3,3,3)), fprime=cos_array)
#print('success!')
#print(arr_cos)
#print(np.cos)


arr_sin = np.asarray([np.sin for j in range(27)])
arr_sin = arr_sin.reshape((3,3,3))
#print(arr_sin.shape)

inverse_std = 1/np.std(data_proxy)
W_proxy = np.arange(4,31).reshape(3,3,3)
identity_matrix = np.asarray([np.identity(3),np.identity(3),np.identity(3)])
W_true_proxy = np.dot(identity_matrix, inverse_std)

input_for_gauss = 0.5*((np.linalg.norm(np.dot(W_true_proxy,(data_proxy - convolution_calculated))))**2 + lambda_proxy*np.sum(abs((M_g_proxy)*(np.gradient(chi_proxy)))))

def input_Gauss(data_proxy):
    return 0.5*((np.linalg.norm(np.dot(W_true_proxy,(data_proxy - convolution_calculated))))**2 + lambda_proxy*np.sum(abs((M_g_proxy)*(np.gradient(chi_proxy)))))


def input_fprime(data_proxy):
    return 12354.

# alternative without big derivative:
#other_test = scipy.optimize.fmin_tnc(func=input_Gauss,x0=0.,stepmx=25)
#other_other_test = scipy.optimize.minimize(fun=input_Gauss,method='L-BFGS-B',x0=0.)

#other_adfaother_test = scipy.optimize.minimize(fun=input_Gauss,method='BFGS',x0=0.)

#nelder = scipy.optimize.minimize(fun=input_Gauss, x0=0., args=(), method='Nelder-Mead', tol=None, callback=None, options={'disp': False, 'initial_simplex': None, 'maxiter': None, 'xatol': 0.0001, 'return_all': False, 'fatol': 0.0001, 'func': None, 'maxfev': None})

#aaddfda = scipy.optimize.minimize(fun=input_Gauss, x0=0.,method='Nelder-Mead')


import nibabel as nib
from pymrt.input_output import load, save


# put thtis into the equation to actually make up the function for input f
def abcd():
    abcd = load('/home/raid3/vonhof/Documents/Riccardo Data/1703_phantomStuff/dipole_kernel_128.nii.gz')
    return abcd

other_adfaother_test = scipy.optimize.minimize(fun=abcd,method='BFGS',x0=0.)

print(abcd.shape)
