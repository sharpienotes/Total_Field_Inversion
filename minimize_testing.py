import os
import datetime
import math
import numpy as np
import scipy
from scipy import optimize
import nibabel as nib
from pymrt.input_output import load, save
import pymrt.utils as pmu

#-------------------------- the kernel function:------------------------------#
def dipole_kernel(
        shape,
        origin=0.5,
        theta=0.0,
        phi=0.0):
    """
    Generate the 3D dipole kernel in the Fourier domain.

    .. math::

        C=\\frac{1}{3}-\\frac{k_z \\cos(\\theta)\\cos(\\phi)
        -k_y\\sin(\\theta) \\cos(\\phi)+k_x\\sin(\\phi))^2}
        {k_x^2 + k_y^2 + k_z^2}

    Args:
        shape (tuple[int]): 3D-shape of the dipole kernel array.
            If not a 3D array, the function fails.
        origin (float|tuple[float]): Relative position of the origin.
            Values are in the [0, 1] interval.
        theta (int|float): Angle of 1st rotation (along x-axis) in deg.
            Equivalent to the projection in the yz-plane of the angle between
            the main magnetic field B0 (i.e. the principal axis of the dipole
            kernel) and the z-axis. If phi is 0, the projection simplifies to
            the identity.
        phi (int|float): Angle of 2nd rotation (along y-axis) in deg.
            Equivalent to the projection in the xz-plane of the angle between
            the main magnetic field B0 (i.e. the principal axis of the dipole
            kernel) and the z-axis. If theta is 0, the projection simplifies to
            the identity.

    Returns:
        dk_arr (np.ndarray): The dipole kernel in the Fourier domain.
            Values are in the (1/3, -2/3) range.
    """
    #     / 1   (kz cos(th) cos(ph) - ky sin(th) cos(ph) + kx sin(ph))^2 \
    # C = | - - -------------------------------------------------------- |
    #     \ 3                      kx^2 + ky^2 + kz^2                    /
    # convert input angles to radians
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)
    # generate the dipole kernel
    assert (len(shape) == 3)
    kk_x, kk_y, kk_z = pmu.coord(shape, origin)
    with np.errstate(divide='ignore', invalid='ignore'):
        dk_arr = (1.0 / 3.0 - (
            kk_z * np.cos(theta) * np.cos(phi)
            - kk_y * np.sin(theta) * np.cos(phi)
            + kk_x * np.sin(phi)) ** 2 /
                  (kk_x ** 2 + kk_y ** 2 + kk_z ** 2))
    # fix singularity at |k|^2 == 0 in the denominator
    singularity = np.isnan(dk_arr)
    dk_arr[singularity] = 1.0 / 3.0
    return dk_arr
#--------------------------                      ------------------------------#

#-------------------------- place holders for data ----------------------------#
# some of them are redundant, just in place for now, to change later!
db_zero = np.arange(2097152).reshape(128, 128, 128) # can use sphere here instead
db_zero = load('/home/raid3/vonhof/Documents/Riccardo_Data/1703_phantomStuff/phantom_db0.nii.gz')

matrix_proxy = np.ones((128,128,128))
inverse_std = 1/np.std(db_zero)
W_proxy = np.arange(4,2097156).reshape(128,128,128)
identity_matrix = np.asarray([np.identity(3),np.identity(3),np.identity(3)])
W_true_proxy = np.dot(identity_matrix, inverse_std)
W_true_proxy = matrix_proxy

convolution_proxy = np.arange(2097152).reshape(128,128,128)
M_g_proxy = np.arange(2,2097154).reshape(128,128,128)
M_g_proxy = matrix_proxy
chi_proxy = np.sin(np.arange(50,2097202).reshape(128,128,128))
chi_prime_proxy = np.cos(np.arange(50,2097202).reshape(128,128,128))
d_proxy = np.arange(70,2097222).reshape(128,128,128)


# convolution of d and chi:
kernel = (dipole_kernel(shape=(128,128,128), origin=0.))
chi_fourier = np.fft.fftn(chi_proxy)
a = abs(W_proxy)
norm_part = np.linalg.norm(db_zero)
convolution_calculated = kernel*chi_fourier
lambda_proxy = np.power(10.,-3)
P_b_proxy = 30
#--------------------------                      ------------------------------#


#---------------------------- function input   --------------------------------#
def data_input(
        chi, f=db_zero, d=kernel, M_G=matrix_proxy, W=matrix_proxy,
        lambda_=lambda_proxy):
    print([x.shape for x in (chi, f, d, M_G, W, lambda_)])
    chi = chi.reshape(f.shape)
    return 0.5*((np.linalg.norm(W* (f - np.fft.ifftn(d * np.fft.fftn(chi))))) ** 2 + lambda_ * np.sum(abs((M_G) * (np.gradient(chi)))))

#---------------------------- function use   --------------------------------#
# x is the solution array, corresponds to chi_star in approach for GN
begin_time=datetime.datetime.now()
filepath = 'chi.npz'
if not os.path.isfile(filepath):
    minimization = scipy.optimize.minimize(fun=data_input,method='BFGS',x0=matrix_proxy)
    #minimization = scipy.optimize.minimize(fun=data_input,method='L-BFGS-B',x0=0.)
    print('Your results are: \n '+str(minimization))
    chi_arr = minimization['x']
    np.savez(filepath, chi_arr=chi_arr)
else:
    data = np.load(filepath)
    chi_arr = data['chi_arr']
save(
    '/home/raid3/vonhof/Documents/Riccardo Data/1703_phantomStuff/phantom_chi_star.nii.gz', chi_arr.reshape(db_zero.shape))


p_proxy = np.ones((128,128,128))*(-1.81266051)
chi_star = np.multiply(p_proxy,db_zero)
print('The result is of shape: ' +str(chi_star.shape))
print(np.sum(db_zero)/(128*128*128))
print(np.sum(chi_star)/(128*128*128))
print((np.sum(chi_star)/(128*128*128))/(np.sum(db_zero)/(128*128*128)))
print(data_input(db_zero))
end_time=datetime.datetime.now()
print('{!s}'.format(end_time - begin_time))
