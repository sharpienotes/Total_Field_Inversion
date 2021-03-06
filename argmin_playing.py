import datetime
import numpy as np
import scipy
from scipy import optimize
import pymrt.utils as pmu
import pymrt.geometry as pmg

from pymrt.constants import GAMMA, GAMMA_BAR

begin_time = datetime.datetime.now()


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
    dk_arr = (1.0 / 3.0 - (
        kk_z * np.cos(theta) * np.cos(phi) - kk_y * np.sin(theta) * np.cos(phi)
        + kk_x * np.sin(phi)) ** 2 / (kk_x ** 2 + kk_y ** 2 + kk_z ** 2))
    # fix singularity at |k|^2 == 0 in the denominator
    singularity = np.isnan(dk_arr)
    dk_arr[singularity] = 1.0 / 3.0
    # print('testing: '+str(kk_x)+str(kk_y)+str(kk_z))
    return dk_arr


# definition of the place holders for data input for later:
data_proxy = np.arange(20, 47).reshape(3, 3, 3)  # can use sphere here instead

inverse_std = 1 / np.std(data_proxy)
W_proxy = np.arange(4, 31).reshape(3, 3, 3)
identity_matrix = np.asarray([np.identity(3), np.identity(3), np.identity(3)])
W_true_proxy = np.dot(identity_matrix, inverse_std)

convolution_proxy = np.arange(27).reshape(3, 3, 3)
M_g_proxy = np.arange(2, 29).reshape(3, 3, 3)
chi_proxy = np.sin(np.arange(50, 77).reshape(3, 3, 3))
chi_prime_proxy = np.cos(np.arange(50, 77).reshape(3, 3, 3))
d_proxy = np.arange(70, 97).reshape(3, 3, 3)

# convolution of d and chi:
kernel = np.fft.fftn(dipole_kernel(shape=(3, 3, 3), origin=0.))
chi_fourier = np.fft.fftn(chi_proxy)
a = abs(W_proxy)
norm_part = np.linalg.norm(data_proxy)
convolution_calculated = kernel * chi_fourier
lambda_proxy = np.power(10., -3)
P_b_proxy = 30

# np.sum(abs()) corresponds to the l_1 norm
# np.linalg.norm() corresponds to the l_2 norm (Frobenius), which is then
# squared
# todo: check if zero padding is necessary (Kressler et al. p. 9)
# todo: add the inverse fourier transform of the convolution back to real space
# todo: fix convolution_calculated back to real space

input_for_gauss = 0.5 * ((np.linalg.norm(np.dot(W_true_proxy, (
    data_proxy - convolution_calculated)))) ** 2 + lambda_proxy * np.sum(
    abs((M_g_proxy) * (np.gradient(chi_proxy)))))


def input_Gauss(data_proxy):
    return 0.5 * (
        (np.linalg.norm(
            np.dot(W_true_proxy, (data_proxy - convolution_calculated)))) ** 2 +
        lambda_proxy * np.sum(abs(M_g_proxy * np.gradient(chi_proxy))))


def input_fprime(data_proxy):
    return 12354.


print(f'\n The big Gaussian input mess boils down to: {input_for_gauss}\n')
# testing = scipy.optimize.fmin_ncg(f=input_Gauss, x0=0., fprime=input_fprime)
print('very success {P_b_proxy}!'.format_map(locals()))
print('very success {x}!'.format(x=12))
print('{} success {}!'.format('very', 12))
print('{} success {c}!'.format('very', c=12))

# print(np.linalg.norm(np.dot(W_true_proxy,(data_proxy -
# convolution_calculated))))
test_2 = scipy.optimize.minimize(fun=input_Gauss, method='BFGS', x0=0.)
print(test_2)
print(test_2['success'])

end_time = datetime.datetime.now()
print('{!s}'.format(end_time - begin_time))
time_elapsed = end_time - begin_time
print('Time elapsed: {c} seconds!'.format(c=time_elapsed))
