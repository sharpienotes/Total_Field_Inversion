# is the same as test_tnc but has no bounds set
# same as tnc_no_bounds only with 50 iterations

import os
import datetime
import numpy as np
import scipy
from scipy import optimize
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

#-------------------------- the actual function:------------------------------#
def chi_star(
        f=None,
        shape=None,
        d=None,
        W=None,
        M_G=None,
        chi=None,
        lambda_=np.power(10.,-3)):

    """
    Minimization using scipy.optimize.minimize() and method='BFGS'.

    The function takes input data and a guess for chi to perform
    the minimization according to the scipy.optimize.minimize() function
    with the method='BFGS'.
    It calculates all necessary parameters if not provided for the input
    of the minimization to be of the form:

    .. math:
        0.5 * ((np.linalg.norm(
            W * (f - np.fft.ifftn(d * np.fft.fftn(chi))))) ** 2 +
                    lambda_ * np.sum(abs((M_G) * (np.gradient(chi)))))

    Args:
        f (np.ndarray): Input data that is to be minimized.
            If not specified, takes it as (3,3,3) array of ones.
        shape (tuple[int]): The shape of the input data.
            By default this is (3,3,3).
        d (np.ndarray): The kernel of the respective dipole field,
            by default the dipole kernel in Fourier space according to the
            'dipole_kernel' function.
        W (np.ndarray): Data weighting term.
            By default array of ones in shape of f.
        M_G (np.ndarray): Mask of the data.
            By default an array of ones in shape of f.
        chi (np.ndarray): Term that is actively fitted/guessed.
            By default an array of ones in the shape of f.
        lambda_ (float): Numerical parameter taken according to the situation.
            By default, it is 0.001.
        time_ (bool): Boolean, gives choice if time for run should be printed.
            By default it is printed (i.e. set to True).

    Returns:

    """
    # : setting the initial values:
    if f is None:
        f = np.ones((3,3,3))
    if shape is None:
        shape = f.shape
    ones = np.ones((shape))
    if chi is None:
        chi = ones
    if W is None:
        W = ones
    if M_G is None:
        M_G = ones
    if d is None:
        d = dipole_kernel(shape=shape, origin=0.)

    def chi_star_func(chi=chi, f=f, d=d, W=W, M_G=M_G, lambda_=lambda_):
        """Calculates the input for the minimazation."""
        chi = chi.reshape(shape)
        result = 0.5 * (np.linalg.norm(
            W * (f - np.fft.ifftn(d * np.fft.fftn(chi)))) ** 2 +
                    lambda_ * np.sum(abs((M_G) * (pmu.gradient(arr=chi)))))
        return result

    # performs the minimization
    minimization = scipy.optimize.minimize(
        fun=chi_star_func,method='TNC',x0=load('/home/raid3/vonhof/Documents/Riccardo_Data/230317/phantom_32_phs.nii.gz'),
        bounds=None,
        options=dict(maxiter=1, disp=20))

    # print('Your results are: \n '+str(minimization))  # debug

    chi_arr = minimization['x'].reshape(shape)
    return chi_arr


# ======================================================================
# calling the function:
if __name__ == '__main__':
    begin_time = datetime.datetime.now()
    chi_arr = chi_star(
        f=load('/home/raid3/vonhof/Documents/Riccardo_Data/230317/phantom_32_phs.nii.gz') * 100)

    # saving the results here:
    i = 0
    template = '/home/raid3/vonhof/Documents/Riccardo_Data/230317/phantom_32_phs_TNC_nobound_oneit_{i}.nii.gz'
    filepath = template.format_map(locals())
    while os.path.isfile(filepath):
        i += 1
        filepath = template.format_map(locals())
    save(filepath ,chi_arr)

    end_time = datetime.datetime.now()
    time_elapsed = end_time - begin_time
    print('Time elapsed: {c} seconds!'.format(c=time_elapsed))

    # results for:
    #options=dict(maxiter=500, disp=500))
    # NIT   NF   F                       GTG
#     0    1  6.250653576944698E+02   3.09316951E+02
# tnc: fscale = 0.045968
# tnc: stepmx = 1000
#     1    5  4.401179716665604E+02   1.66555222E+02
#     2   14  2.333503445130736E+02   2.08673438E+01
#     3   20  2.314651422641286E+02   1.84598353E+01
#     4   26  2.071453779162987E+02   1.81406194E+01
#     4   50  2.071453779162987E+02   1.81406194E+01
# tnc: Maximum number of function evaluations reached
# Time elapsed: 15:11:06.933591 seconds!er=50, disp=50))


# options=dict(maxiter=1, disp=20))
# s.nii.gz') * 100)
#   NIT   NF   F                       GTG
#     0    1  4.389605592698246E+06   1.33324869E+06
#     0    1  4.389605592698246E+06   1.33324869E+06
# tnc: Maximum number of function evaluations reached
# Time elapsed: 0:54:41.390572 seconds!


