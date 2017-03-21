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


#data = load('/home/raid3/vonhof/Documents/Riccardo Data/1703_phantomStuff/phantom_db0.nii.gz')
def initial(
        f= None,
        shape_= None,
        d= None,
        W = None,
        M_G = None,
        chi = None,
        lambda_ = np.power(10.,-3)):
    """
    Takes data and puts it into the argmin(chi) equation.
    """

    if f == None:
        f = np.ones((3,3,3))

    if shape_ == None:
        shape_ = f.shape

    ones_ = np.ones(shape_)

    #[ones_ if param==None for param in (W,M_G,chi)]

    if W == None:
        W = ones_
    if M_G == None:
        M_G = ones_
    if chi == None:
        chi = ones_
    if d == None:
        d = dipole_kernel(shape=shape_, origin=0.)

    return 0.5*((np.linalg.norm(
        W * (f - d * np.fft.fftn(chi)))
                ) ** 2 +
                lambda_ * np.sum(abs((M_G) * (np.gradient(chi)))))


    #print([x.shape for x in (chi, f, d, M_G, W, lambda_)])
    #return([x for x in (data, shape_)])






















# x is the solution array, corresponds to chi_star in approach for GN
# creation of output file:
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


# time it took to finish the run:
end_time=datetime.datetime.now()
time_elapsed = end_time - begin_time
print('Time elapsed: {c} seconds!'.format(c=time_elapsed))
