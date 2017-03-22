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


def chi_star(
        f=None,
        shape=None,
        d=None,
        W=None,
        M_G=None,
        chi=None,
        lambda_=np.power(10.,-3),
        time=True):
    """
    The function takes input data and a guess for chi to perform
    the minimization according to the scipy.optimize.minimize() function
    with the method='BFGS'. It calculates all necessary parameters if not
    provided for the input of the minimization to be of the form:
    0.5*((np.linalg.norm(
        W * (f - d * np.fft.fftn(chi)))
                ) ** 2 +
                lambda_ * np.sum(abs((M_G) * (np.gradient(chi)))))
        
    Args:
        f (np.ndarray): Input data that is to be minimized. If not specified,
            takes it as (3,3,3) array of ones.
        shape (tuple[int]): The shape of the input data.
            By default this is (3,3,3).
        d (np.ndarray): The kernel of the respective dipole field, by default
            the dipole kernel in Fourier space according to the
            'dipole_kernel' function.
        W (np.ndarray): Data weighting term, by default array of ones in the
            shape of f.
        M_G (np.ndarray): Mask of the data, by default an array of ones in the
            shape of f.
        chi (np.ndarray): Term that is actively fitted/guessed. By default an
            array of ones in the shape of f.
        lambda_ (float): Numerical parameter taken according to the situation.
            By default, it is 0.001.
        time (bool): Boolean, gives choice if time for run should be printed.
            By default it is printed (i.e. set to True).

    Returns:
        1) The argument inside the argmin function to be minimized.
        2) A file that contains chi as an array,
            which is the solution of the minimization function (called x).
        3) If that file already exists it gives a message that it does. 
        4) The time elapsed per run is printed on screen by default.
            Set time=False if not wanted.
    """

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

    chi = chi.reshape(f.shape)

    if d is None:
        d = dipole_kernel(shape=shape, origin=0.)

    print(chi.shape)
    print(W.shape)
    print(M_G.shape)
    print(f.shape)
    print(shape)
    print(d.shape)

    def chi_star_func(chi,
        f=f,
        d=d,
        W=W,
        M_G=M_G,
        lambda_=lambda_):
        """

        Args:
            chi (np.ndarray): Parameter that is actively minimized,
                default from above.
            f (np.ndarray): Input data, default from above.
            d (np.ndarray): Dipole kernel, default from above.
            W (np.ndarray): Data weighting factor, default from above.
            M_G (np.ndarray): Mask, default from above.
            lambda_ (float):Numerical value default from above.

        Returns:
            Calculates the input for the minimazation as:
                0.5*((np.linalg.norm(
            W * (f - d * np.fft.fftn(chi)))
                    ) ** 2 +
                    lambda_ * np.sum(abs((M_G) * (np.gradient(chi)))))

        """
        return 0.5*((np.linalg.norm(
            W * (f - d * np.fft.fftn(chi)))
                    ) ** 2 +
                    lambda_ * np.sum(abs((M_G) * (np.gradient(chi)))))

    begin_time = datetime.datetime.now()

# performs minimization and writes results of minimization in an output file
    filepath = 'chi.npz'
    if not os.path.isfile(filepath):
        minimization = scipy.optimize.minimize(
            fun=chi_star_func,method='BFGS',x0=ones)

        print('Your results are: \n '+str(minimization))

        chi_arr = minimization['x']
        np.savez(filepath, chi_arr=chi_arr)
        return minimization['x']

    else:
        data = np.load(filepath)
        chi_arr = data['chi_arr']
        save(
            '/home/raid3/vonhof/Documents/TFI_phan/phantom_chi_star.nii.gz',
                chi_arr.reshape(f.shape))

        print('Your file was successfully reshaped.')

# time counter for run
    end_time = datetime.datetime.now()
    time_elapsed = end_time - begin_time
    if time is True:
        print('Time elapsed: {c} seconds!'.format(c=time_elapsed))


# calling the function:
chi_star(f=load('/home/raid3/vonhof/Documents/Riccardo Data/1703_phantomStuff/phantom_db0.nii.gz'))
