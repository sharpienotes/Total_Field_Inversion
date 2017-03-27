# the gauss newton method using the ncg function of scipy optimize
# with f' as the dyn update from the appendix of tfi paper method
# cf appendix_functionTFI for the whole code on the update stuff

import datetime
import numpy as np
from pymrt.input_output import load, save
import pymrt.utils as pmu
import scipy
from scipy import optimize

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
    # todo: set f, chi and P back to what it was before testing!
    if f is None:
        f = np.ones((3,3,3))
        f = np.multiply(f,15.)
    if shape is None:
        shape = f.shape
    #print(shape)
    ones = np.ones(shape)

    if chi is None:
        chi = np.random.rand(*shape)
    if W is None:
        W = ones
    if M_G is None:
        M_G = ones
    if d is None:
        d = dipole_kernel(shape=shape, origin=0.)

    P = np.random.rand(*shape)

    # setting initial values and introducing alpha, beta, gamma for deriv_func
    epsilon = pow(10,-6)
    alpha = 1.
    beta = 1.
    gamma = 1.

    #-------------------------- derivative function  --------------------------#
    def deriv_func(d=d,W=W, P=P, lambda_=lambda_, M_G=M_G, chi=chi,
            epsilon=epsilon,f=f, alpha=alpha, beta_calc=beta, gamma=gamma):
        '''
            Computes the update for the Gauss-Newton method.

            Uses the simplification of: alpha*beta = gamma,
            where beta is the update to be computed (i.e. dy_n).

        Args:
            The ones that are not known from above are:
            P (): the R2* threshold map
            epsilon (): a small number, arbitrary
        Returns:
            beta, the update to be computed
        '''
        #todo: put y wherever gradP appears?
        #print(P.shape, d.shape, W.shape)
        a = P * np.fft.ifftn(d * np.fft.fftn(W * W))* \
            np.fft.ifftn(d * np.fft.fftn(P))

        b = lambda_ * P * pmu.inv_gradient(M_G)
        c = 1/np.sqrt(np.abs(M_G * pmu.gradient(chi)) **2 + epsilon)
        d = M_G * pmu.gradient(P)
        alpha = a + b * c * d
        e = P * np.fft.ifftn(d * np.fft.fftn(W * W))
        g = np.fft.ifftn(d * np.fft.fftn(chi))
        h = lambda_ * P * pmu.inv_gradient(M_G)
        j = M_G * pmu.gradient(chi)
        gamma = e * (f - g) - h * c * j

        # computation of beta (element-wise division of ndarrays)
        beta_calc = np.divide(gamma, alpha)
        return beta_calc

    #---------------------- ###################  ------------------------------#

    # calling the update computation function:
    #alpha, beta_calc, gamma = deriv_func(alpha=alpha, beta_calc=beta, gamma=gamma)
    #print('The update is the following: \n {b} '.format(b=beta))

    #---------------------- ###################  ------------------------------#
    def chi_star_func(chi=chi, f=f, d=d, W=W, M_G=M_G, lambda_=lambda_):
        """Calculates the input for the minimazation."""
        chi = chi.reshape(shape)
        result = 0.5 * (np.linalg.norm(
            W * (f - np.fft.ifftn(d * np.fft.fftn(chi)))) ** 2 +
                    lambda_ * np.sum(abs((M_G) * (np.gradient(chi)))))
        return result

    def calculation(chi_star_func,deriv_func, beta_calc=beta):
        result = chi_star_func(chi=chi, f=f, d=d, W=W, M_G=M_G, lambda_=lambda_)
        beta_calc = deriv_func(d=d,W=W, P=P, lambda_=lambda_, M_G=M_G, chi=chi,
            epsilon=epsilon,f=f, alpha=alpha, beta_calc=beta, gamma=gamma)

        chi_res = scipy.optimize.fmin_ncg(f=chi_star_func, x0=chi, fprime=deriv_func)
        # xopt is zeroth term
        return chi_res[0]

    chi_res = calculation(chi_star_func,deriv_func, beta_calc=beta)
    return chi_res
# =============================================================================
if __name__ == '__main__':
    begin_time = datetime.datetime.now()

    # execute whole function (with two sub-functions)
    chi_res = chi_star(f=load('/home/raid3/vonhof/Documents/Riccardo_Data/230317/phantom_32_phs.nii.gz'), chi=load('/home/raid3/vonhof/Documents/Riccardo_Data/230317/phantom_32_phs.nii.gz'))

    save('/home/raid3/vonhof/Documents/Riccard_Data/230317/phantom_32_chi_ncg_app.nii.gz', chi_res)
# todo: save is not the main problem, rest also has a bug somewhere
    end_time = datetime.datetime.now()
    time_elapsed = end_time - begin_time
    print('Time elapsed: {c} seconds.'.format(c=time_elapsed))

#todo: solve reshape chi thing for wrong shape of stuff
# todo: make it work at all


#chi_res = chi_star(f=load('/home/raid3/vonhof/Documents/Riccardo_Data/230317/phantom_32_phs.nii.gz'), chi=load('/home/raid3/vonhof/Documents/Riccardo_Data/230317/phantom_32_phs.nii.gz'))
