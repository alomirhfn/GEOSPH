from pysph.sph.equation import Equation
from compyle.api import declare
from math import (pi, fabs, sqrt, log, pow, exp, tan, sin, cos, acos, isnan,
                  atan)

from matrix_operations import (matrix_multiply, matrix_transpose, sign,
                               flat_to_voight, voight_to_flat,
                               matrix_multiply_vector,
                               eig_vals_3x3_analytical_erick,
                               matrix_eigenvectors_erick, vector_normalize,
                               matrix_eigenvectors, eig_vals_3x3_analytical,
                               matrix_zero)

from CASM.casm_model import casm_model
from CASM.casm_model import casm_model_EL
from CASM.pegasus_file import pegasus_alg
from CASM.correction import correction
from CASM.pegasus_file import pegasus_unload

# ============================= Helper Functions ==============================

def yield_criterion_dp(q=0.0, p=0.0, aphi=0.0, ac=0.0, c=0.0):
    """
    Calculates the yield function value.

    If the value calculated is greater than zero, it means the material is
     yielding at the particle. It only implements one type of yield function
     called generalized Von Mises. If aphi is zero, returns the Von Mises yield
     criterion, otherwise, returns the Drucker-Prager criterion.

    Parameters
    ----------
    :param p:
    :param q:
    :param aphi:
    :param ac:
    :param c:

    Output
    -----------
    :return:
    """
    return sqrt(2.0/3.0) * q + aphi * p - ac * c

def yield_criterion_mcc(q=0.0, p=0.0, pc=0.0, ms=0.0):
    """
    Calculates the yield function value.

    If the value calculated is greater than zero, it means the material is
     yielding at the particle. It only implements one type of yield function
     called Modified Cam-Clay (MCC).

    Parameters
    ----------
    :param p:
    :param q:
    :param pc:
    :param ms:

    Output
    -----------
    :return: y = numerical result of yield function
    """
    return (q * q) / (ms * ms) + p * (p - pc)

def yield_criterion_mohrc(q=0.0, p=0.0, phi=0.0, c=0.0, theta=0.0):
    r"""
    Calculates the Mohr-Coulomb yield function value. This implementation was
    taken from the paper of Zhao et al (2019) A generic approach to modelling
    flexible confined boundary conditions in SPH and its application.
    (https://onlinelibrary.wiley.com/doi/10.1002/nag.2918)

    @param q: Von Mises stress
    @param p: Mean normal stress
    @param phi: Mohr-Coulomb internal friction angle (in radians)
    @param c: Mohr-Coulomb cohesion
    @param theta: Lode's angle
    @return: y = value of the yield function
    """
    return (3 * sin(phi) * p + 0.5 * (3 * (1 - sin(phi)) * sin(theta) +
                                     sqrt(3) * (3 + sin(phi)) * cos(theta)) *
            (q / sqrt(3)) - 3 * c * cos(phi))

# ===========================================================================


class DruckerPragerSolverExact(Equation):

    def __init__(self, dest, sources, c_model=1):
        self.c_model = c_model
        super(DruckerPragerSolverExact, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_q, d_sigma, d_sigma_tr, d_sigma_dev,
                   d_sigma_dot, d_eps_dot, d_spin_dot, d_eps_p_dot, d_aphi,
                   d_apsi, d_ac, d_bulk, d_shear, d_cohesion, d_flag, dt):

        i, idx = declare("int", 2)
        sigma, sig_dot, sig_spin, sig_spin_t = declare("matrix(9)", 4)
        spin_dot = declare("matrix(9)")

        # Index used to access values in arrays
        idx = 9 * d_idx

        # Flag particle to use trial values
        d_flag[d_idx] = 0

        # Initialize stress, stress rate, and plastic rate tensors
        for i in range(9):
            sigma[i] = d_sigma[idx + i]
            sig_dot[i] = d_sigma_dot[idx + i]
            spin_dot[i] = d_spin_dot[idx + i]
            d_eps_p_dot[idx + i] = 0.0
            sig_spin[i] = 0.0
            sig_spin_t[i] = 0.0

        # If not elastic material
        if self.c_model > 0:

            p = d_p[d_idx]
            q = d_q[d_idx]
            aphi = d_aphi[d_idx]
            apsi = d_apsi[d_idx]
            ac = d_ac[d_idx]
            c = d_cohesion[d_idx]

            # Check for yielding
            y = yield_criterion_dp(q, p, aphi, ac, c)

            # If yielding
            if y > 1e-9:

                # Elastic constants
                k = d_bulk[d_idx]
                g = d_shear[d_idx]

                # Plastic multiplier
                dgamma = y / (2 * g + k * aphi * apsi)

                # ============ Check if return to the cone is valid ===========
                norm_s = sqrt(2.0 / 3.0) * q
                if norm_s - 2 * g * dgamma >= 0.0:

                    # Pre-compute some constants
                    n_d = 2 * g * dgamma / norm_s
                    p_i = k * apsi * dgamma
                    dg_dot = dgamma / dt
                    g_d = dg_dot / norm_s
                    g_v = dg_dot * apsi / 3.0

                    # Update stress and plastic strain rate
                    for i in range(9):
                        sigma[i] = (
                                d_sigma_tr[idx + i] -
                                n_d * d_sigma_dev[idx + i]
                        )
                        d_eps_p_dot[idx + i] = (
                            g_d * d_sigma_dev[idx + i]
                        )

                        if i % 4 == 0:
                            sigma[i] -= p_i
                            d_eps_p_dot[idx + i] += g_v

                # =============== Return to apex or stress-free ===============
                else:
                    # Hack: Cap tensile strength at 0.0 instead of ac*c/aphi.
                    # This instantly relieves tensile stresses driving
                    # particles apart.
                    p_apex = 0.0
                    # p_apex = ac * c / aphi

                    # Only generate volumetric plastic strain if psi allows it
                    # AND we are shifting p
                    if apsi > 1e-9:
                        deps_p_vol = (p - p_apex) / k
                        deps_p_vol_dot = deps_p_vol / (3.0 * dt)
                    else:
                        # Strictly incompressible flow for psi = 0
                        deps_p_vol_dot = 0.0

                    # Update stress and strain rates
                    for i in range(9):
                        sigma[i] = 0.0

                        # Dev plastic strain completely relaxes the trial
                        # dev stress
                        d_eps_p_dot[idx + i] = (
                                d_sigma_dev[idx + i] / (2.0 * g * dt)
                        )

                        if i % 4 == 0:
                            sigma[i] += p_apex
                            d_eps_p_dot[idx + i] += deps_p_vol_dot

                # Flag particle to update stress and strain
                d_flag[d_idx] = 1

        # Jaumann stress rate (~ large deformation)
        matrix_multiply(sigma, spin_dot, sig_spin, 3)
        matrix_multiply(spin_dot, sigma, sig_spin_t, 3)

        if d_flag[d_idx] == 0:
            for i in range(9):
                sigma[i] = d_sigma_tr[idx + i]

        for i in range(9):
            sig_dot[i] = (sigma[i] - d_sigma[idx + i]) / dt
            d_sigma_dot[idx + i] = sig_dot[i] - sig_spin[i] + sig_spin_t[i]

    def _get_helpers_(self):
        return[yield_criterion_dp, matrix_multiply]


class MohrCoulombSolverBui(Equation):

    def __init__(self, dest, sources, c_model=1, debug=0):
        self.c_model = c_model
        self.debug = debug
        super(MohrCoulombSolverBui, self).__init__(dest, sources)

    def initialize(self, d_idx, d_sigma, d_sigma_tr, d_sigma_dev, d_sigma_dot,
                   d_p, d_q, d_eps_dot, d_eps_p, d_eps_p_dot, d_ep_acc,
                   d_ep_eff, d_spin_dot, d_phi, d_psi, d_cohesion, d_bulk,
                   d_shear, d_flag, d_gid, dt):

        i, idx = declare("int", 2)
        sigma, sdev, sig_dot, sig_spin, sig_spin_t = declare("matrix(9)", 5)
        tt, ss, spin_dot = declare("matrix(9)", 3)

        # Index used to access values in arrays
        idx = 9 * d_idx

        # Flag particle to use trial values
        d_flag[d_idx] = 0

        # Initialize stress and stress rate tensors
        for i in range(9):
            sigma[i] = d_sigma_tr[idx + i]
            sig_dot[i] = d_sigma_dot[idx + i]
            spin_dot[i] = d_spin_dot[idx + i]
            sdev[i] = d_sigma_dev[idx + i]

        # Calculate third stress deviation invariant
        matrix_multiply(sdev, sdev, ss, 3)

        j3 = 0.0
        for i in range(9):
            j3 += sdev[i] * ss[i]
        j3 /= 3.0

        # Second deviatoric stress invariant
        j2 = pow(d_q[d_idx] / sqrt(3), 2)

        # Lode's angle
        theta = pi / 6.0
        if j2 > 1e-6:  # For particles in a hydrostatic stress state
            cos_3t = 1.5 * sqrt(3) * j3 / pow(sqrt(j2), 3)

            # Necessary due to float point precision where abs(cos(3theta)) can
            #  be greater than one due to round-off error.
            if abs(cos_3t) > 1.0:
                cos_3t = sign(cos_3t)
            theta = acos(cos_3t) / 3.0

        if isnan(theta):
            theta = pi / 6.0
            printf("\n ISSUE WITH LODE'S ANGLE IN MC CONSTITUTIVE UPDATE!\n")
            printf("Issue with particle: %d\n\n", d_gid[d_idx])

        # TODO: Consider storing these values to simply call them from previous
        #  time step
        # MC material parameters
        phi = d_phi[d_idx] * pi / 180
        psi = d_psi[d_idx] * pi / 180
        c = d_cohesion[d_idx]

        # Elastic constants
        k = d_bulk[d_idx]
        g = d_shear[d_idx]

        # If not elastic material
        if self.c_model > 0:

            # ================== ACCUMULATED PLASTIC STRAINS ==================

            # Calculate accumulated total and deviatoric plastic strains, and
            #  initialize temp Cauchy stress
            ep_vol = (d_eps_p[idx] + d_eps_p[idx + 4] + d_eps_p[idx + 8]) / 3.0
            ep_dev2 = 0.0
            ep_acc2 = 0.0
            for i in range(9):

                # Accumulated total and deviatoric plastic strains
                ep = d_eps_p[idx + i]
                ep_acc2 += ep * ep
                if i % 4 == 0:
                    ep -= ep_vol
                ep_dev2 += ep * ep

            ep_acc = sqrt(ep_acc2)
            ep_eff = sqrt(1.0 * ep_dev2 / 2.0)
            d_ep_eff[d_idx] = ep_eff
            d_ep_acc[d_idx] = ep_acc

            # =========================== SOFTENING ===========================

            # Softening coefficients (set to zero for no softening)
            eta_phi = 0.0
            eta_psi = 0.0
            eta_c = 0.0

            # Peak and residual values
            phi_p = d_phi[d_idx]
            phi_res = 15.0
            psi_p = d_psi[d_idx]
            c_p = d_cohesion[d_idx]
            c_res = 100.0

            # Calculate new values of material parameters
            phi = ((phi_res + (phi_p - phi_res) * exp(-eta_phi * ep_eff)) *
                   pi / 180.0)
            psi = psi_p * exp(-eta_psi * ep_acc) * pi / 180.0
            c = c_res + (c_p - c_res) * exp(-eta_c * ep_eff)

            # =================================================================

            # Mean normal stress and Von Mises stress
            p = d_p[d_idx]
            q = d_q[d_idx]

            # Check for yielding
            y = yield_criterion_mohrc(q, p, phi, c, theta)

            # If yielding
            if y > 1e-6:

                # Plastic multiplier rate
                dg_dot = 0.0

                # Volumetric strain rate
                eps_dot_v = (d_eps_dot[idx] + d_eps_dot[idx + 4] +
                             d_eps_dot[idx + 8])

                if abs(3 * sqrt(3) * j3 / (2 * pow(j2, 1.5))) < 1:

                    # Yield function and plastic potential derivatives w.r.t.
                    #  the stress tensor (dF/dSigma, dG/dSigma)
                    dfdi1 = sin(phi)

                    dfdj2 = ((3 * (1 - sin(phi)) * sin(theta) +
                              sqrt(3) * (3 + sin(phi)) * cos(theta)) /
                             (4 * sqrt(j2)) +
                             (3 * (1 - sin(phi)) * cos(theta) -
                              sqrt(3) * (3 + sin(phi)) * sin(theta)) *
                             3 * sqrt(3) * j3 / (8 * pow(j2, 2) *
                                                 sin(3 * theta)))

                    dfdj3 = (-(3 * (1 - sin(phi)) * cos(theta) -
                               sqrt(3) * (3 + sin(phi)) * sin(theta)) *
                             sqrt(3) / (4 * j2 * sin(3 * theta)))

                    dgdi1 = sin(psi)

                    dgdj2 = ((3 * (1 - sin(psi)) * sin(theta) +
                              sqrt(3) * (3 + sin(psi)) * cos(theta)) /
                             (4 * sqrt(j2)) +
                             (3 * (1 - sin(psi)) * cos(theta) -
                              sqrt(3) * (3 + sin(psi)) * sin(theta)) *
                             3 * sqrt(3) * j3 / (8 * pow(j2, 2) *
                                                 sin(3 * theta)))

                    dgdj3 = (-(3 * (1 - sin(psi)) * cos(theta) -
                               sqrt(3) * (3 + sin(psi)) * sin(theta)) *
                             sqrt(3) / (4 * j2 * sin(3 * theta)))

                    # H denominator of the plastic multiplier rate
                    ss2 = 0.0
                    for i in range(9):
                        ss2 += ss[i] * ss[i]

                    h = (9 * k * dfdi1 * dgdi1 + 4 * g * j2 * dfdj2 * dgdj2 +
                         6 * g * j3 * (dfdj2 * dgdj3 + dgdj2 * dfdj3) +
                         2 * g * (ss2 - 4 * pow(j2,2) / 3) * dfdj3 * dgdj3)

                    # Auxiliary term t
                    pt = 2 * j2 / 3
                    for i in range(9):
                        pt_i = 0.0
                        if i % 4 == 0:
                            pt_i = pt
                        tt[i] = ss[i] - pt_i

                    # Plastic multiplier rate
                    dg_dot = 3 * k * dfdi1 * eps_dot_v
                    for i in range(9):
                        dg_dot += (2 * g * (dfdj2 * sdev[i] + dfdj3 * tt[i]) *
                                   d_eps_dot[idx + i])

                    dg_dot /= h

                    # Calculate new sigma and sigma rate
                    p_corr = 3 * k * sin(psi)
                    epd = 2 * g * dgdj2
                    ept = 2 * g * dgdj3
                    for i in range(9):
                        sig = epd * sdev[i] + ept * tt[i]
                        if i % 4 == 0:
                            sig += p_corr
                        sig_dot[i] -= dg_dot * sig

                        # Small-strain updated stress
                        sigma[i] = d_sigma[idx + i] + sig_dot[i] * dt

                else:  # Singularity return

                    # New Mohr-Coulomb parameters
                    aphi_den = (3 * (1 - sin(phi)) * sin(theta) +
                                sqrt(3) * (3 + sin(phi)) * cos(theta))

                    apsi_den = (3 * (1 - sin(psi)) * sin(theta) +
                                sqrt(3) * (3 + sin(psi)) * cos(theta))

                    aphi = 2 * sin(phi) / aphi_den
                    apsi = 2 * sin(psi) / apsi_den
                    aphic = 6 * cos(phi) / aphi_den
                    apsic = 6 * cos(psi) / apsi_den

                    # Auxiliary pre-computed values
                    sqj2 = q / sqrt(3.0)  # Square-root of J2
                    epd = g / sqj2  # Equivalent deviatoric strain

                    # Plastic multiplier rate
                    dg_dot = 3 * aphi * k * eps_dot_v
                    for i in range(9):
                        dg_dot += (epd * sdev[i] * d_eps_dot[idx + i])

                    dg_dot /= (9 * aphi * apsi * k + g)

                    # Pre-compute mean normal stress correction
                    p_corr = 3 * k * apsi

                    # Update stress and stress rate
                    for i in range(9):

                        # Stress rate
                        sig = epd * sdev[i]
                        if i % 4 == 0:
                            sig += p_corr
                        sig_dot[i] -= dg_dot * sig

                # Flag particle to update stress and strain
                d_flag[d_idx] = 1

        # Jaumann stress rate (~ large deformation)
        matrix_multiply(sigma, spin_dot, sig_spin, 3)
        matrix_multiply(spin_dot, sigma, sig_spin_t, 3)

        for i in range(9):
            d_sigma_dot[idx + i] = sig_dot[i] - sig_spin[i] + sig_spin_t[i]

        if d_flag[d_idx] > 0:

            # ======================= BUI'S CORRECTIONS =======================

            # Updated stress
            for i in range(9):
                sigma[i] = d_sigma[idx + i] + d_sigma_dot[idx + i] * dt

            # Update mean normal stress
            p = (sigma[0] + sigma[4] + sigma[8]) / 3.0

            # Update deviatoric stress
            for i in range(9):
                sdev[i] = sigma[i]
                if i % 4 == 0:
                    sdev[i] -= p

            # Update J2 and J3
            matrix_multiply(sdev, sdev, ss, 3)

            j2 = 0.0
            j3 = 0.0
            for i in range(9):
                j2 += pow(sdev[i], 2)
                j3 += sdev[i] * ss[i]
            j3 /= 3.0
            j2 /= 2.0

            # Update Lode's angle
            if j2 > 1e-6:  # For particles in a hydrostatic stress state
                cos_3t = 1.5 * sqrt(3) * j3 / pow(sqrt(j2), 3)

                # Necessary due to float point precision where abs(cos(3theta))
                # can be greater than one due to round-off error.
                if abs(cos_3t) > 1.0:
                    cos_3t = sign(cos_3t)
                theta = acos(cos_3t) / 3.0

            if isnan(theta):
                theta = pi / 6.0
                printf("ISSUE WITH LODE'S ANGLE IN MC CONSTITUTIVE UPDATE!\n")
                printf("Issue with particle: %d\n\n", d_gid[d_idx])

            # Update MC parameters
            aphi_den = (3 * (1 - sin(phi)) * sin(theta) +
                        sqrt(3) * (3 + sin(phi)) * cos(theta))

            aphi = 2 * sin(phi) / aphi_den
            aphic = 6 * cos(phi) / aphi_den

            # First stress invariant for tension cracking correction
            i1n = 3 * p

            # Check if return to the lateral yield surface is valid
            if aphic * c - aphi * i1n < 0.0:  # Not valid!

                for i in range(0, 9, 4):
                    d_sigma_dot[idx + i] -= ((i1n - aphic * c / aphi) /
                                             (3.0 * dt))

            # Return to the linear part of the DP yield surface is valid
            else:

                sqj2 = sqrt(j2)

                # Correction factor for deviations from the cone
                rn = (aphic * c - aphi * i1n) / sqj2

                if abs(rn - 1.0) > 1e-6:
                    for i in range(9):
                        pn = 0.0
                        if i % 4 == 0:
                            pn = i1n / 3.0

                        d_sigma_dot[idx + i] = (rn * sdev[i] + pn -
                                                d_sigma[idx + i]) / dt

            # ===================== UPDATE PLASTIC STRAIN =====================

            # Mean normal stress rate
            p_dot = (d_sigma_dot[idx] + d_sigma_dot[idx + 4] +
                     d_sigma_dot[idx + 8]) / 3.0

            # Volumetric elastic strain rate
            eps_e_dot_v = p_dot / k

            # Elastic strain calculations
            for i in range(9):
                p = 0.0
                eps_e_dot_h = 0.0

                # Hydrostatic components
                if i % 4 == 0:
                    p = p_dot
                    eps_e_dot_h = eps_e_dot_v / 3.0

                # Correct elastic strain
                eps_e_dot = eps_e_dot_h + (d_sigma_dot[idx + i] - p) / (2 * g)

                # Updated plastic strain
                d_eps_p_dot[idx + i] = d_eps_dot[idx + i] - eps_e_dot

    def _get_helpers_(self):
        return [yield_criterion_mohrc, matrix_multiply, sign]


class ModifiedCamClay(Equation):
    def __init__(self, dest, sources, nu=0.3, c_model=1, tol=1e-5,
                 max_iter=100):
        self.nu = nu
        self.c_model = c_model
        self.tol = tol
        self.max_iter = max_iter
        super(ModifiedCamClay, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_q, d_sigma, d_sigma_tr, d_sigma_dev,
                   d_void_ratio, d_void_ref, d_sigma_dot, d_eps_dot,
                   d_spin_dot, d_eps_p_dot, d_ep_acc, d_pc, d_lambda_mcc,
                   d_kappa_mcc, d_ms, d_bulk, d_shear, d_flag, dt):

        i, idx, count = declare("int", 3)
        sigma, sig_dot, sig_spin, sig_spin_t = declare("matrix(9)", 4)
        spin_dot = declare("matrix(9)")

        # Index used to access values in arrays
        idx = 9 * d_idx

        # Flag particle to use trial values
        d_flag[d_idx] = 0

        # Consolidation parameters
        lambda_mcc = d_lambda_mcc[d_idx]
        kappa_mcc = d_kappa_mcc[d_idx]
        void_ratio = d_void_ratio[d_idx]
        void_ref = d_void_ref[d_idx]
        v = (1 + void_ratio) / (lambda_mcc - kappa_mcc)

        # Initialize stress and stress rate tensors
        for i in range(9):
            sigma[i] = d_sigma_tr[idx + i]
            sig_dot[i] = d_sigma_dot[idx + i]
            sig_spin[i] = 0.0
            sig_spin_t[i] = 0.0

        # If not elastic material
        if self.c_model > 0:
            p = d_p[d_idx]
            q = d_q[d_idx]
            pc = d_pc[d_idx]
            pcn = pc
            ms = d_ms[d_idx]

            # Initialize temporary vectors
            for i in range(9):
                sigma[i] = d_sigma[idx + i]
                spin_dot[i] = d_spin_dot[idx + i]

            # Check for yielding
            y = yield_criterion_mcc(q, p, pc, ms)

            # If yielding
            if y > 1e-8:

                # Trial hydrostatic stress and von Mises stress
                ptr = p
                qtr = q

                # Elastic constants
                k = d_bulk[d_idx]
                g = d_shear[d_idx]

                # Counter of iterations for NR
                count = 0

                # Initialize consistency parameter
                dg = 0.0

                # Newton-Raphson loop
                while y > self.tol and count < self.max_iter:

                    # Pre-calculate the partial derivatives of the yield
                    # surface w.r.t. the consistency parameter, dg (DeltaL_k
                    # in ConsUpdate.m Enrique Approximate CPP)
                    a = (-2 * k * pow(dg * (2 * p - pc) + 1 / v, 2) *
                         (p - pc / 2)) / (8 * k * pow(p - pc / 2, 2) *
                                          pow(dg, 3) + 8 *
                                          (k * (1 / v) + p / 2 - pc / 4) *
                                          (p - pc / 2) * pow(dg, 2) + 2 *
                                          (1 / v) * (k * (1 / v) + 2 * p -
                                                     pc - pcn / 2) * dg +
                                          pow(1 / v, 2))

                    b1 = -q / (dg + pow(ms, 2) / (6 * g))

                    c1 = (((-2 * p + pc) * (1 / v) * pcn) /
                          (8 * k * pow(p - pc / 2, 2) * pow(dg, 3) + 8 *
                          (k * (1 / v) + p / 2 - pc / 4) * (p - pc / 2) *
                          pow(dg, 2) + 2 * (1 / v) * (k * (1 / v) + 2 * p -
                                                      pc - pcn / 2) * dg +
                          pow(1 / v, 2)))

                    # Derivative of the yield function (Fp_k)
                    dy = ((2 * p - pc) * a + (2 * q / pow(ms, 2)) * b1 -
                          p * c1)

                    # Update the consistency parameter
                    dg -= y / dy

                    # Auxiliary values to find pc
                    b2 = (1 / v) * (1 + 2 * k * dg) + 2 * dg * ptr
                    c2 = pcn * (1 / v) * (1 + 2 * k * dg)
                    d1 = b2 / dg
                    d2 = sqrt(pow(b2 / dg, 2) - (4 * c2) / dg)

                    # Roots of second order polynomial used to determine pc
                    pc_a = 0.5 * (d1 + d2)
                    pc_b = 0.5 * (d1 - d2)

                    # Update pc
                    if pc_a < 0:
                        pc = pc_a
                    else:
                        pc = pc_b

                    # Update p and q
                    p = (ptr + k * dg * pc) / (1 + 2 * k * dg)
                    q = qtr / (1 + 6 * g * dg / pow(ms, 2))

                    # Calculate if stress on the yield surface
                    y = yield_criterion_mcc(q, p, pc, ms)

                    # Increment iteration counter
                    count += 1

                # Check if convergence obtained
                if y > self.tol:
                    printf("\n")
                    printf("====== NR DID NOT CONVERGE AFTER ")
                    printf("%d", count)
                    printf("ITERATIONS ======")
                    printf('Residual F')
                    printf("%.16f\n", y)
                    printf('\n')

                d_pc[d_idx] = pc

                # Rate of plastic multiplier
                dg_dot = dg / dt

                # Volumetric strain rate
                eps_dot_v = d_eps_dot[idx] + d_eps_dot[idx + 4] + \
                            d_eps_dot[idx + 8]

                # Construct deformation tensors
                norm_s = sqrt(2.0 / 3.0) * qtr
                for i in range(9):

                    if norm_s > 0:
                        # Deviatoric plastic flow directions
                        n_i = d_sigma_dev[idx + i] / norm_s
                    else:
                        n_i = 0

                    # Total strain rate decomposition
                    eps_dot_d = d_eps_dot[idx + i]
                    if i % 4 == 0:
                        eps_dot_d -= eps_dot_v / 3.0

                    # Plastic strain rates
                    eps_p_dot_d = sqrt(6) * dg_dot * q * n_i / (ms * ms)
                    eps_p_dot_v = dg_dot * (2 * p - pc)
                    eps_p_dot_h = 0.0
                    if i % 4 == 0:
                        eps_p_dot_h = eps_p_dot_v / 3.0
                    d_eps_p_dot[idx + i] = eps_p_dot_d + eps_p_dot_h

                    # Stress rate
                    eps_e_dot_d = eps_dot_d - eps_p_dot_d
                    eps_e_dot_v = eps_dot_v - eps_p_dot_v
                    p_i = 0.0
                    if i % 4 == 0:
                        p_i = k * eps_e_dot_v
                    sdot = p_i + 2 * g * eps_e_dot_d

                    # Small-strain stress rate
                    sig_dot[i] = sdot

                    # Small-strain updated stress
                    sigma[i] += sdot * dt

                # Update accumulated plastic strain and uniaxial yield stress
                d_ep_acc[d_idx] += dg

                # Flag particle to update stress and strain
                d_flag[d_idx] = 1

        if d_flag[d_idx] == 0:
            matrix_multiply(sigma, spin_dot, sig_spin, 3)
            matrix_multiply(spin_dot, sigma, sig_spin_t, 3)

        for i in range(9):
            d_sigma_dot[idx + i] = sig_dot[i] - sig_spin[i] + sig_spin_t[i]

        # Update stress invariants
        # TODO: Find out how to do this in the output rather than every time
        #   step here!!!
        p_dot = (d_sigma_dot[idx] + d_sigma_dot[idx + 4] +
                 d_sigma_dot[idx + 8]) / 3.0
        p_n = (d_sigma[idx] + d_sigma[idx + 4] + d_sigma[idx + 8]) / 3.0
        ph = p_n + p_dot * dt
        d_p[d_idx] = ph
        p = d_p[d_idx]

        norm_s2 = 0.0
        for i in range(9):
            s_i = d_sigma[idx + i] + d_sigma_dot[idx + i] * dt
            if i % 4 == 0:
                s_i -= ph
            norm_s2 += s_i * s_i
        d_q[d_idx] = sqrt(3 * norm_s2 / 2)

        # Update bulk and shear moduli
        if p != 0:
            ph = ph / 1000
            pc = pc / 1000
            void_ratio = (
                    void_ref - lambda_mcc * log(-pc) + kappa_mcc * log(pc / ph)
            )
            k = abs((1 + void_ratio) * p / kappa_mcc)
            g = 3 * k * (1 - 2 * self.nu) / (2 * (1 + self.nu))
            d_bulk[d_idx] = k
            d_void_ratio[d_idx] = void_ratio
            d_shear[d_idx] = g

    def _get_helpers_(self):
        return [yield_criterion_mcc, matrix_multiply]


class CASM(Equation):

    def __init__(self, dest, sources, c_model=1, bulkw=2e9, debug=0):
        self.c_model = c_model
        self.debug = debug
        self.bulkw = bulkw
        super(CASM, self).__init__(dest, sources)

    def initialize(self, d_idx, d_sigma_tr, d_sigma_dot, d_spin_dot,
                   d_eps_p_dot, d_flag, d_sigma, d_bulk, d_shear, dt,
                   d_eps_dot, d_p_kappa, d_p_nu, d_p_gamma, d_p_lambda, d_p_n,
                   d_p_r, d_p_m, d_p_phi, d_p_p1, d_p_ratiops, d_p_lode,
                   d_p_nc1max, d_p_stol, d_p_ftol, d_p_maxitsc, d_p_maxitsp,
                   d_p_maxitspu, d_p_nsub, d_mtc, d_a, d_pw, d_s_undrained,
                   d_s_ocr, d_s_plast_flag, d_s_e, d_s_y, d_s_p0, d_s_ev,
                   d_s_v, d_s_q, d_s_p, d_s_eta, d_s_teta, d_s_m_teta,
                   d_s_eta_m_teta, d_s_v0, d_s_yield, d_s_bulk, d_s_shear,
                   d_s_mc, d_p, d_q, d_aphi, d_apsi, d_ac, d_sy, d_eps_p,
                   d_ep_eff, d_ep_acc, d_psi, d_sigma_dev, d_h_mod, d_phi):

        # -----------------------------------------------------------------
        # Declarando variaveis internas
        # -----------------------------------------------------------------

        i, idx = declare("int", 2)
        sigma, sig_dot, sig_spin, sig_spin_t = declare("matrix(9)", 4)
        spin_dot = declare("matrix(9)")

        eigvals = declare("matrix(3)")
        eigvals0 = declare("matrix(3)")
        props = declare("matrix(21)")  # vetor com propriedades do modelos
        # constitutivo aplicado na partícula
        stVar0 = declare("matrix(18)")  # vetor com parâmetros de estado
        # iniciais
        stress0 = declare("matrix(6)")  # tensões iniciais representadas num
        # vetor de 6 posições
        stress0_flat = declare("matrix(9)")  # tensões iniciais representadas
        # num vetor de 9 posições
        stVarN = declare("matrix(18)")  # vetor com novos parâmetros de estado
        stressN_flat = declare("matrix(9)")  # tensões atualizadas
        # representadas num vetor de 9 posições
        stressN = declare("matrix(6)")  # tensões atualizadas representadas num
        # vetor de 6 posições
        stVarN0 = declare("matrix(18)")  # vetor com parâmetros de estado
        # iniciais apos transicao elastica (pegasus)
        stressN0_flat = declare("matrix(9)")  # tensões iniciais após transição
        # elástica (Pegasus)
        stressN0 = declare("matrix(6)")
        stVarN1 = declare("matrix(18)")  # usado no algoritmo de retorno
        stressN1_flat = declare("matrix(9)")  # usado no algoritmo de retorno
        stressN1 = declare("matrix(6)")
        stVarN2 = declare("matrix(18)")  # usado no algoritmo de retorno
        stressN2_flat = declare("matrix(9)")  # usado no algoritmo de retorno
        stressN2 = declare("matrix(6)")
        eps_flat = declare("matrix(9)")  # incremento de deformação
        # representada num vetor de 9 posições, com notaçãode engenharia
        # (gxy = 2*exy)
        eps_flat2 = declare("matrix(9)")  # incremento de deformação
        # representada num vetor de 9 posições
        eps = declare("matrix(6)")  # incremento de deformação representada num
        # vetor de 6 posições
        epsN = declare("matrix(6)")  # incremento de deformação usado no
        # algoritmo de retorno
        epsN_flat = declare("matrix(9)")  # incremento de deformação usado no
        # algoritmo de retorno
        eps_EL = declare("matrix(6)")  # incremento de deformação elástica
        # representada num vetor de 6 posições
        eps_dot_p = declare("matrix(6)")  # incremento de deformação elástica
        # representada num vetor de 6 posições
        eps_dot_p_flat = declare("matrix(9)")  # incremento de deformação
        # elástica representada num vetor de 9 posições
        stress_inc = declare("matrix(6)")  # icremento de tensões representadas
        # num vetor de 6 posições
        stress_inc_flat = declare("matrix(9)")  # incremento de tensões
        # iniciais representadas num vetor de 9 posições

        cosAng = declare("float")  # ângulo para verificação de carregamento

        Ftol, lambdaS, yield1, yield2, yield_flag = declare("matrix(1)", 5)
        yield_flag1, maxitsP, pegasus, pegasus0  = declare("matrix(1)", 4)
        pegasus1, maxitsC, maxitsPU, nsub = declare("matrix(1)", 4)

        CE = declare("matrix(36)")  # matriz elástica vetorizada

        # zerando as variaveis criadas, para garantir que nao tem lixo

        matrix_zero(eigvals, 3)
        matrix_zero(props, 21)
        matrix_zero(stVar0, 18)
        matrix_zero(stress0_flat, 9)
        matrix_zero(stVarN, 18)
        matrix_zero(stressN_flat, 9)
        matrix_zero(stVarN0, 18)
        matrix_zero(stressN0_flat, 9)
        matrix_zero(stVarN1, 18)
        matrix_zero(stressN1_flat, 9)
        matrix_zero(stVarN2, 18)
        matrix_zero(stressN2_flat, 9)
        matrix_zero(eps_flat, 9)
        matrix_zero(eps_flat2, 9)
        matrix_zero(eps, 6)
        matrix_zero(epsN, 6)
        matrix_zero(eps_flat, 9)
        matrix_zero(eps_EL, 6)
        matrix_zero(eps_dot_p, 6)
        matrix_zero(eps_dot_p_flat, 9)
        matrix_zero(stress_inc, 6)
        matrix_zero(stress_inc_flat, 9)
        matrix_zero(CE, 36)

        # ----------------------------------------------------------------
        # Passando Algumas variaveis iniciais
        # ----------------------------------------------------------------

        # Index used to access values in arrays
        idx = 9 * d_idx

        # Flag particle to use trial values
        d_flag[d_idx] = 0

        for i in range(9):
            stress0_flat[i] = d_sigma[idx + i] / 1000.0
            stressN_flat[i] = d_sigma[idx + i] / 1000.0  # precisa disso?
            eps_flat2[i] = d_eps_dot[idx + i] * dt
            if i % 4 == 0:
                eps_flat[i] = d_eps_dot[idx + i] * dt
            else:
                eps_flat[i] = d_eps_dot[idx + i] * dt

        flat_to_voight(eps_flat, eps)

        # ----------------------------------------------------------------
        # Passando parametros
        # ----------------------------------------------------------------

        # inclinacao da linha de recompressao
        kappa = d_p_kappa[d_idx]

        # Poisson
        nu = d_p_nu[d_idx]

        # Elevacao da linha de ref para tensao de ref p1
        gamma = d_p_gamma[d_idx]

        # inclinacao da CSL no plano exlnp'
        _lambda = d_p_lambda[d_idx]

        # expoente na funcao de plastificacao
        n = d_p_n[d_idx]

        # r que controla a ditancia entre a CSL e NCL
        r = d_p_r[d_idx]

        # angulo de atrito critico (graus)
        m = d_p_m[d_idx]
        phi = d_p_phi[d_idx]

        # tensao de referencia para gamma (em pa)
        p1 = - d_p_p1[d_idx] / 1000.0

        # razao entre a tensao de referencia para gama (p1) e a tensao
        # considerada baixa
        ratioPS = d_p_ratiops[d_idx]

        # habilita a variação do ângulo de Lode, se for diferente de zero
        lode = d_p_lode[d_idx]

        # tensao minima considerada
        psmall = p1 * ratioPS

        # razao de sobreadensamento
        ocr = d_s_ocr[d_idx]

        # ve se nao cria como variavel interna pra evitar de ficar
        Mtc = d_mtc[d_idx]  # M na condição de compressão triaxial
        a = d_a[d_idx]  # parâmetro usado na equação de Mteta

        # -----------------------------------------------------------------
        # Inicializando variaveis de estado
        # -----------------------------------------------------------------

        # Obs: entra aqui apenas no primeiro loop
        if d_s_e[d_idx] == 0.0:
            d_s_ev[d_idx] = 0.0

            # Calculando tensões principais
            eig_vals_3x3_analytical_erick(stress0_flat, eigvals)
            s1 = - eigvals[0]  # compressão negativo
            s2 = - eigvals[1]
            s3 = - eigvals[2]

            # Calculando os invariantes de cambrigde iniciais (0)
            d_s_p[d_idx] = (s1 + s2 + s3) / 3.0
            d_s_q[d_idx] = (
                    sqrt(((s1 - s2) ** 2.0) + ((s2 - s3) ** 2.0) +
                         ((s3 - s1) ** 2.0)) / sqrt(2.0)
            )

            d_s_teta[d_idx] = (
                    atan((s1 - 2.0 * s2 + s3) / (sqrt(3.0) * (s1 - s3))) *
                    180.0 / pi
            )

            if lode == 0.0:
                d_s_teta[d_idx] = 30.0
            if s1 == s3:
                d_s_teta[d_idx] = 30.0  # condição isotrópica

            # Controle de material superficial
            if d_s_p[d_idx] < psmall:
                d_s_p[d_idx] = psmall
                d_s_q[d_idx] = 0.0
                sigma[0] = psmall
                sigma[1] = 0.0
                sigma[2] = 0.0
                sigma[3] = 0.0
                sigma[4] = psmall
                sigma[5] = 0.0
                sigma[6] = 0.0
                sigma[7] = 0.0
                sigma[8] = psmall
                d_s_teta[d_idx] = 30.0

            p = d_s_p[d_idx]
            q = d_s_q[d_idx]
            teta = d_s_teta[d_idx]

            d_s_m_teta[d_idx] = (
                    Mtc * ((2.0 * a) /
                           (1.0 + a + (1.0 - a) *
                            (sin(-3.0 * teta * pi / 180.0)))) ** (1.0 / 4.0)
            )
            M_teta = d_s_m_teta[d_idx]

            # se nao entrar aq, recebe a tensão passada como parâmetro, não
            #  seria preciso passar p0, mas em situacoes em que se deseja
            #  promover uma perturbacao no inicio da simulacao, seria assumido
            #  que o estado de tensao obedece sempre a condicao de consistencia
            if d_s_p0[d_idx] == 0.0:
                # calcula p0 a partir da tensão inicial e OCR
                d_s_p0[d_idx] = (
                        ocr * p * (exp(log(r) * (q / (M_teta * p)) ** n))
                )

            p0 = d_s_p0[d_idx]

            # ínidice de vazios específico
            d_s_v[d_idx] = (
                    gamma + kappa * log(p0 / (r * p)) -
                    _lambda * log(p0 / (r * p1))
            )

            v = d_s_v[d_idx]
            d_s_v0[d_idx] = v  # recebendo v0
            d_s_e[d_idx] = v - 1.0  # ínidice de vazios

            # parâmetro de estado
            d_s_y[d_idx] = v + _lambda * log(p / p1) - gamma

            # valor da função superfície de plastificação, apenas para
            #  montioramento do erro
            d_s_yield[d_idx] = (
                    (q ** n) / ((M_teta * p) ** n) + log(p / p0) / log(r)
            )

            d_s_bulk[d_idx] = v * p / kappa
            d_s_shear[d_idx] = (
                    1.5 * d_s_bulk[d_idx] * (1.0 - 2.0 * nu) / (1.0 + nu)
            )

        p = d_s_p[d_idx]
        q = d_s_q[d_idx]
        teta = d_s_teta[d_idx]
        M_teta = d_s_m_teta[d_idx]
        p0 = d_s_p0[d_idx]
        v = d_s_v[d_idx]
        e = d_s_e[d_idx]
        y = d_s_y[d_idx]
        # _yield = d_s_yield[d_idx]
        bulk = d_s_bulk[d_idx]
        shear = d_s_shear[d_idx]

        # ----------------------------------------------------------------
        # Passando variaveis de estado para vetores
        # ----------------------------------------------------------------
        stVar0[0] = d_s_plast_flag[d_idx]  # indicador: -1 elástico; 1 plástico
        stVar0[1] = d_s_ocr[d_idx]  # Passando OCR
        stVar0[2] = d_s_e[d_idx]  # Passando ínidice de vazios
        stVar0[3] = d_s_y[d_idx]  # Passando o parâmetro de estado
        stVar0[4] = d_s_p0[d_idx]  # Passando tensão de pré-adensamento
        stVar0[5] = d_s_ev[d_idx]  # contador de deformação volumétrica
        stVar0[6] = d_s_v[d_idx]  # Passando índice de vazios específico
        stVar0[7] = d_s_q[d_idx]  # passando q
        stVar0[8] = d_s_p[d_idx]  # passando p
        stVar0[9] = d_s_eta[d_idx]  # passando eta=q/p
        stVar0[10] = d_s_teta[d_idx]  # passando ângulo de lode
        stVar0[11] = d_s_m_teta[d_idx]  # passando Mteta
        stVar0[12] = d_s_eta_m_teta[d_idx]  # passando n/Mteta
        stVar0[13] = d_s_v0[d_idx]  # índice de vazios específico inicial v0
        stVar0[14] = d_s_yield[d_idx]  # erro na condição de consistência
        stVar0[15] = d_s_bulk[d_idx]  # Módulo voumétrico
        stVar0[16] = d_s_shear[d_idx]  # Módulo cisalhante
        stVar0[17] = cosAng  # ângulo para verificação de carregamento

        # ----------------------------------------------------------------
        # Inicializando variaveis para o algoritmo de retorno
        # ----------------------------------------------------------------

        T = 0.0  # tempo para o algortimo de retorno
        dT_ar = 1.0  # delta de tempo para o algortimo de retorno

        # expoente para definir o número máximos de divisão do subestep para o
        # algorítimo de integração explícita com controle automático de erro
        nc1Max = d_p_nc1max[d_idx]

        # erro máximo tolerado associado a Euler (integração explícita)
        if nc1Max == 0:
            nc1Max = 4  # Default

        # precisa estar casado com nc1Max e ncMax; número de zeros após o ponto
        dtMin = 0.1 ** (nc1Max)
        ncMax = (1 / dtMin)  # número máximo de divisões do T
        ratio = 1.0  # q
        residual = 1.0  # erro residual
        Stol = d_p_stol[d_idx]

        # erro máximo tolerado para o estado de tensão esta fora da superfície
        # de plastificação
        if Stol == 0.0:
            Stol = 0.1  # Default
        Ftol[0] = d_p_ftol[d_idx]

        # número máximo de interações para o algoritmo CORRECTION
        if Ftol[0] == 0.0:
            Ftol[0] = 0.000000001  # Default
        c = 0.0  # controlado do laço 1 do algoritimo de retorno
        c1 = 0.0  # controlador do laco 2 do algoritimo de retorno
        maxitsC[0] = d_p_maxitsc[d_idx]

        # número máximo de interações para o algoritmo PEGASUS
        if maxitsC[0] == 0:
            maxitsC[0] = 10  # Default
        maxitsP[0] = d_p_maxitsp[d_idx]

        if maxitsP[0] == 0:
            maxitsP[0] = 10  # Default

        pegasus[0] = 0.0  # alpha inicial para o código do PEGASUS
        pegasus0[0] = 0.0  # alpha0 inicial para o código do PEGASUS
        pegasus1[0] = 1.0  # alpha1 inicial para o código do PEGASUS

        # número máximo de interações para o algoritmo PEGASUS_UNLOAD
        maxitsPU[0] = d_p_maxitspu[d_idx]
        if maxitsPU[0] == 0:
            maxitsPU[0] = 3  # Default

        # número máximo de subdivisões para PEGASUS Unload
        nsub[0] = d_p_nsub[d_idx]
        if nsub[0] == 0:
            nsub[0] = 10  # Default

        # Teste de plastificação
        mod = 1  # casm_model (1 - elastic trial; 2 - plastic correction)
        lambdaS[0] = 1.0
        yield1[0] = -1.0
        yield2[0] = -1.0
        yield_flag[0] = 0.0  # 0 elástico, controlador interno plastificação
        yield_flag1[0] = 0.0  # 0 elástico, controlador interno plastificação
        yield_old = stVar0[14]  # Yield anterior para vericação do pegasus

        # ---------------------------------------------------
        # So fiz isso por causa da diferenca entre as tensoes
        # Calculando tensões principais iniciais (0)
        if (stress0_flat[0] - stress0_flat[8]) == 0.0:
            stress0_flat[0] -= 0.00001

        eig_vals_3x3_analytical_erick(stress0_flat, eigvals0)
        s1_0 = - eigvals0[0]  # compressão positivo
        s2_0 = - eigvals0[1]
        s3_0 = - eigvals0[2]

        # Calculando os invariantes de cambrigde iniciais (0)
        p_0 = (s1_0 + s2_0 + s3_0) / 3.0
        q_0 = (
                sqrt(((s1_0 - s2_0) ** 2.0) + ((s2_0 - s3_0) ** 2.0) +
                     ((s3_0 - s1_0) ** 2.0)) / sqrt(2.0)
        )

        Mteta0 = 0.0
        teta0 = 0.0
        if lode == 0.0:
            teta0 = 30.0  # Mlode = Mtc
            Mteta0 = Mtc
        else:
            if fabs(s1_0 - s3_0) < psmall:
                teta0 = 30.0  # condição isotrópica
                Mteta0 = Mtc
            else:
                teta0 = (
                        (atan((s1_0 - 2.0 * s2_0 + s3_0) /
                              (sqrt(3.0) * (s1_0 - s3_0)))) * 180.0 / pi
                )

                Mteta0 = (
                        Mtc * ((2.0 * a) /
                               (1.0 + a + (1.0 - a) *
                                (sin(-3.0 * teta0 * pi / 180.0)))) **
                        (1.0 / 4.0)
                )

        yield_old = (
                (q_0 ** n) / ((Mteta0 * p_0) ** n) + log(p_0 / p0) / log(r)
        )
        stVar0[14] = yield_old  # Yield anterior para vericação do pegasus
        stVar0[11] = Mteta0

        # So fiz isso por causa da diferenca entre as tensoes
        # ---------------------------------------------------

        # ----------------------------------------------------------------
        # Passandro parâmetros
        # ----------------------------------------------------------------

        props[0] = kappa
        props[1] = nu
        props[2] = gamma
        props[3] = _lambda
        props[4] = n
        props[5] = r
        props[6] = m
        props[7] = phi
        props[8] = p1
        props[9] = ratioPS
        props[10] = lode
        props[11] = nc1Max
        props[12] = Stol
        props[13] = Ftol[0]
        props[14] = maxitsC[0]
        props[15] = maxitsP[0]
        props[16] = maxitsPU[0]
        props[17] = nsub[0]
        props[18] = psmall
        props[19] = Mtc
        props[20] = a

        # ----------------------------------------------------------------
        # Chamando o modelo constitutivo
        # ----------------------------------------------------------------

        p = stVar0[8]
        p0 = stVar0[4]
        pTR = - (d_sigma_tr[0] + d_sigma_tr[4] + d_sigma_tr[8]) / 3.0
        er = 0

        if d_s_mc[d_idx] > 0.0:

            # Flag particle to use trial values
            d_flag[d_idx] = 0

            # Initialize stress, stress rate, and plastic rate tensors
            for i in range(9):
                sigma[i] = d_sigma_tr[idx + i]
                sig_dot[i] = d_sigma_dot[idx + i]
                spin_dot[i] = d_spin_dot[idx + i]
                d_eps_p_dot[idx + i] = 0.0

            # If not elastic material
            if self.c_model > 0 and d_s_mc[d_idx] < 2.0:

                p = d_p[d_idx]
                q = d_q[d_idx]
                aphi = d_aphi[d_idx]
                apsi = d_apsi[d_idx]
                ac = d_ac[d_idx]
                sy = d_sy[d_idx]
                h_mod = 0.0

                # Calculate accumulated total and deviatoric plastic strains,
                # and initialize temp Cauchy stress
                ep_vol = (
                        (d_eps_p[idx] + d_eps_p[idx + 4] + d_eps_p[idx + 8]) /
                        3.0
                )
                ep_dev2 = 0.0
                ep_acc2 = 0.0
                for i in range(9):
                    # Initialize stress
                    sigma[i] = d_sigma[idx + i]

                    # Accumulated total and deviatoric plastic strains
                    ep = d_eps_p[idx + i]
                    ep_acc2 += ep * ep
                    if i % 4 == 0:
                        ep -= ep_vol
                    ep_dev2 += ep * ep

                ep_acc = sqrt(ep_acc2)
                ep_eff = sqrt(2.0 * ep_dev2 / 3.0)
                d_ep_eff[d_idx] = ep_eff
                d_ep_acc[d_idx] = ep_acc

                # ======================== SOFTENING ==========================

                # Softening coefficients
                eta_phi = 0.0
                eta_psi = 0.0
                eta_c = 0.0

                # Peak and residual values
                phi_p = d_phi[d_idx]  # se MC passo phi, não o p_phi
                phi_res = 15.0
                psi_p = d_psi[d_idx]
                c_p = 5000.0
                c_res = 0.0

                # Calculate new values of material parameters
                phi = (
                        (phi_res + (phi_p - phi_res) *
                         exp(-eta_phi * ep_eff)) * pi / 180.0
                )
                psi = (psi_p * exp(-eta_psi * ep_acc)) * pi / 180.0
                c = c_res + (c_p - c_res) * exp(-eta_c * ep_eff)

                # Update D-P parameters
                sy = sqrt(3.0) * c
                aphi = sqrt(6) * tan(phi) / sqrt(3 + 4 * pow(tan(phi), 2))
                apsi = sqrt(6) * tan(psi) / sqrt(3 + 4 * pow(tan(psi), 2))
                ac = sqrt(2) / sqrt(3 + 4 * pow(tan(phi), 2))

                # =============================================================

                # Check for yielding
                y = yield_criterion_dp(q, p, aphi, ac, sy)

                # If yielding
                if y > 1e-6:

                    # Elastic constants
                    k = d_bulk[d_idx]
                    g = d_shear[d_idx]

                    # Plastic modulus
                    if sy > 0:  # No hardening if no cohesion
                        h_mod = d_h_mod[d_idx]

                    # Plastic multiplier
                    dgamma = y / (2 * g + k * aphi * apsi + ac * ac * h_mod)

                    # Volumetric strain rate
                    eps_dot_v = d_eps_dot[idx] + d_eps_dot[idx + 4] + \
                                d_eps_dot[idx + 8]

                    # Check if return to the cone is valid
                    norm_s = sqrt(2.0 / 3.0) * q
                    if norm_s - 2 * g * dgamma < 0.0:

                        # Initialize variables for apex calculations
                        dgamma = 0.0
                        pn = (sigma[0] + sigma[4] + sigma[8]) / 3.0

                        if apsi != 0.0:

                            # Plastic multiplier and rate
                            dgamma = (aphi * p - ac * sy) / \
                                     (k * aphi * apsi + ac * ac * h_mod)
                            dg_dot = dgamma / dt

                            # Volumetric plastic strain rate
                            eps_p_dot_v = dg_dot * apsi

                            # Volumetric and deviatoric elastic strain rate
                            eps_e_dot_v = eps_dot_v - eps_p_dot_v

                            # Update stress and strain rates
                            for i in range(9):

                                # Get rid of spin rate
                                spin_dot[i] = 0.0

                                # Update plastic strain rate
                                ep_dot = d_eps_dot[idx + i]
                                if i % 4 == 0:
                                    ep_dot = eps_p_dot_v / 3.0
                                d_eps_p_dot[idx + i] = ep_dot

                                # Updated small strain stress
                                dp = k * eps_e_dot_v * dt
                                s_i = -sigma[i]
                                if i % 4 == 0:
                                    s_i += pn + dp
                                sigma[i] += s_i

                                # Updated stress rate
                                sig_dot[i] = s_i / dt

                        else:
                            # If dilation angle is zero, the particle is
                            # treated as if all excess deformation is plastic
                            # and the maximum stress is: p_max = ac*sy/aphi
                            p_max = ac * sy / aphi  # Maximum tensile stress
                            eps_e_dot_v = (p_max - pn) / (3 * k * dt)

                            for i in range(9):
                                eps_e_dot = 0.0
                                sig_dot[i] = -sigma[i] / dt
                                spin_dot[i] = 0.0

                                if i % 4 == 0:
                                    eps_e_dot = eps_e_dot_v
                                    sig_dot[i] += p_max / dt

                                d_eps_p_dot[idx + i] = (d_eps_dot[idx + i] -
                                                        eps_e_dot)

                    # Return to the smooth part of the cone
                    else:

                        # Rate of plastic multiplier
                        dg_dot = dgamma / dt

                        # Volumetric plastic strain
                        eps_p_dot_v = dg_dot * apsi

                        for i in range(9):

                            # Deviatoric plastic flow directions
                            n_i = d_sigma_dev[idx + i] / norm_s

                            # Total strain rate decomposition
                            eps_dot_d = d_eps_dot[idx + i]
                            if i % 4 == 0:
                                eps_dot_d -= eps_dot_v / 3.0

                            # Plastic strain rate
                            eps_p_dot_d = dg_dot * n_i

                            eps_p_dot_h = 0.0
                            if i % 4 == 0:
                                eps_p_dot_h = eps_p_dot_v / 3.0
                            d_eps_p_dot[idx + i] = eps_p_dot_d + eps_p_dot_h

                            # Stress rate
                            eps_e_dot_d = eps_dot_d - eps_p_dot_d
                            eps_e_dot_v = eps_dot_v - eps_p_dot_v
                            p_i = 0.0
                            if i % 4 == 0:
                                p_i = k * eps_e_dot_v
                            sdot = p_i + 2 * g * eps_e_dot_d

                            # Small-strain stress rate
                            sig_dot[i] = sdot

                            # Small-strain updated stress
                            sigma[i] += sdot * dt

                    # Update uniaxial yield stress
                    dlamb = dgamma * ac
                    d_sy[d_idx] += h_mod * dlamb

                    # Flag particle to update stress and strain
                    d_flag[d_idx] = 1

            # Jaumann stress rate (~ large deformation)
            matrix_multiply(sigma, spin_dot, sig_spin, 3)
            matrix_multiply(spin_dot, sigma, sig_spin_t, 3)

            for i in range(9):
                d_sigma_dot[idx + i] = sig_dot[i] - sig_spin[i] + sig_spin_t[i]

            for i in range(9):
                stressN_flat[i] += sig_dot[i] * dt / 1000.0

            # Calculando tensões principais após a previsão elástica (I)
            eig_vals_3x3_analytical_erick(stressN_flat, eigvals)
            s1_N = - eigvals[0]  # compressão positivo
            s2_N = - eigvals[1]
            s3_N = - eigvals[2]

            # Calculando os invariantes de cambrigde iniciais (0)
            pN = (s1_N + s2_N + s3_N) / 3.0
            qN = (
                    sqrt(((s1_N - s2_N) ** 2.0) + ((s2_N - s3_N) ** 2.0) +
                         ((s3_N - s1_N) ** 2.0)) / sqrt(2.0)
            )

            pN = (
                    -(stressN_flat[0] + stressN_flat[4] + stressN_flat[8]) /
                    3.0
            )

            if pN < psmall:
                pN = psmall
                qN = Mtc * pN
            ocr = 1.0
            p0 = (pN * exp(log(r) * ((qN / (Mtc * qN)) ** n)))
            if isnan(p0): p0 = stVar0[4]
            if p0 < psmall: p0 = psmall
            evInc = - (eps[0] + eps[1] + eps[2])
            ev = stVar0[5] + evInc  # deformação volumétrica acumulada
            v = stVar0[13] * (1.0 - ev)  # referecnial inicial
            y = 0
            e = v - 1.0
            if v < gamma:
                v = gamma
                ev = 1 - v / stVar0[13]
                e = v - 1.0
            tetaN = 30.0
            MtetaN = Mtc

            stVarN[0] = 1.0
            stVarN[1] = ocr
            stVarN[2] = e
            stVarN[3] = y
            stVarN[4] = p0
            stVarN[5] = ev
            stVarN[6] = v
            stVarN[7] = qN
            stVarN[8] = pN
            stVarN[9] = qN / pN
            stVarN[10] = tetaN
            stVarN[11] = MtetaN
            stVarN[12] = qN / (pN * MtetaN)
            stVarN[13] = stVar0[13]
            stVarN[14] = yield1[0]
            stVarN[15] = stVar0[15]
            stVarN[16] = stVar0[16]
            stVarN[17] = stVar0[17]

            for i in range(18):
                stVarN[i] = stVar0[i]

            if d_s_mc[d_idx] > 0.0:
                evInc = - (eps[0] + eps[1] + eps[2])
                ev = stVar0[5] + evInc  # deformação volumétrica acumulada
                v = stVar0[13] * (1.0 - ev)  # referecnial inicial
                e = v - 1.0

                stVarN[1] = 0
                stVarN[2] = e
                stVarN[3] = 0
                stVarN[4] = 0
                stVarN[5] = ev
                stVarN[6] = v
                stVarN[7] = qN
                stVarN[8] = pN
                stVarN[9] = qN / pN
                stVarN[12] = qN / (pN * MtetaN)

            stVarN[0] = 2.0

        else:
            # -----------------------------------------------------------------
            # CASM
            # -----------------------------------------------------------------
            # Nessa parte acredito que só precisa chamar a parcela el[astica,
            # pois caso ocorra plastificação, a correção plástica feita
            # anteriormente é perdida]

            casm_model(props, stress0_flat, stVar0, eps, yield1,
                       stressN_flat, stVarN, yield2, Ftol, lambdaS, yield_flag,
                       eps_flat, d_idx, mod)

            # ------------------------------------------------------------------

            # ----------------------------------------------------------------
            # Algoritmo de Retorno
            # ----------------------------------------------------------------

            er = 0
            if yield_flag[0] == 1.0:
                # Plastificou
                mod = 0  # tentativa de otmizar o código usando apenas aparte
                # da revisão plástica ou correção, ainda não funciona, precisa
                # ser implementado

                if yield_old < - Ftol[0] and yield2[0] > Ftol[0]:
                    # if er == 1:
                    for i in range(9):
                        stressN0_flat[i] = stress0_flat[i]
                    for i in range(18):
                        stVarN0[i] = stVar0[i]
                    # Transicao elastico para elastoplastico
                    pegasus_alg(props, stress0_flat, stVar0, eps,
                                stressN0_flat, stVarN0, Ftol, maxitsP, pegasus,
                                pegasus0, pegasus1, d_idx, yield_old)
                else:
                    for i in range(9):
                        stressN0_flat[i] = stress0_flat[i]
                    for i in range(18):
                        stVarN0[i] = stVar0[i]
                    cosAng = stVarN[17]
                    if cosAng < -0.2:

                        # Pegasus unload
                        pegasus_unload(props, stress0_flat, stVar0, eps, Ftol,
                                       maxitsPU, nsub, pegasus, pegasus0,
                                       pegasus1, d_idx)

                        if pegasus0[0] > 0.0 and pegasus1[0] < 1.0:
                            pegasus_alg(props, stress0_flat, stVar0, eps,
                                        stressN0_flat, stVarN0, Ftol, maxitsP,
                                        pegasus, pegasus0, pegasus1, d_idx,
                                        yield_old)

                while T < 1.0 and c <= ncMax:

                    residual = 1.0
                    c += 1.0
                    c1 = 0.0

                    while residual > Stol and T < 1.0 and c1 < nc1Max:
                        # Teste de aceitação do step

                        if fabs(residual) > Stol and c1 > 0.0:  # passa apenas
                            # na segunda vez

                            # Redução do step
                            ratio = max(0.9 * sqrt(Stol / residual), 0.1)
                            dT_ar = dT_ar * ratio
                            if dT_ar < dtMin:
                                dT_ar = dtMin
                            if (dT_ar + T) > 1.0:
                                dT_ar = 1.0 - T  # assegura T não passe de 1
                            if dT_ar < 0.0:
                                dT_ar = 0.0  # assegura que dT seja positivo
                        c1 += 1.0

                        for i in range(6):
                            epsN[i] = eps[i] * (1.0 - pegasus[0]) * dT_ar

                        voight_to_flat(epsN, epsN_flat)

                        # Euler modificado

                        casm_model(props, stressN0_flat, stVarN0, epsN, yield1,
                                   stressN1_flat, stVarN1, yield2, Ftol,
                                   lambdaS, yield_flag1, epsN_flat, d_idx, mod)

                        casm_model(props, stressN1_flat, stVarN1, epsN, yield1,
                                   stressN2_flat, stVarN2, yield2, Ftol,
                                   lambdaS, yield_flag1, epsN_flat, d_idx, mod)

                        flat_to_voight(stressN0_flat, stressN0)
                        flat_to_voight(stressN1_flat, stressN1)
                        flat_to_voight(stressN2_flat, stressN2)

                        Aux1 = 0.0
                        Aux2 = 0.0
                        for i in range(6):
                            dSig1 = stressN1[i] - stressN0[i]
                            dSig2 = stressN2[i] - stressN1[i]
                            stressN[i] = (stressN0[i] + (dSig1 + dSig2) / 2.0)
                            Aux1 = Aux1 + (dSig2 - dSig1) ** (2.0)
                            Aux2 = Aux2 + (stressN[i]) ** 2.0

                        # posso melhorar isso só passando o loop pelas
                        # variáveis de estado que precisam ser atualizadas aqui
                        for i in range(18):
                            Aux3 = stVarN1[i] - stVarN0[i]
                            Aux4 = stVarN2[i] - stVarN1[i]
                            stVarN[i] = stVarN0[i] + (Aux3 + Aux4) / 2.0
                        dp01 = stVarN1[4] - stVarN0[4]
                        dp02 = stVarN2[4] - stVarN1[4]
                        dp03 = dp02 - dp01

                        residual = (
                                0.5 * max((sqrt(Aux1) / sqrt(Aux2)),
                                          (fabs(dp03) / stVarN[4]))
                        )

                        if dT_ar < dtMin: c1 = nc1Max + 1  # sai do loop 2

                    # Verificando superfície de plastificacao
                    p0N = stVarN[4]
                    qN = stVarN[7]
                    pN = stVarN[8]
                    MtetaN = stVarN[11]
                    yield1[0] = (
                            (qN ** n) / ((MtetaN * pN) ** n) + log(pN / p0N) /
                            log(r)
                    )
                    stVarN[14] = yield1[0]

                    voight_to_flat(stressN, stressN_flat)  # Coloquei isso dps,
                    # mas acho que precisava se não ele não estava usando as
                    # correções

                    fail = 1.0  # usado para controlar o aumento de dT_ar
                    if fabs(yield1[0]) > Ftol[0]:
                        # Nao devia precisar disso, talvez se dt atingir dtmin,
                        #  tenta fazer a correção
                        fail = 0.0
                        for i in range(6):
                            stressN0[i] = stressN[i]
                        for i in range(18):
                            stVarN0[i] = stVarN[i]

                        voight_to_flat(stressN0, stressN0_flat)

                        correction(props, stressN0_flat, stVarN0, yield1,
                                   stressN_flat, stVarN, Ftol, maxitsC, d_idx)

                    # Obs: acho que so precisa fazer isso se T<1, evita pelo
                    #  menos o ultimo ciclo de calculo, tentar implementar esse
                    #  if depois
                    flat_to_voight(stressN_flat, stressN)
                    for i in range(6):
                        stressN0[i] = stressN[i]
                    for i in range(18):
                        stVarN0[i] = stVarN[i]
                    voight_to_flat(stressN0, stressN0_flat)

                    # incrementando o tempo
                    T += dT_ar
                    if T > 0.0 and dT_ar < 1.0:
                        # Aumento do step
                        ratio = min((0.9 * sqrt(Stol / residual)), 1.1)

                        # controla o aumento de dT, caso o step falhe
                        if fail == 0.0:
                            ratio = min(ratio, 1.0)
                        dT_ar = dT_ar * ratio
                        if dT_ar < dtMin:
                            dT_ar = dtMin
                        if (dT_ar + T) > 1.0:
                            dT_ar = 1.0 - T  # assegura que T não passe de 1
                        if dT_ar < 0.0:
                            dT_ar = 0.0  # assegura que dT seja positivo

        # ----------------------------------------------------------------
        # Correção de tensões negativas
        # ----------------------------------------------------------------

        pN = stVarN[8]
        er = 1
        if pN <= psmall and stVarN[0] != 2.0:
            # Flag particle to use trial values
            d_flag[d_idx] = 0

            # Initialize stress, stress rate, and plastic rate tensors
            for i in range(9):
                sigma[i] = d_sigma_tr[idx + i]
                sig_dot[i] = d_sigma_dot[idx + i]
                spin_dot[i] = d_spin_dot[idx + i]
                d_eps_p_dot[idx + i] = 0.0

            # If not elastic material
            if self.c_model > 0:

                p = d_p[d_idx]
                q = d_q[d_idx]
                aphi = d_aphi[d_idx]
                apsi = d_apsi[d_idx]
                ac = d_ac[d_idx]
                sy = d_sy[d_idx]
                h_mod = 0.0

                # Calculate accumulated total and deviatoric plastic strains,
                # and initialize temp Cauchy stress
                ep_vol = (
                        (d_eps_p[idx] + d_eps_p[idx + 4] + d_eps_p[idx + 8]) /
                        3.0
                )
                ep_dev2 = 0.0
                ep_acc2 = 0.0
                for i in range(9):
                    # Initialize stress
                    sigma[i] = d_sigma[idx + i]

                    # Accumulated total and deviatoric plastic strains
                    ep = d_eps_p[idx + i]
                    ep_acc2 += ep * ep
                    if i % 4 == 0:
                        ep -= ep_vol
                    ep_dev2 += ep * ep

                ep_acc = sqrt(ep_acc2)
                ep_eff = sqrt(2.0 * ep_dev2 / 3.0)
                d_ep_eff[d_idx] = ep_eff
                d_ep_acc[d_idx] = ep_acc

                # ======================== SOFTENING ==========================

                # Softening coefficients
                eta_phi = 0.0
                eta_psi = 0.0
                eta_c = 0.0

                # Peak and residual values
                phi_p = d_p_phi[d_idx]
                phi_res = 15.0
                psi_p = d_psi[d_idx]
                c_p = 5000.0
                c_res = 0.0

                # Calculate new values of material parameters
                phi = (
                        (phi_res + (phi_p - phi_res) *
                         exp(-eta_phi * ep_eff)) * pi / 180.0
                )
                psi = (psi_p * exp(-eta_psi * ep_acc)) * pi / 180.0
                c = c_res + (c_p - c_res) * exp(-eta_c * ep_eff)

                # Update D-P parameters
                sy = sqrt(3.0) * c
                aphi = sqrt(6) * tan(phi) / sqrt(3 + 4 * pow(tan(phi), 2))
                apsi = sqrt(6) * tan(psi) / sqrt(3 + 4 * pow(tan(psi), 2))
                ac = sqrt(2) / sqrt(3 + 4 * pow(tan(phi), 2))

                # =============================================================

                # Check for yielding
                y = yield_criterion_dp(q, p, aphi, ac, sy)

                # If yielding
                if y > 1e-6:

                    # Elastic constants
                    k = d_bulk[d_idx]
                    g = d_shear[d_idx]

                    # Plastic modulus
                    if sy > 0:  # No hardening if no cohesion
                        h_mod = d_h_mod[d_idx]

                    # Plastic multiplier
                    dgamma = y / (2 * g + k * aphi * apsi + ac * ac * h_mod)

                    # Volumetric strain rate
                    eps_dot_v = (
                            d_eps_dot[idx] + d_eps_dot[idx + 4] +
                            d_eps_dot[idx + 8]
                    )

                    # Check if return to the cone is valid
                    norm_s = sqrt(2.0 / 3.0) * q
                    if norm_s - 2 * g * dgamma < 0.0:

                        # Initialize variables for apex calculations
                        dgamma = 0.0
                        pn = (sigma[0] + sigma[4] + sigma[8]) / 3.0

                        if apsi != 0.0:

                            # Plastic multiplier and rate
                            dgamma = (
                                    (aphi * p - ac * sy) /
                                    (k * aphi * apsi + ac * ac * h_mod)
                            )
                            dg_dot = dgamma / dt

                            # Volumetric plastic strain rate
                            eps_p_dot_v = dg_dot * apsi

                            # Volumetric and deviatoric elastic strain rate
                            eps_e_dot_v = eps_dot_v - eps_p_dot_v

                            # Update stress and strain rates
                            for i in range(9):

                                # Get rid of spin rate
                                spin_dot[i] = 0.0

                                # Update plastic strain rate
                                ep_dot = d_eps_dot[idx + i]
                                if i % 4 == 0:
                                    ep_dot = eps_p_dot_v / 3.0
                                d_eps_p_dot[idx + i] = ep_dot

                                # Updated small strain stress
                                dp = k * eps_e_dot_v * dt
                                s_i = -sigma[i]
                                if i % 4 == 0:
                                    s_i += pn + dp
                                sigma[i] += s_i

                                # Updated stress rate
                                sig_dot[i] = s_i / dt

                        else:
                            # If dilation angle is zero, the particle is
                            # treated as if all excess deformation is plastic
                            # and the maximum stress is: p_max = ac*sy/aphi
                            p_max = ac * sy / aphi  # Maximum tensile stress
                            eps_e_dot_v = (p_max - pn) / (3 * k * dt)

                            for i in range(9):
                                eps_e_dot = 0.0
                                sig_dot[i] = -sigma[i] / dt
                                spin_dot[i] = 0.0

                                if i % 4 == 0:
                                    eps_e_dot = eps_e_dot_v
                                    sig_dot[i] += p_max / dt

                                d_eps_p_dot[idx + i] = (
                                        d_eps_dot[idx + i] - eps_e_dot
                                )

                    # Return to the smooth part of the cone
                    else:

                        # Rate of plastic multiplier
                        dg_dot = dgamma / dt

                        # Volumetric plastic strain
                        eps_p_dot_v = dg_dot * apsi

                        for i in range(9):

                            # Deviatoric plastic flow directions
                            n_i = d_sigma_dev[idx + i] / norm_s

                            # Total strain rate decomposition
                            eps_dot_d = d_eps_dot[idx + i]
                            if i % 4 == 0:
                                eps_dot_d -= eps_dot_v / 3.0

                            # Plastic strain rate
                            eps_p_dot_d = dg_dot * n_i

                            eps_p_dot_h = 0.0
                            if i % 4 == 0:
                                eps_p_dot_h = eps_p_dot_v / 3.0
                            d_eps_p_dot[idx + i] = eps_p_dot_d + eps_p_dot_h

                            # Stress rate
                            eps_e_dot_d = eps_dot_d - eps_p_dot_d
                            eps_e_dot_v = eps_dot_v - eps_p_dot_v
                            p_i = 0.0
                            if i % 4 == 0:
                                p_i = k * eps_e_dot_v
                            sdot = p_i + 2 * g * eps_e_dot_d

                            # Small-strain stress rate
                            sig_dot[i] = sdot

                            # Small-strain updated stress
                            sigma[i] += sdot * dt

                    # Update uniaxial yield stress
                    dlamb = dgamma * ac
                    d_sy[d_idx] += h_mod * dlamb

                    # Flag particle to update stress and strain
                    d_flag[d_idx] = 1

            # Jaumann stress rate (~ large deformation)
            matrix_multiply(sigma, spin_dot, sig_spin, 3)
            matrix_multiply(spin_dot, sigma, sig_spin_t, 3)

            for i in range(9):
                d_sigma_dot[idx + i] = sig_dot[i] - sig_spin[i] + sig_spin_t[i]

            for i in range(9):
                stressN_flat[i] = stress0_flat[i]
                stressN_flat[i] += sig_dot[i] * dt / 1000.0

            # Calculando tensões principais após a previsão elástica (I)
            eig_vals_3x3_analytical_erick(stressN_flat, eigvals)
            s1_N = - eigvals[0]  # compressão positivo
            s2_N = - eigvals[1]
            s3_N = - eigvals[2]

            # Calculando os invariantes de cambrigde iniciais (0)
            pN = (s1_N + s2_N + s3_N) / 3.0
            qN = (
                    sqrt(((s1_N - s2_N) ** 2.0) + ((s2_N - s3_N) ** 2.0) +
                         ((s3_N - s1_N) ** 2.0)) / sqrt(2.0)
            )

            pN = - (stressN_flat[0] + stressN_flat[4] + stressN_flat[8]) / 3.0
            if pN < psmall:
                pN = psmall
                qN = Mtc * pN
            ocr = 1.0

            p0 = (pN * exp(log(r) * ((qN / (Mtc * qN)) ** n)))
            if isnan(p0):
                p0 = stVar0[4]
            if p0 < psmall:
                p0 = psmall
            evInc = -(eps[0] + eps[1] + eps[2])
            ev = stVar0[5] + evInc  # deformação volumétrica acumulada
            v = stVar0[13] * (1.0 - ev)  # referecnial inicial
            y = 0
            e = v - 1.0
            if v < gamma:
                v = gamma
                ev = 1 - v / stVar0[13]
                e = v - 1.0
            tetaN = 30.0
            MtetaN = Mtc

            stVarN[0] = 1.0
            stVarN[1] = ocr
            stVarN[2] = e
            stVarN[3] = y
            stVarN[4] = p0
            stVarN[5] = ev
            stVarN[6] = v
            stVarN[7] = qN
            stVarN[8] = pN
            stVarN[9] = qN / pN
            stVarN[10] = tetaN
            stVarN[11] = MtetaN
            stVarN[12] = qN / (pN * MtetaN)
            stVarN[13] = stVar0[13]
            stVarN[14] = yield1[0]
            stVarN[15] = stVar0[15]
            stVarN[16] = stVar0[16]
            stVarN[17] = stVar0[17]

            for i in range(18):
                stVarN[i] = stVar0[i]

            stVarN[0] = 3.0  # Mostra que o modelo é DP

        # ----------------------------------------------------------------
        # Atualizando variaveis de estado
        # ----------------------------------------------------------------

        # Passando parâmetros de estado
        d_s_plast_flag[d_idx] = stVarN[0]  # passando indicador se plástico
        d_s_ocr[d_idx] = stVarN[1]  # Passando OCR
        d_s_e[d_idx] = stVarN[2]  # Passando ínidice de vazios
        d_s_y[d_idx] = stVarN[3]  # Passando o parâmetro de estado
        d_s_p0[d_idx] = stVarN[4]  # Passando p0
        d_s_ev[d_idx] = stVarN[5]  # passando o contador de ev
        d_s_v[d_idx] = stVarN[6]  # Passando v
        d_s_q[d_idx] = stVarN[7]  # passando q
        d_s_p[d_idx] = stVarN[8]  # passando p
        d_s_eta[d_idx] = stVarN[9]  # passando eta=q/p
        d_s_teta[d_idx] = stVarN[10]  # passando ângulo de lode
        d_s_m_teta[d_idx] = stVarN[11]  # passando Mteta
        d_s_eta_m_teta[d_idx] = stVarN[12]  # passando n/Mteta
        d_s_v0[d_idx] = stVar0[13]  # passando v0
        d_s_yield[d_idx] = stVarN[14]  # Passando f
        if d_s_mc[d_idx] == 0.0:
            d_bulk[d_idx] = stVarN[15] * 1000  # Módulo voumétrico
            d_shear[d_idx] = stVarN[16] * 1000  # Módulo cisalhante
        d_s_bulk[d_idx] = stVarN[15]  # Módulo voumétrico
        d_s_shear[d_idx] = stVarN[16]  # Módulo cisalhante

        if stVarN[0] == 1:
            d_flag[d_idx] = 1  # indicador de plastificacao

        # ----------------------------------------------------------------
        # Atualizando tensoes e deformações plastica
        # ----------------------------------------------------------------

        # incremento de deformacoes plasticas
        a1 = 1.0 / (d_s_bulk[d_idx] * 3.0 * (1.0 - 2.0 * nu))
        a2 = - nu * a1
        a3 = 1.0 / d_s_shear[d_idx]

        CE[0] = a1
        CE[7] = a1
        CE[14] = a1
        CE[1] = a2
        CE[2] = a2
        CE[6] = a2
        CE[8] = a2
        CE[12] = a2
        CE[13] = a2
        CE[21] = a3
        CE[28] = a3
        CE[35] = a3

        for i in range(9):
            stress_inc_flat[i] = (stressN_flat[i] - stress0_flat[i]) * 1000

        flat_to_voight(stress_inc_flat, stress_inc)
        matrix_multiply_vector(CE, stress_inc, eps_EL, 6)

        for i in range(6):
            eps_dot_p[i] = (eps[i] - eps_EL[i]) / dt

        voight_to_flat(eps_dot_p, eps_dot_p_flat)

        for i in range(9):
            d_sigma_dot[idx + i] = (
                    (stressN_flat[i] - stress0_flat[i]) * 1000 / dt
            )
            sigma[i] = stressN_flat[i] / 1000.0
            spin_dot[i] = d_spin_dot[idx + i]
            d_eps_p_dot[idx + i] = eps_dot_p_flat[i]

        # Jaumann stress rate (~ large deformation)
        matrix_multiply(sigma, spin_dot, sig_spin, 3)
        matrix_multiply(spin_dot, sigma, sig_spin_t, 3)

        d_flag[d_idx] = 1

        # ----------------------------------------------------------------
        # Atualizando poropressoes
        # ----------------------------------------------------------------
        if d_s_undrained[d_idx] == 1.0:
            d_pw[d_idx] += self.bulkw * (eps[0] + eps[1] + eps[2])
        if d_pw[d_idx] > 0.0:
            d_pw[d_idx] = 0.0
        # ------------------------------

    def _get_helpers_(self):
        return [yield_criterion_dp, matrix_multiply, flat_to_voight,
                matrix_multiply_vector, voight_to_flat,
                eig_vals_3x3_analytical_erick, casm_model,
                matrix_eigenvectors_erick, vector_normalize, matrix_transpose,
                matrix_eigenvectors, eig_vals_3x3_analytical, matrix_zero,
                correction, pegasus_alg, pegasus_unload, casm_model_EL,
                yield_criterion_dp]
