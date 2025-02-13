from pysph.sph.equation import Equation
from compyle.api import declare
from math import (pi, fabs, sqrt, log, pow, exp, tan, sin, cos, acos, isnan,
                  atan)

from matrix_operations import matrix_multiply, matrix_multiply_vector


# ============================= Helper Functions ==============================

def yield_criterion_dp(q=0.0, p=0.0, aphi=0.0, ac=0.0, sy=0.0):
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
    :param sy:

    Output
    -----------
    :return:
    """
    return sqrt(2.0/3.0) * q + aphi * p - ac * sy

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

# =============================================================================

class DruckerPragerSolverExact(Equation):

    def __init__(self, dest, sources, c_model=1, debug=0):
        self.c_model = c_model
        self.debug = debug
        super(DruckerPragerSolverExact, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_q, d_sigma, d_sigma_tr, d_sigma_dev,
                   d_sigma_dot, d_eps_dot, d_spin_dot, d_eps_p, d_eps_p_dot,
                   d_ep_acc, d_ep_eff, d_aphi, d_apsi, d_ac, d_sy, d_h_mod,
                   d_bulk, d_shear, d_phi, d_cohesion, d_psi, d_flag, dt):

        i, idx = declare("int", 2)
        sigma, sig_dot, sig_spin, sig_spin_t = declare("matrix(9)", 4)
        spin_dot = declare("matrix(9)")

        # Index used to access values in arrays
        idx = 9*d_idx

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

            # Calculate accumulated total and deviatoric plastic strains, and
            #  initialize temp Cauchy stress
            ep_vol = (d_eps_p[idx] + d_eps_p[idx + 4] + d_eps_p[idx + 8]) / 3.0
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

            # =========================== SOFTENING ===========================

            # Softening coefficients
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
            psi = (psi_p * exp(-eta_psi * ep_acc)) * pi / 180.0
            c = c_res + (c_p - c_res) * exp(-eta_c * ep_eff)

            # Update D-P parameters
            sy = sqrt(3.0) * c
            aphi = sqrt(6) * tan(phi) / sqrt(3 + 4 * pow(tan(phi), 2))
            apsi = sqrt(6) * tan(psi) / sqrt(3 + 4 * pow(tan(psi), 2))
            ac = sqrt(2) / sqrt(3 + 4 * pow(tan(phi), 2))

            # =================================================================

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
                dgamma = y / (2*g + k*aphi*apsi + ac*ac*h_mod)

                # Volumetric strain rate
                eps_dot_v = d_eps_dot[idx] + d_eps_dot[idx + 4] + \
                            d_eps_dot[idx + 8]

                # Check if return to the cone is valid
                norm_s = sqrt(2.0/3.0) * q
                if norm_s - 2*g*dgamma < 0.0:

                    # Initialize variables for apex calculations
                    dgamma = 0.0
                    pn = (sigma[0] + sigma[4] + sigma[8]) / 3.0

                    if apsi != 0.0:

                        # Plastic multiplier and rate
                        dgamma = (aphi*p - ac*sy) / \
                                 (k*aphi*apsi + ac*ac*h_mod)
                        dg_dot = dgamma / dt

                        # Volumetric plastic strain rate
                        eps_p_dot_v = dg_dot*apsi

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
                            dp = k*eps_e_dot_v*dt
                            s_i = -sigma[i]
                            if i % 4 == 0:
                                s_i += pn + dp
                            sigma[i] += s_i

                            # Updated stress rate
                            sig_dot[i] = s_i/dt

                    else:
                        # If dilation angle is zero, the particle is treated as
                        #  if all excess deformation is plastic and the maximum
                        #  stress is: p_max = ac*sy/aphi
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
                        eps_p_dot_d = dg_dot*n_i

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

    def _get_helpers_(self):
        return[yield_criterion_dp, matrix_multiply]


class ModifiedCamClay(Equation):
    def __init__(self, dest, sources, nu=0.3, c_model=1, tol=1e-5,
                 max_iter=100, debug=0):
        self.nu = nu
        self.c_model = c_model
        self.tol = tol
        self.max_iter = max_iter
        self.debug = debug
        super(ModifiedCamClay, self).__init__(dest, sources)

    def initialize(self, d_idx, d_p, d_q, d_sigma, d_sigma_tr, d_sigma_dev,
                   d_void_ratio, d_void_ref, d_sigma_dot, d_eps_dot,
                   d_spin_dot, d_eps_p_dot, d_ep_acc, d_pc, d_lambda_mcc,
                   d_kappa_mcc, d_ms, d_bulk, d_shear, d_flag, d_gid, dt):

        i, idx, count = declare("int", 3)
        sigma, sig_dot, sig_spin, sig_spin_t = declare("matrix(9)", 4)
        spin_dot = declare("matrix(9)")

        # Index used to access values in arrays
        idx = 9*d_idx

        # Flag particle to use trial values
        d_flag[d_idx] = 0

        # Consolidation parameters
        lambda_mcc = d_lambda_mcc[d_idx]
        kappa_mcc = d_kappa_mcc[d_idx]
        void_ratio = d_void_ratio[d_idx]
        void_ref = d_void_ref[d_idx]
        v = (1 + void_ratio) / (lambda_mcc - kappa_mcc)
        pc0 = -98000

        # Initialize stress and stress rate tensors
        for i in range(9):
            sigma[i] = d_sigma_tr[idx + i]
            sig_dot[i] = d_sigma_dot[idx + i]

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
                    dy = (2 * p - pc) * a + (2 * q / pow(ms, 2)) * b1 - p * c1

                    # Update the consistency parameter
                    dg -= y / dy

                    # Auxiliary values to find pc
                    b2 = (1 / v) * (1 + 2 * k * dg) + 2 * dg * ptr
                    c2 = pcn * (1 / v) * (1 + 2 * k * dg)
                    d1 = (b2 / dg)
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

                    # d_n[idx+i] = n_i

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

        # Jaumann stress rate (~ large deformation)
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
            void_ratio = void_ref - lambda_mcc * log(
                -pc) + kappa_mcc * log(pc / ph)
            ph = ph * 1000
            k = abs((1 + void_ratio) * p / kappa_mcc)
            g = 3 * k * (1 - 2 * self.nu) / (2 * (1 + self.nu))
            d_bulk[d_idx] = k
            d_void_ratio[d_idx] = void_ratio
            d_shear[d_idx] = g

    def _get_helpers_(self):
        return [yield_criterion_mcc, matrix_multiply]

# =============================================================================

class WaterPressure(Equation):
    def __init__(self, dest, sources, bulkw=2e9, sim_dim=2, drained=1):
        self.bulkw = bulkw
        self.sim_dim = sim_dim
        self.drained = drained
        super(UndrainedPressure, self).__init__(dest, sources)

    def initialize(self, d_idx, d_pwdt):
        d_pwdt[d_idx] = 0.0

    def loop(self, d_idx, d_pwdt, d_l_mat, s_idx, s_m, s_rho, VIJ, DWIJ):

        i, idx = declare("int", 2)
        dwij = declare("matrix(3)")
        l_mat = declare("matrix(9)")

        # Convert inv(L) matrix into a c-array
        idx = 9*d_idx
        for i in range(9):
            l_mat[i] = d_l_mat[idx + i]

        # Correct the kernel gradient
        matrix_multiply_vector(l_mat, DWIJ, dwij, 3)

        # Precompute quantities used to calculate bi, aii, and sum_aij*pj
        vj = s_m[s_idx] / s_rho[s_idx]

        # Compute divergence of velocity and position finite difference term
        vdw = 0.0

        for i in range(3):
            vdw -= VIJ[i] * dwij[i]

        divv = vj * vdw

        # Update pore pressure rate
        d_pwdt[d_idx] += self.bulkw * divv

    def post_loop(self, d_idx, d_pw, d_pwdt, dt):

        # Update pore pressure
        d_pw[d_idx] += d_pwdt[d_idx] * dt

        # No tensile pressure
        if d_pw[d_idx] > 0.0:
            d_pw[d_idx] = 0.0

    def _get_helpers_(self):
        return [matrix_multiply_vector]