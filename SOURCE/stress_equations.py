from pysph.sph.equation import Equation
from compyle.api import declare
from math import sqrt

from matrix_operations import (matrix_multiply, matrix_determinant,
                               matrix_determinant_2x2, matrix_transpose)


class TrialStress(Equation):
    def __init__(self, dest, sources):
        super(TrialStress, self).__init__(dest, sources)

    def initialize(self, d_idx, d_sigma, d_sigma_tr, d_sigma_dot, d_eps_dot,
                   d_bulk, d_shear, dt):

        i, idx = declare("int", 2)
        deps_dev = declare("matrix(9)")
        idx = 9 * d_idx

        # Calculate volumetric strain increment
        deps_vol = d_eps_dot[idx] + d_eps_dot[idx + 4] + d_eps_dot[idx + 8]

        # Calculate deviatoric strain rate
        ep_vol = deps_vol / 3.0
        for i in range(9):
            deps_dev[i] = d_eps_dot[idx + i]
            if i % 4 == 0:
                deps_dev[i] -= ep_vol

        # Material elastic parameters
        k = d_bulk[d_idx]
        g = d_shear[d_idx]

        # Calculate trial stress state
        p_dot = k * deps_vol
        for i in range(9):
            dsig = 2 * g * deps_dev[i]
            if i % 4 == 0:
                dsig += p_dot
            d_sigma_tr[idx + i] = d_sigma[idx + i] + dsig * dt
            d_sigma_dot[idx + i] = dsig


class TrialStressDecomposition(Equation):
    def __init__(self, dest, sources):
        super(TrialStressDecomposition, self).__init__(dest, sources)

    def initialize(self, d_idx, d_sigma_tr, d_p, d_q, d_sigma_dev):
        i, idx = declare("int", 2)
        idx = 9 * d_idx

        # Hydrostatic stress
        p = (d_sigma_tr[idx] + d_sigma_tr[idx + 4] + d_sigma_tr[idx + 8]) / 3.0

        # Von Mises stress and deviatoric stress tensor
        d_p[d_idx] = p
        s2 = 0.0
        for i in range(9):
            s = d_sigma_tr[idx + i]
            if i % 4 == 0:
                s -= p
            s2 += s * s
            d_sigma_dev[idx + i] = s

        d_q[d_idx] = sqrt(3.0 * s2 / 2.0)


class MyStressRegularization(Equation):
    def __init__(self, dest, sources, freq=10):
        self.freq = freq
        super(MyStressRegularization, self).__init__(dest, sources)

    def initialize(self, d_idx, d_sigma, d_sigo, d_sigma_tr, d_sigma_dev):
        i, idx = declare("int", 2)
        idx = 9 * d_idx

        # Just to re-utilize these arrays and not create new ones to hold these
        #  temporary values
        for i in range(9):
            d_sigma_tr[idx + i] = d_sigma[idx + i]
            d_sigma_dev[idx + i] = d_sigo[idx + i]

    def loop(self, d_idx, d_sigma, d_sigo, d_sigma_tr, d_sigma_dev, d_wsum,
             s_idx, s_m, s_rho, s_sigma_tr, s_sigma_dev, WIJ, t, dt):

        i, idx, isx, f, n = declare("int", 5)

        if self.freq > 0:

            # Regularization frequency and step number
            f = int(self.freq)
            n = int(t / dt)

            if n % f == 0:
                idx = 9*d_idx
                isx = 9*s_idx
                wij = s_m[s_idx] * WIJ / (s_rho[s_idx] * d_wsum[d_idx])

                for i in range(9):
                    d_sigma[idx + i] += wij * (s_sigma_tr[isx + i] -
                                               d_sigma_tr[idx + i])
                    d_sigo[idx + i] += wij * (s_sigma_dev[isx + i] -
                                              d_sigma_dev[idx + i])


class StressRegularization(Equation):
    def __init__(self, dest, sources, freq=40):
        self.freq = freq
        super(StressRegularization, self).__init__(dest, sources)

    def initialize(self, d_idx, d_sigma, d_sigma_reg):
        i, idx = declare("int", 2)
        idx = 9 * d_idx

        for i in range(9):
            d_sigma_reg[idx + i] = d_sigma[idx + i]

    def loop(self, d_idx, d_sigma, d_sigma_reg, d_wsum, s_idx, s_m, s_rho,
             s_sigma, WIJ, t, dt):

        i, idx, isx, f, n = declare("int", 5)

        # Regularization frequency and step number
        f = int(self.freq)
        n = int(t / dt)

        if self.freq > 0 and n % f == 0:
            idx = 9 * d_idx
            isx = 9 * s_idx
            wij = s_m[s_idx] * WIJ / (s_rho[s_idx] * d_wsum[d_idx])

            for i in range(9):
                d_sigma_reg[idx + i] += (
                        wij * (s_sigma[isx + i] - d_sigma[idx + i])
                )

    def post_loop(self, d_idx, d_sigma, d_sigma_reg):
        i, idx = declare("int", 2)
        idx = 9 * d_idx

        for i in range(9):
            d_sigma[idx + i] = d_sigma_reg[idx + i]


class PwRegularization(Equation):
    def __init__(self, dest, sources, freq=40):
        self.freq = freq
        super(PwRegularization, self).__init__(dest, sources)

    def initialize(self, d_idx, d_pw, d_pw_reg):
        d_pw_reg[d_idx] = d_pw[d_idx]

    def loop(self, d_idx, d_pw, d_pw_reg, d_wsum, s_idx, s_m, s_rho, s_pw, WIJ,
             t, dt):
        f, n = declare("int", 2)
        f = int(self.freq)
        n = int(t / dt)

        wij = s_m[s_idx] * WIJ / (s_rho[s_idx] * d_wsum[d_idx])

        if self.freq > 0 and n % f == 0:
            d_pw_reg[d_idx] += wij * (s_pw[s_idx] - d_pw[d_idx])

    def post_loop(self, d_idx, d_pw, d_pw_reg):
        d_pw[d_idx] = d_pw_reg[d_idx]


class StressDiffusion(Equation):
    r"""
    Stress diffusion based on:

    Feng, R., Fourtakas, G., Rogers, B. D., & Lombardi, D. (2021).
    Large deformation analysis of granular materials with stabilized and
    noise-free stress treatment in smoothed particle hydrodynamics (SPH).
    Computers and Geotechnics, 138, 104356.
    https://doi.org/10.1016/j.compgeo.2021.104356

    Notes: This version uses the free surface identification of the particle
     shifting algorithm and just corrects the stress for non-free surface
     particles.

    Implemented: 02/01/2025
    Tested: 02/02/2025
    Status: The diffusion term removes a lot of the checkerboard pattern even
     for eta < 0.1 (as little as 0.001). However, it introduces errors in the
     deformation in the long run (like observed with stress smoothing). The
     larger the dissipation, the longer it takes, but always happens.
     Additionally, at some point, the simulation fails with particles becoming
     clusters, even with particle shifting on. Using the diffusion only every N
     steps like stress smoothing, does not seem to help either. (02/02/2025)
    """
    def __init__(self, dest, sources, h, c0, eta=0.001, sim_dim=2):
        self.h = h
        self.c0 = c0
        self.eta = eta
        self.sim_dim = sim_dim
        super(StressDiffusion, self).__init__(dest, sources)

    def initialize(self, d_idx, d_sig_diff):
        i, idx = declare("int", 2)
        idx = 9 * d_idx

        # Initialize diffusion stress rate
        for i in range(9):
            d_sig_diff[idx + i] = 0.0

    def loop(self, d_idx, d_sigma, d_pstype, d_sig_diff, s_idx, s_m, s_rho,
             s_sigma, XIJ, R2IJ, DWIJ, EPS):
        i, idx, jdx = declare("int", 3)

        # For inner particles only
        if d_pstype[d_idx] == 0:

            # Laplacian pre-calculation
            xdw = (
                    s_m[s_idx] *
                    (XIJ[0] * DWIJ[0] + XIJ[1] * DWIJ[1] + XIJ[2] * DWIJ[2]) /
                    (s_rho[s_idx] * R2IJ + EPS)
            )

            # Calculate diffusion stress rate
            idx = 9 * d_idx
            jdx = 9 * s_idx
            for i in range(9):
                d_sig_diff[idx + i] += (
                        (d_sigma[idx + i] - s_sigma[jdx + i]) * xdw
                )

    def post_loop(self, d_idx, d_sigma_dot, d_sig_diff):
        i, idx = declare("int", 2)
        idx = 9 * d_idx

        # Scaling coefficient pre-calculation
        scoeff = 2.0 * self.eta * self.h * self.c0

        for i in range(9):
            d_sigma_dot[idx + i] += scoeff * d_sig_diff[idx + i]
