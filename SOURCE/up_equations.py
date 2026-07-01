from pysph.sph.equation import Equation
from compyle.api import declare
from matrix_operations import matrix_multiply_vector
from math import cos, pi
import numpy as np


class UndrainedPressure(Equation):
    # TODO (05/20/2026): Because I calculate the position divergence for all
    #  particles for shifting, I can simplify this function a lot, and have
    #  everything in the initialize() method.
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
        idx = 9 * d_idx
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

    def post_loop(self, d_idx, d_pw, d_pwdt, d_divr):

        # If free surfaces are drained
        if self.drained == 1:

            # Ranges of Div(x) for free surface particle identification and
            #  pressure correction according to Chow et al. (2018).
            al = 1.6
            au = 1.9  # From paper, originally, au = 1.8
            if self.sim_dim == 3:
                al = 2.6
                au = 2.8

            # Pressure smoothing coefficient alpha from Skillen et al. (2013)
            dr = d_divr[d_idx]  # Position divergence
            if dr < al:
                a = 0.0
                d_pw[d_idx] = 0.0
            elif dr > au:
                a = 1.0
            else:
                a = 0.5 * (1 - cos(pi * (dr - al) / (au - al)))

            # Update pore pressure rate for free surface
            d_pwdt[d_idx] *= a

    def _get_helpers_(self):
        return [matrix_multiply_vector]
