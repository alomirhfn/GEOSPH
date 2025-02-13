from pysph.sph.equation import Equation
from compyle.api import declare
from math import sqrt, fabs
from matrix_operations import matrix_multiply_vector

r""" 
This file implements all formulations necessary to perform particle shifting.

The formulations implemented are based on the following work:
    - Zhang, S., Wang, F., Hu, X., & Lourenco, S.D.N. (2025). A Unified 
      Transport-Velocity Formulation for SPH Simulation of Cohesive Granular 
      Materials. Computers and Geotechnics (in press).
"""

class ParticleShiftPreCalcs(Equation):

    def __init__(self, dest, sources, sim_dim = 2):
        self.simdim = sim_dim
        super(ParticleShiftPreCalcs, self).__init__(dest, sources)

    def initialize(self, d_idx, d_divx, d_gradc, d_gfick, d_n, d_pstype):
        i, idx = declare("int", 2)

        d_divx[d_idx] = 0.0
        d_pstype[d_idx] = 0

        idx = 3 * d_idx
        for i in range(3):
            d_gradc[idx + i] = 0.0
            d_gfick[idx + i] = 0.0
            d_n[idx + i] = 0.0

    def loop(self, d_idx, d_gid, d_divx, d_gradc, d_gfick, d_l_mat, d_wdp,
             s_idx, s_gid, s_m, s_rho, XIJ, WIJ, DWIJ):
        """
        According to Michel et al. (2022) the gradient of concentration (or
         Fick's law) should point towards the region of highest concentration
         of particles.
        """
        i, idx = declare("int", 2)
        dwij = declare("matrix(3)")
        l_mat = declare("matrix(9)")

        if d_gid[d_idx] != s_gid[s_idx]:

            # Correct the kernel gradient
            idx = 9 * d_idx
            for i in range(9):
                l_mat[i] = d_l_mat[idx + i]
            matrix_multiply_vector(l_mat, DWIJ, dwij, 3)

            # Calculate position divergence, corrected concentration gradient
            #  (Sum d~W) for normal vector (n) calculations, and Fick's
            #  concentration gradient, i.e., uncorrected concentration gradient
            #  (Sum dW).
            vj = s_m[s_idx] / s_rho[s_idx]  # Part j's volume
            cj = vj * (1 + 0.2 * (WIJ / d_wdp[0]) ** 4)  # Concentration coeff.
            idx = 3 * d_idx
            for i in range(3):
                d_divx[d_idx] -= vj * XIJ[i] * DWIJ[i]
                d_gradc[idx + i] -= vj * dwij[i]  # For normal calculation
                d_gfick[idx + i] -= cj * dwij[i]  # For shifting direction

    def _get_helpers_(self):
        return [matrix_multiply_vector]


class ZhangParticleShift(Equation):

    def __init__(self, dest, sources, h0, simdim=2):
        self.h = h0
        self.simdim = simdim
        super(ZhangParticleShift, self).__init__(dest, sources)

    def initialize(self, d_idx, d_divx, d_pstype, d_gradc, d_n, d_phips):
        i, idx = declare("int", 2)
        idx = 3 * d_idx

        # Classify particle as free surface or not (1-FS, 0-Not FS, 2-Ext.)
        if d_divx[d_idx] < 1.0:
            d_pstype[d_idx] = 2
        elif d_divx[d_idx] < 0.75 * self.simdim:
            d_pstype[d_idx] = 1

        # === Calculate free surface normal for free surface particles only ===
        if d_pstype[d_idx] == 1:

            # Calculate the norm of the concentration gradient
            norm_gc = 0.0
            for i in range(3):
                norm_gc += d_gradc[idx + i] ** 2

            norm_gc = sqrt(norm_gc)

            # Calculate the particle normal
            for i in range(3):
                d_n[idx + i] = d_gradc[idx + i] / norm_gc
        # =====================================================================

        # Set FS/NFS identifier
        d_phips[d_idx] = 0.0
        if d_pstype[d_idx] == 1:  # If free surface particle
            d_phips[d_idx] = 1.0  # Removes normal shifting
        elif d_pstype[d_idx] == 2: # If external surface
            d_phips[d_idx] = -1.0  # Flag it

    def loop(self, d_idx, d_pstype, d_n, d_phips, s_idx, s_pstype, s_n, RIJ):
        """
        This loop is only used to identify near surface particles and their
        normal vectors.

        There is no need to check for inner particles as the condition of
         having a free surface particle neighbor will never be satisfied.
        """
        i, idx, jdx = declare("int", 3)

        # Check if neighbor is a free surface particle but not itself
        if d_pstype[d_idx] == 0 and s_pstype[s_idx] == 1:

            # The next few lines find nearest free surface particle. If
            #  distance between current FS neighbor and current NFS particle is
            #  smaller than the one previously calculated, assign the current
            #  FS neighbor as the closest one and its normal vector to the
            #  current NFS particle.
            if d_phips[d_idx] < RIJ:
                d_phips[d_idx] = RIJ

                # Assign FS neighbor's normal vector to current particle
                idx = 3 * d_idx
                jdx = 3 * s_idx
                for i in range(3):
                    d_n[idx + i] = s_n[jdx + i]

    def post_loop(self, d_idx, d_vps, d_n, d_gfick, dt, d_phips):
        """
        Implementation following Zhang et al. (2025)
        """
        i, idx = declare("int", 2)
        idx = 3 * d_idx

        # Calculate shifting velocity, vps, for inner particles
        coeff = 0.2 * (self.h ** 2) / dt
        if d_phips[d_idx] == 0.0:
            for i in range(3):
                d_vps[idx + i] = coeff * d_gfick[idx + i]

        # Correct shifting velocity for free and near free surface particles
        elif d_phips[d_idx] > 0.0:

            # Calculate normal component of the particle shift
            gf_n = 0.0
            for i in range(3):
                gf_n += d_gfick[idx + i] * d_n[idx + i]

            # Reduce/remove normal component for FS/NFS particles
            for i in range(3):
                d_vps[idx + i] = (
                        coeff * (d_gfick[idx + i] - gf_n * d_n[idx + i])
                )

        # Do not apply particle shifting to external particles
        else:
            for i in range(3):
                d_vps[idx + i] = 0.0
