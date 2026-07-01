from pysph.sph.equation import Equation
from compyle.api import declare
from pysph.sph.wc.linalg import gj_solve, augmented_matrix
from matrix_operations import (matrix_multiply_vector, full_matrix_inverse,
                               matrix_inverse_exact, matrix_add,
                               matrix_multiply_scalar, matrix_norm, sign,
                               eig_vals_3x3_analytical)
from math import sqrt


class KernelSum(Equation):
    def __init__(self, dest, sources):
        super(KernelSum, self).__init__(dest, sources)

    def initialize(self, d_idx, d_wsum):
        d_wsum[d_idx] = 0.0

    def loop(self, d_idx, d_wsum, s_idx, s_m, s_rho, WIJ):
        d_wsum[d_idx] += s_m[s_idx] * WIJ / s_rho[s_idx]

    def post_loop(self, d_idx, d_wsum):
        if d_wsum[d_idx] == 0.0:
            d_wsum[d_idx] = 1.0


class KernelGradientSum(Equation):
    def __init__(self, dest, sources):
        super(KernelGradientSum, self).__init__(dest, sources)

    def initialize(self, d_idx, d_dwsum):
        i, idx = declare("int", 2)

        idx = 3*d_idx
        for i in range(3):
            d_dwsum[idx + i] = 0.0

    def loop(self, d_idx, d_dwsum, s_idx, s_m, s_rho, DWIJ):
        i, idx = declare("int", 2)

        vj = s_m[s_idx] / s_rho[s_idx]
        idx = 3 * d_idx
        for i in range(3):
            d_dwsum[idx + i] += vj * DWIJ[i]


class KernelGradientCorrection(Equation):
    r"""**Kernel Gradient Correction**

        From [BonetLok1999], equations (42) and (45)

        .. math::
                \nabla \tilde{W}_{ab} = L_{a}\nabla W_{ab}

        .. math::
                L_{a} = \left(\sum \frac{m_{b}}{\rho_{b}} \nabla W_{ab}
                \mathbf{\otimes}x_{ba} \right)^{-1}
        """
    def __init__(self, dest, sources, kgc=0, sim_dim=2):
        self.calc = kgc
        self.dim = sim_dim
        super(KernelGradientCorrection, self).__init__(dest, sources)

    def initialize(self, d_idx, d_m_mat, d_l_mat):
        i, idx = declare('int', 2)

        idx = 9 * d_idx
        for i in range(9):
            d_m_mat[idx + i] = 0.0
            d_l_mat[idx + i] = 0.0

            if i % 4 == 0:
                d_l_mat[idx + i] = 1.0

    def loop(self, d_idx, d_m_mat, s_idx, s_m, s_rho, XIJ, DWIJ, RIJ):
        i, j, idx = declare("int", 3)

        if self.calc and RIJ > 1.0e-12:  # 1e-12 from PySPH docs
            idx = 9 * d_idx
            vj = s_m[s_idx] / s_rho[s_idx]
            for i in range(3):
                for j in range(3):
                    d_m_mat[idx + 3*i + j] -= vj * XIJ[i] * DWIJ[j]

    def post_loop(self, d_idx, d_m_mat, d_l_mat):
        i, idx = declare("int", 2)
        dbg_vec = declare("matrix(3)")
        m_mat, l_mat = declare('matrix(9)', 2)
        idx = 9 * d_idx

        if self.calc:

            if self.dim == 2:
                d_m_mat[idx + 8] = 1.0

            for i in range(9):
                m_mat[i] = d_m_mat[idx + i]

            # Invert the correction matrix
            matrix_inverse_exact(m_mat, l_mat, 3, dbg_vec)

            # Assign it to internal variable
            for i in range(9):
                d_l_mat[idx + i] = l_mat[i]

        # When no correction is performed the L matrix needs to be identity
        else:
            dbg_vec[1] = 1.0
            dbg_vec[2] = 0.0

    def _get_helpers_(self):
        return [matrix_inverse_exact, matrix_multiply_scalar, matrix_add, sign,
                matrix_norm, eig_vals_3x3_analytical]


class PySPHGradientCorrection(Equation):
    r"""
    **Kernel Gradient Correction from PySPH documentation**

    From [BonetLok1999], equations (42) and (45)

    .. math::
            \nabla \tilde{W}_{ab} = L_{a}\nabla W_{ab}

    .. math::
            L_{a} = \left(\sum \frac{m_{b}}{\rho_{b}} \nabla W_{ab}
            \mathbf{\otimes}x_{ba} \right)^{-1}

    NOTE (03/26/2024): The default tolerance of the gradient correction was
    0.1 (i.e., ~10% of correction), which is enough to correct the kernel
    gradient of all particles in a regular grid, except the outermost layer.
    However, I increased that value to 1.0 which still does not correct the
    outermost layer, but it is just shy of that threshold (found to be ~1.02),
    while providing a little bit more correction capability.
    """

    def __init__(self, dest, sources, kgc=0, sim_dim=2, tol=1.0, debug=0):
        self.calc = kgc
        self.dim = sim_dim
        self.tol = tol
        self.debug = debug
        super(PySPHGradientCorrection, self).__init__(dest, sources)

    def loop(self, d_idx, d_m_mat, DWIJ, EPS):
        i, j, idx = declare('int', 3)
        temp = declare('matrix(12)')
        res = declare('matrix(3)')

        if self.calc:
            idx = 9 * d_idx

            for i in range(3):
                for j in range(3):
                    temp[4 * i + j] = d_m_mat[idx + 3 * i + j]
                temp[4 * i + 3] = DWIJ[i]  # Augmented part of matrix

            gj_solve(temp, 3, 1, res)

            # Control variables to check kernel gradient correction results
            res_mag = 0.0
            dwij_mag = 0.0
            for i in range(3):
                res_mag += abs(res[i])
                dwij_mag += abs(DWIJ[i])

            # Some sort of check to control over-correction
            change = abs(res_mag - dwij_mag) / (dwij_mag + EPS)
            if change < self.tol:
                for i in range(3):
                    DWIJ[i] = res[i]

    def _get_helpers_(self):
        return [gj_solve]


class KernelSumSmooth(Equation):
    def __init__(self, dest, sources):
        super(KernelSumSmooth, self).__init__(dest, sources)

    def initialize(self, d_idx, d_wsum_pl):
        d_wsum_pl[d_idx] = 0.0

    def loop(self, d_idx, d_wsum_pl, s_idx, s_m, s_rho, WIJ):
        d_wsum_pl[d_idx] += s_m[s_idx] * WIJ / s_rho[s_idx]

    def post_loop(self, d_idx, d_wsum_pl):
        if d_wsum_pl[d_idx] == 0.0:
            d_wsum_pl[d_idx] = 1.0


###############################################################################
##                           MLS KERNEL CORRECTION                           ##
###############################################################################

class MLSKernelCorrection(Equation):
    r"""
        ** MLS Kernel Correction **

        This class implements the kernel correction to obtain zero- and
        first-order consistency of the kernel interpolation based on the moving
        least square (MLS) formulation presented in Chow et al. (2018).

        References
        ----------
        .. [Chowetal2018]
        A.D. Chow, B.D. Rogers, S.J. Lind, P.K. Stansby, "Incompressible SPH
        (ISPH) with fast Poisson solver on a GPU," "Computer Physics Comms.,"
        226, 81–103, https://doi.org/10.1016/j.cpc.2018.01.005.

    """
    def __init__(self, dest, sources, sim_dim=2):
        self.sim_dim = sim_dim
        super(MLSKernelCorrection, self).__init__(dest, sources)

    def initialize(self, d_idx, d_mls):
        i, idx = declare("int", 2)
        idx = 16 * d_idx

        # Initialize correction MLS matrix
        for i in range(16):
            d_mls[idx + i] = 0.0

    def loop(self, d_idx, d_mls, s_idx, s_m, s_rho, XIJ, WIJ):

        i, j, idx = declare("int", 3)
        bij = declare("matrix(4)")

        # Initialize position vector
        bij[0] = 1.0
        for i in range(3):
            bij[i + 1] = XIJ[i]

        # Auxiliary quantities
        vwij = s_m[s_idx] * WIJ / s_rho[s_idx]

        # Calculate MLS correction matrix Mij
        idx = 16 * d_idx
        for i in range(4):
            for j in range(4):
                d_mls[idx + 4*i + j] += bij[i] * bij[j] * vwij

    def post_loop(self, d_idx, d_mls, d_amls):
        i, j, idx = declare("int", 3)
        res, e = declare("matrix(4)", 2)
        mij = declare("matrix(16)")
        aug_mij = declare("matrix(20)")

        # Convert Mmls to a c-vector
        idx = 16 * d_idx
        for i in range(4):
            for j in range(4):
                mij[4*i + j] = d_mls[idx + 4*i + j]

        # Correction for 2D problem to make diagonal term non-zero
        if self.sim_dim == 2:
            mij[15] = 1.0

        # Define vector e = [1,0,0,0]
        e[0] = 1.0
        e[1] = 0.0
        e[2] = 0.0
        e[3] = 0.0

        # Augmented part of matrix Mmls to use "gj_solve"
        augmented_matrix(mij, e, 4, 1, 4, aug_mij)

        # Invert Mmls and multiply it by e = [1,0,0,0]
        gj_solve(aug_mij, 4, 1, res)

        # Final MLS correction vector
        idx = 4 * d_idx
        for i in range(4):
            d_amls[idx + i] = res[i]

    def _get_helpers_(self):
        return [gj_solve, augmented_matrix]


class MLSKernelCorrectionPySPH(Equation):
    r"""
    This is an adapted version of the code available in the source code of
      PySPH
    """
    def __init__(self, dest, sources, sim_dim=2):
        self.sim_dim = sim_dim
        super(MLSKernelCorrectionPySPH, self).__init__(dest, sources)

    def _get_helpers_(self):
        return [gj_solve, augmented_matrix, full_matrix_inverse,
                matrix_multiply_vector]

    def loop_all(self, d_idx, d_amls, d_x, d_y, d_z, s_x, s_y, s_z, d_h, s_h,
                 s_m, s_rho, SPH_KERNEL, NBRS, N_NBRS):

        i, j, k, n, s_idx, idx, value = declare('int', 7)
        xij = declare('matrix(3)')
        res, res2, uvec, uvec2 = declare('matrix(4)', 4)
        mls, mlsinv = declare('matrix(16)', 2)
        aug_mls = declare('matrix(20)')

        # Current particle's position
        x = d_x[d_idx]
        y = d_y[d_idx]
        z = d_z[d_idx]

        # Initialize MLS correction matrix and vector
        for i in range(16):
            mls[i] = 0.0

        # Calculate MLS correction matrix
        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij[0] = x - s_x[s_idx]
            xij[1] = y - s_y[s_idx]
            xij[2] = z - s_z[s_idx]
            rij = sqrt(xij[0] * xij[0] + xij[1] * xij[1] + xij[2] * xij[2])
            hij = 0.5 * (d_h[d_idx] + s_h[s_idx])
            wij = SPH_KERNEL.kernel(xij, rij, hij)
            vj = s_m[s_idx] / s_rho[s_idx]

            for j in range(4):
                if j == 0:
                    fac1 = 1.0
                else:
                    fac1 = xij[j - 1]
                for k in range(4):
                    if k == 0:
                        fac2 = 1.0
                    else:
                        fac2 = xij[k - 1]
                    mls[4*j + k] += fac1 * fac2 * vj * wij

        # Unit vector, e_1
        res[0] = 1.0
        res[1] = 0.0
        res[2] = 0.0
        res[3] = 0.0

        # Correct MLS matrix for 2D problem
        n = 4  # Size of matrix to invert
        if self.sim_dim == 2:
            n = 3
            mls[15] = 1.0

        # Truncate values of MLS matrix to avoid singularity
        for i in range(16):
            value = int(mls[i] * 1e14 + 0.5)
            mls[i] = 1.0 * value / 1.0e14

        # Invert MLS correction Matrix and multiply it by the unit vector
        augmented_matrix(mls, res, n, 1, aug_mls)

        # Use two different methods to invert the matrix to avoid singularity
        gj_solve(aug_mls, n, 1, res)
        full_matrix_inverse(mls,mlsinv,4)

        if self.sim_dim == 2:
            res[3] = 0.0

        # Calculate the second correction kernel vector
        for i in range(4):
            res2[i] = mlsinv[i]

        # Check for the correction vector with better consistency
        matrix_multiply_vector(mls,res,uvec,4)
        matrix_multiply_vector(mls,res2,uvec2,4)

        # Norm of the unit vector
        normres = sqrt(res[0]*res[0] + res[1]*res[1] + res[2]*res[2])
        normuvec = sqrt(uvec[0]*uvec[0] + uvec[1]*uvec[1] + uvec[2]*uvec[2])
        normuvec2 = sqrt(uvec2[0]*uvec2[0] + uvec2[1]*uvec2[1] +
                        uvec2[2]*uvec2[2])

        # Checks to make sure uvec2 is not NaN
        if normuvec2 > normuvec:
            if abs(normres - 1.0) < 1e-6:
                for i in range(4):
                    res[i] = res2[i]

        idx = 4*d_idx
        for i in range(4):
            d_amls[idx + i] = res[i]
