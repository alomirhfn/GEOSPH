from pysph.sph.equation import Equation
from compyle.api import declare
from matrix_operations import (matrix_inverse_exact, eig_vals_3x3_analytical,
                               matrix_multiply_scalar, matrix_add, matrix_norm,
                               sign)


class KernelSum(Equation):
    def __init__(self, dest, sources, debug=0, bp=0):
        self.debug = debug
        self.bound_part = bp
        super(KernelSum, self).__init__(dest, sources)

    def initialize(self, d_idx, d_wsum):
        d_wsum[d_idx] = 0.0

    def loop(self, d_idx, d_wsum, s_idx, s_m, s_rho, WIJ):
        d_wsum[d_idx] += s_m[s_idx] * WIJ / s_rho[s_idx]

    def post_loop(self, d_idx, d_wsum, d_gid):
        if d_wsum[d_idx] == 0.0:
            d_wsum[d_idx] = 1.0


class KernelGradientSum(Equation):
    def __init__(self, dest, sources, debug=0):
        self.debug = debug
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
    def __init__(self, dest, sources, kgc=0, sim_dim=2, debug=0):
        self.calc = kgc
        self.dim = sim_dim
        self.debug = debug
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

    def post_loop(self, d_idx, d_gid, d_m_mat, d_l_mat):
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
        return [matrix_inverse_exact, eig_vals_3x3_analytical,
                matrix_multiply_scalar, matrix_add, matrix_norm, sign]
