from pysph.sph.equation import Equation
from compyle.api import declare
from matrix_operations import matrix_multiply_vector


class DeformationRates(Equation):
    def __init__(self, dest, sources):
        super(DeformationRates, self).__init__(dest, sources)

    def initialize(self, d_idx, d_eps_dot, d_spin_dot):
        i, idx = declare("int", 2)

        idx = 9*d_idx
        for i in range(9):
            d_eps_dot[idx + i] = 0.0
            d_spin_dot[idx + i] = 0.0

    def loop(self, d_idx, d_eps_dot, d_spin_dot, d_l_mat, s_idx, s_m, s_rho,
             VIJ, DWIJ):

        i, j, idx, iidx = declare("int", 4)
        dwij = declare("matrix(3)")
        grad_v, l_mat = declare("matrix(9)", 2)

        # Initialize grad_v and convert inv(L) matrix into a c-array
        idx = 9*d_idx
        for i in range(9):
            grad_v[i] = 0.0
            l_mat[i] = d_l_mat[idx + i]

        # Correct the kernel gradient
        matrix_multiply_vector(l_mat, DWIJ, dwij, 3)

        # Calculate the velocity gradient
        vj = s_m[s_idx] / s_rho[s_idx]
        for i in range(3):
            dvij = vj * -VIJ[i]
            for j in range(3):
                grad_v[3*i + j] += dvij * dwij[j]

        # Calculate strain rate and spin rate tensors
        for i in range(3):
            iidx = 3 * i
            for j in range(3):
                gv_ij = grad_v[iidx + j]
                gv_ji = grad_v[3*j + i]
                d_eps_dot[idx + iidx + j] += 0.5 * (gv_ij + gv_ji)
                d_spin_dot[idx + iidx + j] += 0.5 * (gv_ij - gv_ji)

    def _get_helpers_(self):
        return [matrix_multiply_vector]
