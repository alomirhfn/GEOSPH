from pysph.sph.equation import Equation
from compyle.api import declare
from math import pow, sqrt
from matrix_operations import matrix_multiply_vector


class MomentumEquation1(Equation):
    r"""
        Balance of linear momentum

        This implementation follows the energy conjugate argument that it
        should be used alongside the continuity equation below, as proposed by
        Violeau (2002).
    """

    def __init__(self, dest, sources, sim_dim=2, sigma_c=0.0):
        self.sim_dim=sim_dim
        self.sigma_c=sigma_c
        super(MomentumEquation1, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_sigma, d_au, d_av, d_aw, d_wsum, s_idx, s_m,
             s_rho, s_sigma, DWIJ):

        i, j, idx, sidx = declare("int", 4)
        dvdt = declare("matrix(3)")

        for i in range(3):
            dvdt[i] = 0.0

        vj = s_m[s_idx] / (d_rho[d_idx] * s_rho[s_idx])
        idx = 9 * d_idx
        sidx = 9 * s_idx

        for i in range(3):
            for j in range(3):
                dvdt[i] += vj * (d_sigma[idx + 3*i + j] +
                                 s_sigma[sidx + 3*i + j]) * DWIJ[j]

        # ==================== CONFINING STRESS ====================

        # Correction for particles near the boundary only
        fi = 0.95
        if self.sim_dim == 3:
            fi = 0.9  # TODO: Value for 3D not checked (03/25/2024)

        if self.sigma_c != 0.0 and d_wsum[d_idx] <= fi:
            s_c = -2 * vj * self.sigma_c
            dvdt[0] += s_c * DWIJ[0]
            dvdt[1] += s_c * DWIJ[1]
            dvdt[2] += s_c * DWIJ[2]

        # ==========================================================

        # Update accelerations
        d_au[d_idx] += dvdt[0]
        d_av[d_idx] += dvdt[1]
        d_aw[d_idx] += dvdt[2]

        # TODO: Enrique's old version had artificial stress and gravity
        #  accelerations added here (09/2025).

    def _get_helpers_(self):
        return [matrix_multiply_vector]


class MomentumEquation2(Equation):
    r"""
        Balance of linear momentum

        This implementation follows the majority of geomechanics works in the
        literature, including most of the works by Bui, Nguyen, and Leung.
        """

    def __init__(self, dest, sources, sim_dim=2, sigma_c=0.0):
        self.sim_dim=sim_dim
        self.sigma_c=sigma_c
        super(MomentumEquation2, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_sigma, d_au, d_av, d_aw, s_idx, s_m, s_rho,
             s_sigma, DWIJ):

        i, j, idx, sidx = declare("int", 4)
        dvdt = declare("matrix(3)")

        for i in range(3):
            dvdt[i] = 0.0

        m = s_m[s_idx]
        rhoi2 = 1.0 / pow(d_rho[d_idx], 2.0)
        rhoj2 = 1.0 / pow(s_rho[s_idx], 2.0)
        idx = 9 * d_idx
        sidx = 9 * s_idx

        for i in range(3):
            for j in range(3):
                dvdt[i] += m * (d_sigma[idx + 3*i + j] * rhoi2 +
                                 s_sigma[sidx + 3*i + j] * rhoj2) * DWIJ[j]

        # ==================== CONFINING STRESS ====================
        vj = m / (d_rho[d_idx] * s_rho[s_idx])

        if self.sigma_c != 0.0:
            s_c = -2 * vj * self.sigma_c
            dvdt[0] += s_c * DWIJ[0]
            dvdt[1] += s_c * DWIJ[1]
            dvdt[2] += s_c * DWIJ[2]

        # ==========================================================

        # Update accelerations
        d_au[d_idx] += dvdt[0]
        d_av[d_idx] += dvdt[1]
        d_aw[d_idx] += dvdt[2]

        # TODO: Enrique's old version had artificial stress and gravity
        #  accelerations added here (09/2025).


class MomentumEquationPw(Equation):

    def __init__(self, dest, sources, sigma_c=0.0):
        self.sigma_c=sigma_c
        super(MomentumEquationPw, self).__init__(dest, sources)

    def initialize(self, d_idx, d_au, d_av, d_aw):
        d_au[d_idx] = 0.0
        d_av[d_idx] = 0.0
        d_aw[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_sigma, d_pw, d_au, d_av, d_aw, s_idx, s_m,
             s_rho, s_sigma, s_pw, DWIJ):

        i, j, idx, sidx = declare("int", 4)
        dvdt = declare("matrix(3)")

        for i in range(3):
            dvdt[i] = 0.0

        m = s_m[s_idx]
        rhoi2 = pow(d_rho[d_idx], 2.0)
        rhoj2 = pow(s_rho[s_idx], 2.0)
        vj = m / (d_rho[d_idx] * s_rho[s_idx])
        vpwij = vj * (s_pw[s_idx] + d_pw[d_idx])
        idx = 9 * d_idx
        sidx = 9 * s_idx

        for i in range(3):
            dvdt[i] += vpwij * DWIJ[i]
            for j in range(3):
                dvdt[i] += m * (d_sigma[idx + 3*i + j] / rhoi2 +
                                 s_sigma[sidx + 3*i + j] / rhoj2) * DWIJ[j]

        # ==================== CONFINING STRESS ====================

        if self.sigma_c != 0.0:
            s_c = -2 * vj * self.sigma_c
            dvdt[0] += s_c * DWIJ[0]
            dvdt[1] += s_c * DWIJ[1]
            dvdt[2] += s_c * DWIJ[2]

        # ==========================================================

        # Update accelerations
        d_au[d_idx] += dvdt[0]
        d_av[d_idx] += dvdt[1]
        d_aw[d_idx] += dvdt[2]


class DensityEquation(Equation):
    r"""
    Continuity equation to update particles' mass densities

    This for of the continuity equation is chosen due to its better
    characteristics when there are large mass or mass density differences in
    the soil (layered soil).

    \begin{equation}
        \frac{d\rho_i}{dt} = \rho_i \sum_{i=1}^N V_j (\boldsymbol{v}_i -
        \boldsymbol{v}_j) \cdot \nabla W_{ij}
    \end{equation}
    """
    def __init__(self, dest, sources):
        super(DensityEquation, self).__init__(dest, sources)

    def initialize(self, d_idx, d_arho):
        d_arho[d_idx] = 0.0

    def loop(self, d_idx, d_rho, d_arho, d_l_mat, s_idx, s_m, s_rho, VIJ,
             DWIJ):

        i, idx = declare("int", 2)
        dwij = declare("matrix(3)")
        l_mat = declare("matrix(9)")

        # Convert inv(L) matrix into a c-array
        idx = 9 * d_idx
        for i in range(9):
            l_mat[i] = d_l_mat[idx + i]

        # Correct the kernel gradient
        matrix_multiply_vector(l_mat, DWIJ, dwij, 3)

        div_vij = VIJ[0] * dwij[0] + VIJ[1] * dwij[1] + VIJ[2] * dwij[2]
        d_arho[d_idx] += d_rho[d_idx] * s_m[s_idx] * div_vij / s_rho[s_idx]

    def _get_helpers_(self):
        return [matrix_multiply_vector]


class SummationDensity(Equation):
    def __init__(self, dest, sources, freq=50):
        self.freq = freq
        super(SummationDensity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop(self, d_idx, d_gid, d_rho, d_wsum, s_idx, s_m, WIJ, t, dt):
        f, n = declare("int", 2)

        # Summation density calculation frequency and step number
        f = int(self.freq)
        n = int(t / dt)

        if self.freq > 0 and n % f == 0:
            d_rho[d_idx] += s_m[s_idx] * WIJ / d_wsum[d_idx]
