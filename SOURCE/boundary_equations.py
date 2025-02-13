from pysph.sph.equation import Equation
from compyle.api import declare

class BoundaryStress(Equation):
    r"""
        Boundary conditions proposed by Adami et al. 2012, where the stress of
        the boundary particles is a simple summation approximation of the
        stress tensor of surrounding neighbor particles within the problem
        domain.

        .. math::

            \sigma_b = \sum_{a=1}^{N_a} V_a \sigma_a \tilde{W}_{ba}

        where the subscripts a and b refer to the internal and boundary
        particles, respectively, and

        .. math::

            \tilde{W}_{ba} = \sum_{a=1}^{N_a} V_a W_{ba}

        References
        ----------
        .. [Adamietal2012] S.Adami, X.Y.Hu, N.A.Adams, "A generalized wall
        boundary condition for smoothed particle hydrodynamics", Journal of
        Computational Physics, 231(21), 2012, pp. 7057-7075.
    """

    def __init__(self, dest, sources, sim_dim=2, debug_bound=0):
        self.sim_dim = sim_dim
        self.debug_bound = debug_bound
        super(BoundaryStress, self).__init__(dest, sources)

    def initialize(self, d_idx, d_sigma):
        i, idx = declare("int", 2)
        idx = 9*d_idx

        for i in range(9):
            d_sigma[idx + i] = 0.0

    def loop(self, d_idx, d_sigma, d_wsum, s_idx, s_m, s_rho, s_f, s_sigma,
             XIJ, WIJ):
        i, idx, jdx = declare("int", 3)

        # Extrapolate stress to the boundary particle
        idx = 9 * d_idx
        jdx = 9 * s_idx
        wij = s_m[s_idx] * WIJ / (s_rho[s_idx] * d_wsum[d_idx])

        for i in range(9):
            d_sigma[idx + i] += wij * s_sigma[jdx + i]

        # For geostatic equilibrium
        jdx = 3 * s_idx
        d_sigma[idx] -= s_f[jdx] * s_rho[s_idx] * XIJ[0] * wij
        d_sigma[idx + 4] -= s_f[jdx + 1] * s_rho[s_idx] * XIJ[1] * wij
        d_sigma[idx + 8] -= s_f[jdx + 2] * s_rho[s_idx] * XIJ[2] * wij


class BoundaryPressure(Equation):
    r"""
        Extrapolates pore-water pressure to boundaries
    """

    def __init__(self, dest, sources, gravity=9.81, sim_dim=2):
        self.g = gravity
        self.sim_dim = sim_dim
        super(BoundaryPressure, self).__init__(dest, sources)

    def initialize(self, d_idx, d_pw):
       d_pw[d_idx] = 0.0

    def loop(self, d_idx, d_pw, d_wsum, s_idx, s_m, s_rho, s_pw, XIJ, WIJ):

        # Extrapolate stress to the boundary particle
        wij = s_m[s_idx] * WIJ / (s_rho[s_idx] * d_wsum[d_idx])
        d_pw[d_idx] += wij * s_pw[s_idx]

        # For geostatic equilibrium
        h = XIJ[1]
        if self.sim_dim == 3:
            h = XIJ[2]

        d_pw[d_idx] += wij * h * self.g * s_rho[s_idx]

    def post_loop(self, d_idx, d_pw):
        if d_pw[d_idx] > 0.0:
            d_pw[d_idx] = 0.0


class DummyBoundary(Equation):
    r"""
        Boundary conditions proposed by Bui and co-authors to impose free-slip
        and fixed boundary conditions using three to four layers of fixed
        particles.

        For the fixed BC, the velocity of boundary particles is the
        opposite of the smoothed average of internal neighbor particles'
        velocities, and the stress tensor is the smoothed average of the
        internal neighbor particles' stress tensors.

        .. math::

            v_b = -\sum_{a=1}^{N_a} V_a v_a \tilde{W}_{ba}

        .. math::

            \sigma_b = \sum_{a=1}^{N_a} V_a \sigma_a \tilde{W}_{ba}

        where the subscripts a and b refer to the internal and boundary
        particles, respectively, and

        .. math::

            \tilde{W}_{ba} = \sum_{a=1}^{N_a} V_a W_{ba}

        For free-slip BC, the normal component of  velocity for the boundary
        particles is the opposite of the smoothed average normal velocity of
        internal neighbor particles' velocities, while the tangential component
        is the same. The stress tensor diagonal components are the average of
        those from neighbor particles, while off-diagonal are the opposite.

        .. math::

            v_{b,n} = -\sum_{a=1}^{N_a} V_{a,n} v_a \tilde{W}_{ba}

        .. math::

            v_{b,t} = \sum_{a=1}^{N_a} V_{a,t} v_a \tilde{W}_{ba}

        where the subscripts n and t refer to normal and tangential components,
        respectively.

        .. math::

            \sigma_b =
            \begin{cases}
              \sum_{a=1}^{N_a}V_a\sigma_a^{ij}\tilde{W}_{ba},& \text{if } i=j\\
              -\sum_{a=1}^{N_a}V_a\sigma_a^{ij}\tilde{W}_{ba},& \text{else}
            \end{cases}

        where i, j refer to cartesian coordinates (indices) of the tensor.

        References
        ----------
        .. [Yangetal2020] E. Yang, H.H. Bui, H. De Sterck, G.D. Nguyen,
         A. Bouazza, "A scalable parallel computing SPH framework for
         predictions of geophysical granular flows", Computers and Geotechnics,
         121, 2020.
    """

    def __init__(self, dest, sources, sim_dim=2, debug_bound=0):
        self.sim_dim = sim_dim
        self.debug_bound = debug_bound
        super(DummyBoundary, self).__init__(dest, sources)

    def initialize(self, d_idx, d_sigma, d_u, d_v, d_w, d_vb):
        i, idx = declare("int", 2)

        # Initialize stress
        idx = 9*d_idx
        for i in range(9):
            d_sigma[idx + i] = 0.0

        # Initialize velocities. The factor of two multiplying the actual
        #  boundary particle velocity is to enforce no-penetration. When
        #  boundaries are moving, without that, particles were penetrating the
        #  boundaries.
        #  TODO: VALIDATE (07/31/2023)
        idx = 3 * d_idx
        d_u[d_idx] = 2 * d_vb[idx]
        d_v[d_idx] = 2 * d_vb[idx + 1]
        d_w[d_idx] = 2 * d_vb[idx + 2]

    def loop(self, d_idx, d_u, d_v, d_w, d_sigma, d_wsum, d_bc, d_n, s_idx,
             s_m, s_rho, s_f, s_u, s_v, s_w, s_sigma, XIJ, WIJ):

        i, idx, jdx = declare("int", 3)

        idx = 9 * d_idx
        jdx = 9 * s_idx
        wij = s_m[s_idx] * WIJ / (s_rho[s_idx] * d_wsum[d_idx])
        u = s_u[s_idx]  # Velocity vector components
        v = s_v[s_idx]
        w = s_w[s_idx]

        if d_bc[d_idx] == 0:  # No-slip BC

            # Extrapolate stress to the boundary particle
            for i in range(9):
                d_sigma[idx + i] += wij * s_sigma[jdx + i]

            # Extrapolate velocity to the boundary particles
            d_u[d_idx] -= wij * u
            d_v[d_idx] -= wij * v
            d_w[d_idx] -= wij * w

        else:  # Free-slip BC

            # Extrapolate stress to the boundary particle
            for i in range(9):
                if i % 4 == 0:
                    d_sigma[idx + i] += wij * s_sigma[jdx + i]
                else:
                    d_sigma[idx + i] -= wij * s_sigma[jdx + i]

            # Extrapolate velocity to the boundary particles
            idx = 3*d_idx
            nx = d_n[idx]  # Normal vector components
            ny = d_n[idx + 1]
            nz = d_n[idx + 2]
            vel = u * nx + v * ny + w * nz  # Normal velocity norm

            d_u[d_idx] += wij * (u - 2 * vel * nx)
            d_v[d_idx] += wij * (v - 2 * vel * ny)
            d_w[d_idx] += wij * (w - 2 * vel * nz)

        # For geostatic equilibrium
        idx = 9 * d_idx
        jdx = 3 * s_idx
        d_sigma[idx] -= s_f[jdx] * s_rho[s_idx] * XIJ[0] * wij
        d_sigma[idx + 4] -= s_f[jdx + 1] * s_rho[s_idx] * XIJ[1] * wij
        d_sigma[idx + 8] -= s_f[jdx + 2] * s_rho[s_idx] * XIJ[2] * wij
