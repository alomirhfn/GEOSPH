from pysph.sph.equation import Equation
from compyle.api import declare
from math import fabs


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

    def __init__(self, dest, sources):
        super(BoundaryStress, self).__init__(dest, sources)

    def initialize(self, d_idx, d_u, d_v, d_w, d_vb, d_sigma):
        i, idx = declare("int", 2)
        idx = 3 * d_idx

        # Initialize velocity
        d_u[d_idx] = d_vb[idx]
        d_v[d_idx] = d_vb[idx + 1]
        d_w[d_idx] = d_vb[idx + 2]

        # Initialize stress
        idx = 9 * d_idx
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


class MyDummyBoundary(Equation):
    r"""
        No-slip and free-slip boundary conditions proposed by Bui et al. and
        modified by Favero Neto.

        This is a variant of dummy boundary conditions that use three to four
        layers of particles fixed in space or moving with prescribed velocity
        $v_b$.

        For the no-slip BC, the velocity of boundary particles is the
        opposite of the smoothed average of internal neighbor particles'
        velocities, and the stress tensor is the smoothed average of the
        internal neighbor particles' stress tensors.

        .. math::

            u_b = -\sum_{a=1}^{N_a} V_a (v_a - 2v_b) \tilde{W}_{ba}

        .. math::

            \sigma_b = \sum_{a=1}^{N_a} V_a \sigma_a \tilde{W}_{ba}

        where the subscripts a and b refer to the internal and boundary
        particles, respectively, and

        .. math::

            \tilde{W}_{ba} = \sum_{a=1}^{N_a} V_a W_{ba}

        For free-slip BC, the normal component of  velocity for the boundary
        particles is the opposite of the smoothed average normal velocity of
        internal neighbor particles' velocities plus twice the boundary
        prescribed normal velocity, while the tangential component is the same.
        The stress tensor diagonal components are the average of the normal
        mean stress from neighbor particles, while off-diagonal are the
        opposite of those from neighbor particles. The factor of two was
        prescribed in Villodi and Ramachandran (2024).

        .. math::

            u_{b,n} = -\sum_{a=1}^{N_a} V_a (v_{a,n} - 2v_{b,n}) \tilde{W}_{ba}

        .. math::

            u_{b,t} = \sum_{a=1}^{N_a} V_a v_{a,t} \tilde{W}_{ba}

        where the subscripts n and t refer to normal and tangential components,
        respectively.

        .. math::

            \sigma_b =
            \begin{cases}
              \sum_{a=1}^{N_a} V_a p_a \tilde{W}_{ba}, & \text{if } i=j\\
              -\sum_{a=1}^{N_a} V_a \sigma_a^{ij} \tilde{W}_{ba}, & \text{else}
            \end{cases}

        where i, j refer to cartesian coordinates (indices) of the tensor.

        References
        ----------
        .. [Yangetal2020] E. Yang, H.H. Bui, H. De Sterck, G.D. Nguyen,
         A. Bouazza, "A scalable parallel computing SPH framework for
         predictions of geophysical granular flows", Computers and Geotechnics,
         121, 2020.
        .. Villodi, N., & Ramachandran, P. (2024). Robust solid boundary
        treatment for compressible smoothed particle hydrodynamics. Physics of
        Fluids, 36(8), 086130. https://doi.org/10.1063/5.0220606

    """

    def __init__(self, dest, sources, sim_dim=2, coeff=0.0, cf=0.0):
        self.sim_dim = sim_dim
        self.coeff = coeff  # Seems to work well with 0 up to 0.4
        self.cf = cf  # For vane simulations, use cf = 1.0 (0.0 otherwise)
        super(MyDummyBoundary, self).__init__(dest, sources)

    def initialize(self, d_idx, d_sigma, d_u, d_v, d_w, d_vb, d_bc, d_n,
                   d_type):
        i, idx = declare("int", 2)

        # Initialize stress
        idx = 9 * d_idx
        for i in range(9):
            d_sigma[idx + i] = 0.0

        # Initialize velocities since vb is the one used to move the boundary
        idx = 3 * d_idx

        # Boundary prescribed velocity components
        mod = 1 + self.cf
        vbx = mod * d_vb[idx]
        vby = mod * d_vb[idx + 1]
        vbz = mod * d_vb[idx + 2]

        # No-slip
        if d_bc[d_idx] == 0:
            d_u[d_idx] = vbx
            d_v[d_idx] = vby
            d_w[d_idx] = vbz

        # Free-slip
        else:

            # Normal vector components
            nx = d_n[idx]
            ny = d_n[idx + 1]
            nz = d_n[idx + 2]

            # Normal prescribed velocity
            vbn = vbx * nx + vby * ny + vbz * nz

            d_u[d_idx] = vbn * nx
            d_v[d_idx] = vbn * ny
            d_w[d_idx] = vbn * nz

    def loop(self, d_idx, d_u, d_v, d_w, d_sigma, d_wsum, d_bc, d_n, s_idx,
             s_m, s_rho, s_f, s_u, s_v, s_w, s_sigma, XIJ, WIJ):

        i, idx, jdx = declare("int", 3)
        jdx = 9 * s_idx
        wij = s_m[s_idx] * WIJ / (s_rho[s_idx] * d_wsum[d_idx])

        # Domain particle velocity vector components
        u = s_u[s_idx]
        v = s_v[s_idx]
        w = s_w[s_idx]

        if d_bc[d_idx] == 0:  # No-slip BC
            idx = 9 * d_idx

            # Extrapolate stress to the boundary particle
            for i in range(9):
                d_sigma[idx + i] += wij * s_sigma[jdx + i]

            # Extrapolate velocity to the boundary particles
            idx = 3 * d_idx
            nx = d_n[idx]
            ny = d_n[idx + 1]
            if abs(nx) > abs(ny):
                nr = abs(ny) / abs(nx)
            else:
                nr = abs(nx) / abs(ny)
            coeff = 1.0 - nr * (1.0 - self.coeff)

            d_u[d_idx] -= coeff * wij * u
            d_v[d_idx] -= coeff * wij * v
            d_w[d_idx] -= coeff * wij * w

        else:  # Free-slip BC
            idx = 3 * d_idx

            # Normal vector components
            nx = d_n[idx]
            ny = d_n[idx + 1]
            nz = d_n[idx + 2]

            # Enforce only normal traction condition
            sxx = s_sigma[jdx]
            syy = s_sigma[jdx + 4]
            szz = s_sigma[jdx + 8]
            sxy = s_sigma[jdx + 1]
            sxz = s_sigma[jdx + 2]
            syz = s_sigma[jdx + 5]

            # Traction vectors components
            tx = sxx * nx + sxy * ny + sxz * nz
            ty = sxy * nx + syy * ny + syz * nz
            tz = sxz * nx + syz * ny + szz * nz

            # Normal component of traction vector
            tn = tx * nx + ty * ny + tz * nz

            # Normal traction vector
            tnx = tn * nx
            tny = tn * ny
            tnz = tn * nz

            # Tangential traction vector
            ttx = tx - tnx
            tty = ty - tny
            ttz = tz - tnz

            # Final stress tensor
            idx = 9 * d_idx
            d_sigma[idx + 0] += wij * (sxx - ttx * nx)
            d_sigma[idx + 1] += wij * (sxy - ttx * ny)
            d_sigma[idx + 2] += wij * (sxz - ttx * nz)
            d_sigma[idx + 3] += wij * (sxy - tty * nx)
            d_sigma[idx + 4] += wij * (syy - tty * ny)
            d_sigma[idx + 5] += wij * (syz - tty * nz)
            d_sigma[idx + 6] += wij * (sxz - ttz * nx)
            d_sigma[idx + 7] += wij * (syz - ttz * ny)
            d_sigma[idx + 8] += wij * (szz - ttz * nz)

            # Normal component of extrapolated velocity
            vel = u * nx + v * ny + w * nz

            d_u[d_idx] += wij * (u - 2 * vel * nx)
            d_v[d_idx] += wij * (v - 2 * vel * ny)
            d_w[d_idx] += wij * (w - 2 * vel * nz)

        # For geostatic equilibrium
        idx = 9 * d_idx
        jdx = 3 * s_idx
        d_sigma[idx] -= s_f[jdx] * s_rho[s_idx] * XIJ[0] * wij
        d_sigma[idx + 4] -= s_f[jdx + 1] * s_rho[s_idx] * XIJ[1] * wij
        d_sigma[idx + 8] -= s_f[jdx + 2] * s_rho[s_idx] * XIJ[2] * wij
