import cython
from textwrap import dedent
from pysph.sph.equation import Equation
from compyle.api import declare
from matrix_operations import matrix_multiply_vector


class MonaghanArtificialViscosity(Equation):
    r"""
    Classical Monaghan artificial viscosity [Monaghan, 2005]_

    .. math::

        \frac{d\mathbf{v}_a}{dt} = -\sum_{b}m_{b}\Pi_{ab}\nabla_{a}W_{ab}

    where

    .. math::

        \Pi_{ab}=\begin{cases}\frac{-\alpha_{\pi}\bar{c}_{ab}\phi_{ab}+
        \beta_{\pi}\phi_{ab}^{2}}{\bar{\rho}_{ab}}, & \mathbf{v}_{ab}\cdot
        \mathbf{r}_{ab}<0\\0, & \mathbf{v}_{ab}\cdot\mathbf{r}_{ab}\geq0
        \end{cases}

    with

    .. math::

        \phi_{ab}=\frac{h\mathbf{v}_{ab}\cdot\mathbf{r}_{ab}}
        {|\mathbf{r}_{ab}|^{2}+\epsilon^{2}}\\

        \bar{c}_{ab}&=&\frac{c_{a}+c_{b}}{2}\\

        \bar{\rho}_{ab}&=&\frac{\rho_{a}+\rho_{b}}{2}

    References
    ----------
    .. [Monaghan2005] J. Monaghan, "Smoothed particle hydrodynamics",
        Reports on Progress in Physics, 68 (2005), pp. 1703-1759.
    """

    def __init__(self, dest, sources, alpha=0.5, beta=0.0, debug=0):
        r"""
        Parameters
        ----------
        alpha : float
            produces a shear and bulk viscosity
        beta : float
            used to handle high Mach number shocks
        """
        self.alpha = alpha
        self.beta = beta
        self.debug = debug
        super(MonaghanArtificialViscosity, self).__init__(dest, sources)

    def initialize(self, d_idx, d_av_a):
        idx = declare("int")
        idx = 3 * d_idx
        d_av_a[idx] = 0.0
        d_av_a[idx + 1] = 0.0
        d_av_a[idx + 2] = 0.0

    def loop(self, d_idx, d_av_a, d_cs, d_l_mat, s_idx, s_m, s_cs, VIJ, XIJ,
             HIJ, R2IJ, RHOIJ1, EPS, DWIJ):
        i, idx = declare("int", 2)
        dwij = declare("matrix(3)")
        l_mat = declare("matrix(9)")

        if self.alpha != 0.0 or self.beta != 0.0:

            # Tests if particles are approaching or departing from each other.
            v_rel = VIJ[0] * XIJ[0] + VIJ[1] * XIJ[1] + VIJ[2] * XIJ[2]

            if v_rel < 0:

                # Convert inv(L) matrix into a c-array
                idx = 9 * d_idx
                for i in range(9):
                    l_mat[i] = d_l_mat[idx + i]

                # Correct the kernel gradient
                matrix_multiply_vector(l_mat, DWIJ, dwij, 3)

                # Average sound speed velocity
                cij = 0.5 * (d_cs[d_idx] + s_cs[s_idx])

                muij = (HIJ * v_rel) / (R2IJ + EPS)
                piij = ((self.alpha * cij * muij - self.beta * muij * muij) *
                        RHOIJ1)
                m_p = s_m[s_idx] * piij

                idx = 3 * d_idx
                d_av_a[idx] += m_p * dwij[0]
                d_av_a[idx + 1] += m_p * dwij[1]
                d_av_a[idx + 2] += m_p * dwij[2]

    def _get_helpers_(self):
        return [matrix_multiply_vector]


class PySPHArtificialStress(Equation):
    r"""
    **Artificial stress to remove tensile instability**

    This implementation is the default one in the PySPH framework.

    The dispersion relations in [Gray2001] are used to determine the
    different components of :math:`R`.
    Angle of rotation for particle :math:`a`

    .. math::
        \tan{2 \theta_a} = \frac{2\sigma_a^{xy}}{\sigma_a^{xx} - \sigma_a^{yy}}

    In rotated frame, the new components of the stress tensor are

    .. math::
        \bar{\sigma}_a^{xx} = \cos^2{\theta_a} \sigma_a^{xx} + 2\sin{\theta_a}
        \cos{\theta_a}\sigma_a^{xy} + \sin^2{\theta_a}\sigma_a^{yy}\\
        \bar{\sigma}_a^{yy} = \sin^2{\theta_a} \sigma_a^{xx} + 2\sin{\theta_a}
        \cos{\theta_a}\sigma_a^{xy} + \cos^2{\theta_a}\sigma_a^{yy}

    Components of :math:`R` in rotated frame:

    .. math::
        \bar{R}_{a}^{xx}=\begin{cases}-\epsilon\frac{\bar{\sigma}_{a}^{xx}}
        {\rho^{2}} & \bar{\sigma}_{a}^{xx}>0\\0 & \bar{\sigma}_{a}^{xx}\leq0
        \end{cases}\\
        \bar{R}_{a}^{yy}=\begin{cases}-\epsilon\frac{\bar{\sigma}_{a}^{yy}}
        {\rho^{2}} & \bar{\sigma}_{a}^{yy}>0\\0 & \bar{\sigma}_{a}^{yy}\leq0
        \end{cases}

    Components of :math:`R` in original frame:

    .. math::
        R_a^{xx} = \cos^2{\theta_a} \bar{R}_a^{xx} +
        \sin^2{\theta_a} \bar{R}_a^{yy}\\
        R_a^{yy} = \sin^2{\theta_a} \bar{R}_a^{xx} +
        \cos^2{\theta_a} \bar{R}_a^{yy}\\
        R_a^{xy} = \sin{\theta_a} \cos{\theta_a}\left(\bar{R}_a^{xx} -
        \bar{R}_a^{yy}\right)
    """

    def __init__(self, dest, sources, as_eps=0.2, dp=0.1, debug=0):
        r"""
        Parameters
        ----------
        eps : float
            constant
        """
        self.eps = as_eps
        self.dp = dp
        self.debug = debug
        super(PySPHArtificialStress, self).__init__(dest, sources)

    def _cython_code_(self):
        code = dedent(
            """
            cimport cython
            from pysph.base.linalg3 cimport eigen_decomposition
            from pysph.base.linalg3 cimport transform_diag_inv
            """
        )
        return code

    def initialize(self, d_idx, d_rho, d_sigma, d_h, d_as_a, d_wdp, d_asig,
                   d_gid, SPH_KERNEL):
        r"""

        :param d_idx:
        :param d_rho:
        :param d_sigma:
        :param d_h:
        :param d_as_a:
        :param d_wdp:
        :param d_asig:
        :param d_gid:
        :param SPH_KERNEL:
        :return:
        """
        i, j, idx = declare("int", 3)
        xij = declare("matrix(3)")
        r = declare('matrix((3,3))')  # Matrix of Eigenvectors (columns)
        rab = declare('matrix((3,3))')  # Artificial stress
        s = declare('matrix((3,3))')  # Stress tensor with pressure.
        v = declare('matrix((3,))')  # Eigenvalues
        rd = declare('matrix((3,))')  # Artificial stress principal directions

        # Initialize artificial stress and acceleration tensors
        idx = 9*d_idx
        for i in range(3):
            d_as_a[3*d_idx + i] = 0.0
            for j in range(3):
                d_asig[idx + 3*i + j] = 0.0

        # Check if material has tensile strength and artificial stress active
        if self.eps > 0.0:

            # initialize variables
            for i in range(3):
                xij[i] = 0.0

            # Calculate kernel at dp (wdp) and the multiplier exponent (n)
            xij[0] = self.dp
            d_wdp[d_idx] = SPH_KERNEL.kernel(xij, self.dp, d_h[d_idx])

            # 1/rho^2
            rho = d_rho[d_idx]
            rho21 = 1.0 / (rho*rho)

            # Initialize the temporary Cauchy stress tensor
            for i in range(3):
                for j in range(3):
                    s[i][j] = d_sigma[idx + 3*i + j]

            # compute the principle stresses
            eigen_decomposition(s, r, cython.address(v[0]))

            # artificial stress corrections
            for i in range(3):
                if v[i] > 0:
                    rd[i] = -self.eps*v[i]*rho21
                else:
                    rd[i] = 0

            # transform artificial stresses in original frame
            transform_diag_inv(cython.address(rd[0]), r, rab)

            # store the values
            for i in range(3):
                for j in range(3):
                    d_asig[idx + 3*i + j] = rab[i][j]

            if d_gid[d_idx] == self.debug and self.eps > 0.0:
                printf("\n")
                printf("=== Artificial Stress Calculations ===\n")
                printf("\n")
                printf("Part i: %d\n", d_gid[d_idx])
                printf("\n")
                printf("Sigma\n")
                printf("%.16e %.16e %.16e\n", d_sigma[idx], d_sigma[idx + 1],
                       d_sigma[idx + 2])
                printf("%.16e %.16e %.16e\n", d_sigma[idx + 3],
                       d_sigma[idx + 4], d_sigma[idx + 5])
                printf("%.16e %.16e %.16e\n", d_sigma[idx + 6],
                       d_sigma[idx + 7], d_sigma[idx + 8])
                printf('\n')
                printf("Eigenvalues PySPH\n")
                printf("%.16e %.16e %.16e\n", v[0], v[1], v[2])
                printf('\n')
                printf("Modified Eigenvalues PySPH\n")
                printf("%.16e %.16e %.16e\n", rd[0], rd[1], rd[2])
                printf('\n')
                printf("Eigenvector matrix\n")
                printf("\n")
                printf("%.16e %.16e %.16e\n", r[0][0], r[0][1], r[0][2])
                printf("%.16e %.16e %.16e\n", r[1][0], r[1][1], r[1][2])
                printf("%.16e %.16e %.16e\n", r[2][0], r[2][1], r[2][2])
                printf('\n')
                printf("Corrected Stress Matrix\n")
                printf("\n")
                printf("%.16e %.16e %.16e\n", rab[0][0], rab[0][1], rab[0][2])
                printf("%.16e %.16e %.16e\n", rab[1][0], rab[1][1], rab[1][2])
                printf("%.16e %.16e %.16e\n", rab[2][0], rab[2][1], rab[2][2])
                printf("\n")
                printf("Part i: %d\n", d_gid[d_idx])
                printf("\n")
                printf("=== Artificial Stress Acceleration ===\n")
                printf("\n")
                printf("%.9f %.9f %.9f\n", d_as_a[3*d_idx],
                       d_as_a[3*d_idx + 1],
                       d_as_a[3*d_idx + 2])
                printf("====================================\n")

    def loop(self, d_idx, d_asig, d_as_a, s_idx, s_m, s_asig, d_wdp, RIJ, WIJ,
             DWIJ):
        r"""
        :param d_idx:
        :param d_asig:
        :param d_as_a:
        :param s_idx:
        :param s_m:
        :param s_asig:
        :param d_wdp:
        :param RIJ:
        :param WIJ:
        :param DWIJ:
        :return:
        """
        i, j, idx, isx = declare("int", 4)
        dvdt = declare("matrix(3)")

        if self.eps > 0.0 and RIJ < self.dp:

            # initialize variables
            for i in range(3):
                dvdt[i] = 0.0

            # Calculate scaling factor, d = fij^n
            n_exp = 2.55
            d = pow(WIJ / d_wdp[d_idx], n_exp)

            # Add artificial stress contribution to the accelerations
            idx = 9*d_idx
            isx = 9*s_idx
            for i in range(3):
                for j in range(3):
                    dvdt[i] += (d_asig[idx + 3*i + j] +
                                s_asig[isx + 3*i + j]) * DWIJ[j]

            dm = d * s_m[s_idx]
            idx = 3*d_idx
            for i in range(3):
                d_as_a[idx + i] += dm * dvdt[i]

    def post_loop(self, d_idx, d_gid, d_as_a):
        if d_gid[d_idx] == self.debug and self.eps > 0:
            printf("\n")
            printf("=== Acceleration with artificial stress ===\n")
            printf("%.9f %.9f %.9f\n", d_as_a[3*d_idx], d_as_a[3*d_idx + 1],
                   d_as_a[3*d_idx + 2])
            printf("====================================\n")
