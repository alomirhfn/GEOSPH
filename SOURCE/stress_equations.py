from pysph.sph.equation import Equation
from compyle.api import declare
from math import sqrt
from matrix_operations import matrix_multiply_vector


class TrialStress(Equation):
    def __init__(self, dest, sources, debug=0):
        self.debug = debug
        super(TrialStress, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gid, d_sigma, d_sigma_tr, d_sigma_dot,
                   d_eps_dot, d_bulk, d_shear, dt):

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
    def __init__(self, dest, sources, debug=0):
        self.debug = debug
        super(TrialStressDecomposition, self).__init__(dest, sources)

    def initialize(self, d_idx, d_gid, d_sigma_tr, d_p, d_q, d_sigma_dev):
        i, idx = declare("int", 2)
        idx = 9*d_idx

        # Hydrostatic stress
        p = (d_sigma_tr[idx] + d_sigma_tr[idx + 4] + d_sigma_tr[idx + 8]) / 3.0

        # Von Mises stress and deviatoric stress tensor
        d_p[d_idx] = p
        s2 = 0.0
        for i in range(9):
            s = d_sigma_tr[idx + i]
            if i % 4 == 0:
                s -= p
            s2 += s*s
            d_sigma_dev[idx + i] = s

        d_q[d_idx] = sqrt(3.0*s2 / 2.0)


class StressRegularization(Equation):
    def __init__(self, dest, sources, freq=10, debug=0):
        self.freq = freq
        self.debug = debug
        super(StressRegularization, self).__init__(dest, sources)

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
