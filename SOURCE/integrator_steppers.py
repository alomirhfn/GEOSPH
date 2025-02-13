from pysph.sph.integrator_step import IntegratorStep
from math import sqrt, cos, sin, pi, atan
from compyle.api import declare


# =============================================================================
# ===================== FORWARD EULER INTEGRATOR STEPPERS =====================
# =============================================================================

class DomainSingleEulerStep(IntegratorStep):

    def __init__(self, sim_type=0, damp_eps=0.01, damp_time=0.0):
        self.sim_type = sim_type
        self.damp_eps = damp_eps
        self.damp_time = damp_time
        super(DomainSingleEulerStep, self).__init__()

    def initialize(self):
        pass

    def stage1(self, d_idx, d_rho0, d_rho, d_arho, d_u, d_v, d_w, d_au, d_av,
               d_aw, d_x, d_y, d_z, d_disp, d_eps, d_eps_e, d_eps_p, d_eps_dot,
               d_eps_p_dot, d_sigma, d_sigma_dot, d_p, d_q, d_flag, d_f, d_Kf,
               d_Kk, d_Ek, d_cs, d_bulk, d_young, d_iflag, d_exit_flag, d_av_a,
               d_as_a, d_vps, d_h, t, dt):
        r"""
        :param d_idx:
        :param d_rho0:
        :param d_rho:
        :param d_arho:
        :param d_u:
        :param d_v:
        :param d_w:
        :param d_au:
        :param d_av:
        :param d_aw:
        :param d_x:
        :param d_y:
        :param d_z:
        :param d_disp:
        :param d_eps:
        :param d_eps_e:
        :param d_eps_p:
        :param d_eps_dot:
        :param d_eps_p_dot:
        :param d_sigma:
        :param d_sigma_dot:
        :param d_p:
        :param d_q:
        :param d_flag:
        :param d_f:
        :param d_Kf:
        :param d_Kk:
        :param d_Ek:
        :param d_cs:
        :param d_bulk:
        :param d_young:
        :param d_iflag:
        :param d_exit_flag:
        :param d_av_a:
        :param d_as_a:
        :param d_h:
        :param t:
        :param dt:

        :return:

        References
        ----------
        .. [BuiFukagawa2013]
        H.H. Bui, R. Fukagawa (2013) "An improved SPH method for saturated
        soils and its application to investigate the mechanisms of embankment
        failure: Case of hydrostatic pore-water pressure." Int. J. Numer.
        Anal. Meth. Geomech. Vol.37, p. 31–50
        """
        i, idx, step = declare("int", 3)
        idx = 3 * d_idx

        # Current time step
        step = int(t/dt)

        # Kinetic damping for stress and strain initialization
        if self.sim_type == 0 or self.sim_type == 2 or self.sim_type == 20:
            if d_Kf[0] < 0.01 and d_Kk[0] < 0.01:
                d_exit_flag[0] = 1  # Signal to exit current run
            else:
                d_iflag[0] = 0

                if d_Ek[1] < d_Ek[0]:

                    # Zero solid velocity
                    d_u[d_idx] = 0.0
                    d_v[d_idx] = 0.0
                    d_w[d_idx] = 0.0

                    # Reset flag to initialize other variables
                    d_iflag[0] = 2

        # ============ Update acceleration ============

        # Kinematic damping
        cd = 0.0
        if t < self.damp_time and self.damp_eps > 0:
            cd = self.damp_eps * d_cs[d_idx] / d_h[d_idx]

        # Add artificial viscosity and stress accelerations, body forces,
        # and kinematic damping to acceleration
        d_au[d_idx] += d_as_a[idx] + d_av_a[idx] + d_f[idx] - cd * d_u[d_idx]
        d_av[d_idx] += (d_as_a[idx + 1] + d_av_a[idx + 1] + d_f[idx + 1] -
                        cd * d_v[d_idx])
        d_aw[d_idx] += (d_as_a[idx + 2] + d_av_a[idx + 2] + d_f[idx + 2] -
                        cd * d_w[d_idx])
        # ===========================================

        # Update velocity
        d_u[d_idx] += dt * d_au[d_idx]
        d_v[d_idx] += dt * d_av[d_idx]
        d_w[d_idx] += dt * d_aw[d_idx]

        # Update the values of position and relative displacement
        dx = dt * (d_u[d_idx] + d_vps[idx])
        dy = dt * (d_v[d_idx] + d_vps[idx + 1])
        dz = dt * (d_w[d_idx] + d_vps[idx + 2])

        # Update position
        d_x[d_idx] += dx
        d_y[d_idx] += dy
        d_z[d_idx] += dz

        # Update accumulated displacement
        d_disp[idx] += dx
        d_disp[idx + 1] += dy
        d_disp[idx + 2] += dz

        # Update total, plastic, and elastic strains, and Cauchy stress
        idx = 9*d_idx
        for i in range(9):
            deps = dt*d_eps_dot[idx + i]
            d_sigma[idx + i] += dt * d_sigma_dot[idx + i]
            d_eps[idx + i] += deps

            if d_flag[d_idx] == 0:
                d_eps_e[idx + i] += deps
            else:
                deps_p = dt * d_eps_p_dot[idx + i]
                d_eps_p[idx + i] += deps_p
                d_eps_e[idx + i] += deps - deps_p

        # Update stress invariants
        d_p[d_idx] = (d_sigma[idx] + d_sigma[idx + 4] + d_sigma[idx + 8]) / 3.0

        # Calculate Von Mises stress, q, and effective plastic strain, ep_eff
        norm_s2 = 0.0
        p = d_p[d_idx]
        for i in range(9):
            s = d_sigma[idx + i]
            if i % 4 == 0:
                s -= p
            norm_s2 += s * s

        d_q[d_idx] = sqrt(3 * norm_s2 / 2)

        # Update density
        d_rho[d_idx] += dt * d_arho[d_idx]

        # Test to make sure density does not drop below a minimum
        rho_min = 0.5 * d_rho0[d_idx]
        if d_rho[d_idx] < rho_min:
            d_rho[d_idx] = rho_min

        # Update numerical sound speed
        rho = d_rho0[d_idx]
        if rho > d_rho[d_idx]:
            rho = d_rho[d_idx]

        d_cs[d_idx] = sqrt(d_young[d_idx] / rho)
        if d_bulk[d_idx] > d_young[d_idx]:
            d_cs[d_idx] = sqrt(d_bulk[d_idx] / rho)

    def stage2(self):
        pass


class BoundaryEulerStep(IntegratorStep):

    def __init__(self):
        super(BoundaryEulerStep, self).__init__()

    def initialize(self):
        pass

    def stage1(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_disp, d_u, d_v, d_vb,
               d_f, d_type, dt, t):

        i, idx = declare("int", 2)
        idx = 3*d_idx

        # Update velocity with prescribed body forces (accelerations)
        for i in range(3):
            d_vb[idx + i] += dt * d_f[idx + i]

        # Step displacement
        dx = 0.0
        dy = 0.0

        # Add another component in case we want to prescribe velocity as well
        dx += dt * d_vb[idx]
        dy += dt * d_vb[idx + 1]
        dz = dt * d_vb[idx + 2]

        # Update position
        d_x[d_idx] += dx
        d_y[d_idx] += dy
        d_z[d_idx] += dz

        # Update accumulated displacement
        d_disp[idx] += dx
        d_disp[idx + 1] += dy
        d_disp[idx + 2] += dz

# =============================================================================
# ======================= LEAP-FROG INTEGRATOR STEPPERS =======================
# =============================================================================

class DomainSingleLFStep(IntegratorStep):
    r"""
    (Predictor-Corrector) Leap-Frog Stepper.

    Implemented on 01/08/2025 and tested on 01/08/2025.
    Very limited testing, only with initialization of an elastic soil column
     and PBC (results were excellent!)

    Formulation from:
    J.P. Gray, J.J. Monaghan, and R.P. Swift (2001). SPH elastic dynamics.
    Comput. Methods Appl. Mech. Engrg, 190, 6641-6662.
    https://doi.org/10.1016/S0045-7825(01)00254-7.

    NOTE 1: initialize() is not adding body forces or any numerical damping to
     the initial initialization. Hence, in general, an = {0, 0, 0} for the
     first step. After the first step, it will have the body force from the
     previous step.

    NOTE 2: Contrary to what one may think at first, the damping and correction
     terms for the acceleration need to be added for both stages, but not body
     forces. If you add body forces to both, they end up being "doubly"
     counted, and the stresses and kinematics are affected (e.g., in
     equilibration problems, the vertical stress is about 1.5x the correct
     solution).

     This happens because adding body force to stage1() will cause it to add
     dt*f to the predicted velocity, which then receives another 0.5*dt*f term
     at the end of stage2(). Adding the body force to either stage1() or
     stage2() seems to have the same effect and works. Adding it to stage2()
     seems more reasonable as it will carry it to initialize() in the next
     step, when a_n-1 is stored.

    NOTE 3: More investigation to test the implementation is necessary
     (01/08/2025).
    """

    def __init__(self, damp_eps=0.01, damp_time=0.0):
        self.damp_eps = damp_eps
        self.damp_time = damp_time
        super(DomainSingleLFStep, self).__init__()

    def initialize(self, d_idx, d_arho, d_arhon, d_au, d_av, d_aw, d_an,
                   d_sigma_dot, d_sigma_dotn, d_eps_dot, d_eps_dotn):

        i, idx = declare("int", 2)
        idx = 3 * d_idx

        # Store current values for a and drho/dt.
        d_an[idx] = d_au[d_idx]
        d_an[idx + 1] = d_av[d_idx]
        d_an[idx + 2] = d_aw[d_idx]
        d_arhon[d_idx] = d_arho[d_idx]

        # Store current values for stress rate and strain rate
        idx = 9 * d_idx
        for i in range(9):
            d_sigma_dotn[idx + i] = d_sigma_dot[idx + i]
            d_eps_dotn[idx + i] = d_eps_dot[idx + i]

    def stage1(self, d_idx, d_x, d_y, d_z, d_disp, d_rho0, d_rho, d_arho, d_u,
               d_v, d_w, d_au, d_av, d_aw, d_av_a, d_as_a, d_eps, d_eps_dot,
               d_sigma, d_sigma_dot, d_vps, d_cs, d_h, t, dt):

        i, idx = declare("int", 2)
        idx = 3 * d_idx

        # Kinematic damping
        cd = 0.0
        if t < self.damp_time and self.damp_eps > 0:
            cd = self.damp_eps * d_cs[d_idx] / d_h[d_idx]

        # Add artificial viscosity and stress, and kinematic damping to
        #  acceleration
        d_au[d_idx] += d_av_a[idx] + d_as_a[idx] - cd * d_u[d_idx]
        d_av[d_idx] += d_av_a[idx + 1] + d_as_a[idx + 1] - cd * d_v[d_idx]
        d_aw[d_idx] += d_av_a[idx + 2] + d_as_a[idx + 2] - cd * d_w[d_idx]

        # Final particle displacements
        dte2 = 0.5 * dt ** 2
        dx = dt * (d_u[d_idx] + d_vps[idx]) + dte2 * d_au[d_idx]
        dy = dt * (d_v[d_idx] + d_vps[idx + 1]) + dte2 * d_av[d_idx]
        dz = dt * (d_w[d_idx] + d_vps[idx + 2]) + dte2 * d_aw[d_idx]

        # Update positions
        d_x[d_idx] += dx
        d_y[d_idx] += dy
        d_z[d_idx] += dz

        # Update displacements
        d_disp[idx] += dx
        d_disp[idx + 1] += dy
        d_disp[idx + 2] += dz

        # Predict velocity
        d_u[d_idx] += dt * d_au[d_idx]
        d_v[d_idx] += dt * d_av[d_idx]
        d_w[d_idx] += dt * d_aw[d_idx]

        # Predict mass density
        d_rho[d_idx] += dt * d_arho[d_idx]

        # Test to make sure density does not drop below a minimum
        rho_min = 0.5 * d_rho0[d_idx]
        if d_rho[d_idx] < rho_min:
            d_rho[d_idx] = rho_min

        # ============ Predict stress and strain ============
        idx = 9 * d_idx
        for i in range(9):
            d_sigma[idx + i] += dt * d_sigma_dot[idx + i]
            d_eps[idx + i] += dt * d_eps_dot[idx + i]

    def stage2(self, d_idx, d_rho, d_arho, d_arhon, d_rho0, d_u, d_v, d_w,
               d_au, d_av, d_aw, d_an, d_eps, d_eps_dot, d_eps_dotn, d_eps_e,
               d_eps_p, d_eps_p_dot, d_sigma, d_sigma_dot, d_sigma_dotn, d_p,
               d_q, d_av_a, d_as_a, d_f, d_cs, d_bulk, d_young, d_h, t, dt):

        i, idx = declare("int", 2)
        sdev = declare("matrix(9)")
        idx = 3 * d_idx

        # Half time-step
        dtd2 = dt / 2.0

        # Kinematic damping
        cd = 0.0
        if t < self.damp_time and self.damp_eps > 0:
            cd = self.damp_eps * d_cs[d_idx] / d_h[d_idx]

        # Final accelerations
        d_au[d_idx] += d_as_a[idx] + d_av_a[idx] + d_f[idx] - cd * d_u[d_idx]

        d_av[d_idx] += (
                d_as_a[idx + 1] + d_av_a[idx + 1] + d_f[idx + 1] -
                cd * d_v[d_idx]
        )

        d_aw[d_idx] += (
                d_as_a[idx + 2] + d_av_a[idx + 2] + d_f[idx + 2] -
                cd * d_w[d_idx]
        )

        # Corrected velocity
        d_u[d_idx] += dtd2 * (d_au[d_idx] - d_an[idx])
        d_v[d_idx] += dtd2 * (d_av[d_idx] - d_an[idx + 1])
        d_w[d_idx] += dtd2 * (d_aw[d_idx] - d_an[idx + 2])

        # Corrected mass density
        d_rho[d_idx] += dtd2 * (d_arho[d_idx] - d_arhon[d_idx])

        # Make sure density does not drop below a minimum
        rho_min = 0.5 * d_rho0[d_idx]
        if d_rho[d_idx] < rho_min:
            d_rho[d_idx] = rho_min

        d_cs[d_idx] = sqrt(d_young[d_idx] / d_rho[d_idx])
        if d_bulk[d_idx] > d_young[d_idx]:
            d_cs[d_idx] = sqrt(d_bulk[d_idx] / d_rho[d_idx])

        # Corrected Cauchy stress, and total strain
        idx = 9 * d_idx
        for i in range(9):
            d_sigma[idx + i] += (
                    dtd2 * (d_sigma_dot[idx + i] - d_sigma_dotn[idx + i])
            )

            d_eps[idx + i] += (
                    dtd2 * (d_eps_dot[idx + i] - d_eps_dotn[idx + i])
            )

        # Update stress invariants and accumulated deviatoric plastic strain
        d_p[d_idx] = (d_sigma[idx] + d_sigma[idx + 4] + d_sigma[idx + 8]) / 3.0

        # Calculate Von Mises stress, q
        norm_s2 = 0.0
        p = d_p[d_idx]
        for i in range(9):
            s = d_sigma[idx + i]
            if i % 4 == 0:
                s -= p
            sdev[i] = s
            norm_s2 += s * s

        d_q[d_idx] = sqrt(3 * norm_s2 / 2)

        # Updated elastic and plastic strains
        for i in range(9):
            d_eps_p[idx + i] += d_eps_p_dot[idx + i] * dt
            d_eps_e[idx + i] = d_eps[idx + i] - d_eps_p[idx + i]


# Copy of Euler stepper. Modify as needed in the future (01/09/2025)
class BoundaryLFStep(IntegratorStep):

    def __init__(self):
        super(BoundaryLFStep, self).__init__()

    def initialize(self):
        pass

    def stage1(self, d_idx, d_x, d_y, d_z, d_x0, d_y0, d_disp, d_u, d_v, d_vb,
               d_f, d_type, dt, t):

        i, idx = declare("int", 2)
        idx = 3*d_idx

        # Update velocity with prescribed body forces (accelerations)
        for i in range(3):
            d_vb[idx + i] += dt * d_f[idx + i]

        # Step displacement
        dx = 0.0
        dy = 0.0

        # Add another component in case we want to prescribe velocity as well
        dx += dt * d_vb[idx]
        dy += dt * d_vb[idx + 1]
        dz = dt * d_vb[idx + 2]

        # Update position
        d_x[d_idx] += dx
        d_y[d_idx] += dy
        d_z[d_idx] += dz

        # Update accumulated displacement
        d_disp[idx] += dx
        d_disp[idx + 1] += dy
        d_disp[idx + 2] += dz

    def stage2(self):
        pass
