from pysph.sph.integrator import Integrator


class MyEulerIntegrator(Integrator):

    def one_time_step(self, t, dt):
        self.stage1()
        self.update_domain()
        self.do_post_stage(dt, 1)
        self.compute_accelerations()


class MyLeapFrogIntegrator(Integrator):
    r"""
      A leap-frog integrator.
    """

    def one_timestep(self, t, dt):
        self.initialize()

        self.stage1()
        self.update_domain()
        self.do_post_stage(dt, 1)

        self.compute_accelerations()

        self.stage2()
        self.do_post_stage(dt, 2)
