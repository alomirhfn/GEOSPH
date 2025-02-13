import os
import numpy as np
from math import sqrt, pi

# =============================================================================
# ================================ PySPH IMPORTS ==============================

# Base imports
from pysph.base.utils import get_particle_array
from pysph.base.particle_array import ParticleArray
from pysph.base.kernels import WendlandQuintic, CubicSpline
from pysph.base.nnps import LinkedListNNPS, DomainManager

# Solver and application
from pysph.solver.application import Application
from pysph.solver.solver import Solver

# SPH equation imports
from pysph.sph.equation import Group

# =============================================================================
# ============================ MY OWN MODULES =================================
from io_utils import (import_parts_data, import_simulation_parameters,
                      get_csv_header, separate_particle_data_arrays,
                      convert_pysph_output)

from integrators import MyEulerIntegrator, MyLeapFrogIntegrator

from integrator_steppers import (
    BoundaryEulerStep, DomainSingleEulerStep,
    BoundaryLFStep, DomainSingleLFStep
)

from deformation_equations import DeformationRates

from boundary_equations import BoundaryStress, DummyBoundary, BoundaryPressure

from constitutive_equations import (ModifiedCamClay as MCCSolver,
                                    DruckerPragerSolverExact as DPSolver,
                                    WaterPressure as UndrainedPressure)

from conservation_equations import (MomentumEquation, DensityEquation,
                                    MomentumEquationPw)

from stress_equations import TrialStressDecomposition, TrialStress

from monaghan_equations import (MonaghanArtificialViscosity as ArtVisc,
                                PySPHArtificialStress as ArtStress)

from kernel_corrections import (KernelGradientCorrection as KernelGradCorrect,
                                KernelSum)

from particle_shifting import (ParticleShiftPreCalcs as PSPreCalcs,
                               ZhangParticleShift as ParticleShift,
)

# =============================================================================
# ============================ Global variables ===============================

TXT_PATH = "/data/Favero_Group/Projects/Input/Model.txt"
CSV_PATH = "/data/Favero_Group/Projects/Input/Model.csv"

# =============================================================================
# =============================================================================


# Define an 'Application' class by subclassing the pysph.solver.application.
# Application class
class SDPySPHApplication(Application):

    def initialize(self):
        """ Initialize user defined parameters for the simulation, f.e.
        constants, etc. One can write a TXT file containing the input values
        for each field defined. For an explanation of the formatting rules,
        look at the docstring for io_utils.import_simulation_parameters. If a
        file path is not provided, define default values
        """

        # Import parameters from a TXT file if available
        if os.path.exists(TXT_PATH):
            sim_params = import_simulation_parameters(TXT_PATH)
            self.dp = float(sim_params['dp'])  # Initial particle distance
            self.kh = float(sim_params['kh'])  # Smoothing length factor
            self.tf = float(sim_params['simTime'])  # Final simulation time
            self.time_step = float(sim_params['stepSize'])  # Initial step
            self.sim_dim = int(sim_params['simDim'])  # Space dimensions
            self.kgc = int(sim_params['CorrNorm'])  # Gradient correction
            self.nnps = sim_params['NNPS'].lower()  # Type of NNPS algorithm
            self.alpha = float(sim_params['alpha'])  # Monaghan alpha
            self.c0 = float(sim_params['c'])  # Initial sound speed
            self.as_epsilon = float(sim_params['as_epsilon'])  # A. Stress
            self.pbc = int(sim_params['PBC'])  # Is periodic problem?
            self.pbcx = int(sim_params['PBCX'])  # Periodic in x?
            self.pbcy = int(sim_params['PBCY'])  # Periodic in y?
            self.pbcz = int(sim_params['PBCZ'])  # Periodic in z?
            self.damp_time = float(sim_params['eqTime'])  # Solution damp

            # Monaghan artificial viscosity coefficient beta
            if 'beta' in sim_params:
                self.beta = float(sim_params['beta'])
            else:
                self.beta = 2.0*self.alpha

            # Type of kernel to use
            kernel_choice = sim_params['kernel'].lower()
            if kernel_choice == 'wendlandquintic':
                self.kernel = WendlandQuintic
            elif kernel_choice == 'cubicspline':
                self.kernel = CubicSpline
            else:
                self.kernel = WendlandQuintic

            # Type of integrator
            if 'integrator' in sim_params:
                self.integrator = int(sim_params['integrator'])
            else:
                self.integrator = int(0)

            # Periodic domain dimensions
            if self.pbcx == 1 or self.pbcy == 1 or self.pbcz == 1:
                self.pbc = 1
            if self.pbc == 1:
                if 'xmin' not in sim_params or 'xmax' not in sim_params:
                    print("Must provide PB dimensions: xmin, xmax")
                    print("")
                    exit()
                else:
                    self.xmin = float(sim_params['xmin'])
                    self.xmax = float(sim_params['xmax'])
                if 'ymin' not in sim_params or 'ymax' not in sim_params:
                    print("Must provide PB dimensions: ymin, ymax")
                    print("")
                    exit()
                else:
                    self.ymin = float(sim_params['ymin'])
                    self.ymax = float(sim_params['ymax'])

                if 'zmin' not in sim_params or 'zmax' not in sim_params:
                    print("Must provide PB dimensions: zmin, zmax")
                    print("")
                    exit()
                else:
                    self.zmin = float(sim_params['zmin'])
                    self.zmax = float(sim_params['zmax'])

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # If no TXT file (or invalid path), EXIT.
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        else:
            print('TXT file not found!')
            exit()

        # Initial smoothing length
        self.h0 = self.kh*self.dp

        # Gravity acceleration
        self.gravity = 9.81

        # Constitutive parameters
        self.c_model = int(1)
        self.y_criterion = int(2)
        self.sy0 = 1e16

        # Output frequency
        self.out_freq = int(1)

        # Debugging option
        self.debug = int(0)

        # Boundary particle for debugging
        self.bp = int(0)

        # Initialize particles for stability
        self.sim_type = 0

        # Confining stress
        self.sigma_c = 0.0

        # Stress and strain regularization frequency
        self.ssr_freq = int(10)

    def add_user_options(self, group):
        """ User specified command line options, i.e. sys.argv calls. This is
        parsed before running the simulation such that the values passed by the
        user can be used to configure the application.

        Parameters
        -----------
        :param group: internal class of PySPH

        Output
        -----------
        :return: None - these options will be available when running the
        simulation from the command line
        """

        group.add_argument(
            "--txt-path", action="store", type=str, dest="txt_path",
            help="Path to input TXT file."
        )

        group.add_argument(
            "--csv-path", action="store", type=str, dest="csv_path",
            help="Path to input CSV file."
        )

        group.add_argument(
            "--dp", action="store", type=float, dest="dp", default=self.dp,
            help="Initial interparticle distance."
        )

        group.add_argument(
            "--kh", action="store", type=float, dest="kh", default=self.kh,
            help="Smoothing length factor that multiplies dp to obtain h."
        )

        group.add_argument(
            "--simdim", action="store", type=int, dest="dim",
            default=self.sim_dim,
            help="Spatial dimensions of the problem (2 for 2D and 3 for 3D)."
        )

        group.add_argument(
            "--kernel-grad-correct", action="store", type=int, dest="kgc",
            default=self.kgc,
            help="Select whether to correct the kernel derivative "
                 "(0 = 'No', 1 = 'Yes')."
        )

        group.add_argument(
            "--integrator_choice", action="store", type=int, dest="integrator",
            default=1,
            help="Select which time integration scheme to use "
                 "(0 = Forward Euler, 1 = Leap-Frog)."
        )

        group.add_argument(
            "--monaghan-alpha", action="store", type=float, dest="alpha",
            default=self.alpha,
            help="Monaghan's artificial viscosity coefficient alpha."
        )

        group.add_argument(
            "--monaghan-beta", action="store", type=float, dest="beta",
            default=0.0,
            help="Monaghan's artificial viscosity coefficient beta."
        )

        group.add_argument(
            "--c0", action="store", type=float, dest="c0", default=self.c0,
            help="Reference numerical sound speed."
        )

        group.add_argument(
            "--monaghan-as-epsilon", action="store", type=float,
            dest="as_epsilon", default=self.as_epsilon,
            help="Monaghan's artificial stress scaling coefficient."
        )

        group.add_argument(
            "--pbc", action="store", type=int, dest="pbc", default=self.pbc,
            help="Select whether the problem has periodic boundary conditions "
                 "(0 = 'No', 1 = 'Yes')."
        )

        group.add_argument(
            "--pbc-x", action="store", type=int, dest="pbcx",
            default=self.pbcx,
            help="Whether the problem is periodic in the x-direction "
                 "(0 = 'No', 1 = 'Yes')."
        )

        group.add_argument(
            "--pbc-y", action="store", type=int, dest="pbcy",
            default=self.pbcy,
            help="Whether the problem is periodic in the y-direction "
                 "(0 = 'No', 1 = 'Yes')."
        )

        group.add_argument(
            "--pbc-z", action="store", type=int, dest="pbcz",
            default=self.pbcz,
            help="Whether the problem is periodic in the z-direction "
                 "(0 = 'No', 1 = 'Yes')."
        )

        group.add_argument(
            "--damp-time", action="store", type=float, dest="damp_time",
            default=self.damp_time,
            help="Final time for which the solution must be damped."
        )

        group.add_argument(
            "--kw", action="store", type=float, dest="kw", default=2.0,
            help="Kernel radius coefficient used to multiply the smoothing "
                 "length: r0 = kw * h0."
        )

        group.add_argument(
            "--gravity", action="store", type=float, dest="g",
            default=self.gravity, help="Gravity acceleration modulus."
        )

        group.add_argument(
            "--c_model", action="store", type=int, dest="c_model",
            default=self.c_model,
            help="Constitutive model for the material. "
                 "0 = Elastic, 1 = Elastoplastic (Default: 1)"
        )

        group.add_argument(
            "--y_criterion", action="store", type=int, dest="y_criterion",
            default=self.y_criterion,
            help="Yield criterion if the constitutive model is not 'EL'. "
                 "1 = Von Mises, 2 = Drucker-Prager, 3 = MCC. (Default: 2)"
        )

        group.add_argument(
            "--debug", action="store", type=int, dest="debug",
            default=self.debug,
            help="Activate printing statements for debugging."
        )

        group.add_argument(
            "--debug-bound", action="store", type=int, dest="bp",
            default=self.bp,
            help="Boundary particle to debug (print stress tensor)."
        )

        group.add_argument(
            "--sim-type", action="store", type=int, dest="sim_type",
            default=0,
            help="Type of simulation performed: '0'-Initialization, "
                 "'1'-Failure. (Default: '0' - Initialization)"
        )

        group.add_argument(
            "--bulkmod-w", action="store", type=float, dest="bulkw",
            default=2e9,
            help="Bulk elastic modulus of water. (Default: 2.0 GPa)"
        )

        group.add_argument(
            "--kdamp-eps", action="store", type=float, dest="kin_damp",
            default=0.001,
            help="Kinematic damping factor. (Default: 0.001)"
        )

        group.add_argument(
            "--sigma_c", action="store", type=float, dest="sigma_c",
            default=0.0,
            help="Confining stress for triaxial-type BCs. (Default: 0.0)"
        )

        group.add_argument(
            "--ssr_freq", action="store", type=int, dest="ssr_freq",
            default=10,
            help="Stress and strain regularization frequency. (Default: 10)"
        )

    def consume_user_options(self):
        """ This is called after the command line arguments are parsed and can
        be accessed in self.options. This is meant to be overridden by the user
        to set up any internal variables that depend on the command line
        arguments passed.
        """

        # Assign changes to simulation parameters due to command line input
        if self.options.dp != self.dp:
            self.dp = self.options.dp

        if self.options.kh != self.kh:
            self.kh = self.options.kh

        if self.options.dim != self.sim_dim:
            self.sim_dim = self.options.dim

        if self.options.kgc != self.kgc:
            self.kgc = self.options.kgc

        # Integrator options
        integrator_choice = self.options.integrator
        if integrator_choice is not None:
            self.integrator = integrator_choice

        if self.options.alpha != self.alpha:
            self.alpha = self.options.alpha

        if self.options.beta is not None:
            self.beta = self.options.beta

        if self.options.c0 != self.c0:
            self.c0 = self.options.c0

        if self.options.as_epsilon != self.as_epsilon:
            self.as_epsilon = self.options.as_epsilon

        if self.options.pbc != self.pbc:
            self.pbc = self.options.pbc

        if self.options.pbcx != self.pbcx:
            self.pbcx = self.options.pbcx

        if self.options.pbcy != self.pbcy:
            self.pbcy = self.options.pbcy

        if self.options.pbcz != self.pbcz:
            self.pbcz = self.options.pbcz

        if self.pbcx == 1 or self.pbcy == 1 or self.pbcz == 1:
            self.pbc = 1
        elif self.pbcx == 0 and self.pbcy == 0 and self.pbcz == 0:
            self.pbc = 0

        if self.options.damp_time != self.damp_time:
            self.damp_time = self.options.damp_time

        if self.options.g != self.gravity:
            self.gravity = self.options.g

        # Reference kernel radius coefficient and radius
        self.kw = self.options.kw
        self.r0 = self.kw * self.h0

        # Time step and final simulation time
        if self.options.time_step is not None:
            self.time_step = self.options.time_step
        if self.options.final_time is not None:
            self.tf = self.options.final_time

        # Number of damping steps
        self.n_damp = int(self.damp_time / self.time_step)
        if self.options.n_damp is not None:
            self.n_damp = self.options.n_damp

        # Kernel choice
        kernel_choice = self.options.kernel
        if kernel_choice is not None:
            kernel_choice = kernel_choice.lower()
            if kernel_choice == 'wendlandquintic':
                self.kernel = WendlandQuintic
            elif kernel_choice == 'cubicspline':
                self.kernel = CubicSpline
            else:
                self.kernel = WendlandQuintic

        # Output dumping frequency
        if self.options.freq is not None:
            self.out_freq = self.options.freq

        # Constitutive model parameters
        if self.options.c_model != self.c_model:
            self.c_model = self.options.c_model

        if self.options.y_criterion != self.y_criterion:
            self.y_criterion = self.options.y_criterion

        # Debugging option
        if self.options.debug != self.debug:
            self.debug = self.options.debug

        # Boundary particle for debugging
        if self.options.bp != self.bp:
            self.bp = self.options.bp

        # Type of simulation (single phase, etc.)
        if self.options.sim_type is not None:
            self.sim_type = self.options.sim_type

        # Bulk elastic modulus of water
        if self.options.bulkw is not None:
            self.bulkw = self.options.bulkw

        # Kinematic damping factor
        if self.options.kin_damp is not None:
            self.damp_eps = self.options.kin_damp

        # Confining pressure to be applied to all particles
        if self.options.sigma_c is not None:
            self.sigma_c = self.options.sigma_c

        # Stress and strain regularization frequency
        if self.options.ssr_freq is not None:
            self.ssr_freq = self.options.ssr_freq

    # This is a mandatory method.
    def create_particles(self):
        """ This method generates the ParticleArrays necessary for the PySPH
        framework to run. It reads a CSV file containing all the information
        properties and geometric properties of each particle and returns a list
        of PySPH ParticleArrays.

        :return: returns a list of PySPH ParticleArrays
        """

        # Read particles properties and simulation parameters from files
        part_arr = import_parts_data(CSV_PATH)

        # Header with particle fields in the CSV file
        header = get_csv_header(CSV_PATH)

        # Create the dictionary of all particles properties
        part_data = separate_particle_data_arrays(part_arr, header)

        # Select particles by type ('tag') and associate their names with those
        #  indices
        types_arr = part_data['type']
        domain_idx = np.where(types_arr < 200)[0]
        boundary_idx = np.where(types_arr >= 200)[0]
        part_types = {
            'domain': domain_idx,  # All domain particles
            'boundary': boundary_idx}  # Fixed boundary particles

        # Initialize the ParticleArrays and add them to a list that will be
        #  returned
        domain = get_particle_array(name='domain')
        boundary = get_particle_array(name='boundary')
        pas = [domain, boundary]

        # Initialize the counter for the total number of particles
        tot_num_parts = 0.0

        # Populate each PySPH ParticleArray in pas with new properties and
        #  corresponding values, except stress and strain.
        for pa_type in pas:

            # TODO: I believe I am generating duplicate (or unused)
            #  properties for particles as it is generating them here and
            #  then later on, I am adding them manually again. Check this
            #  (10/17/2023)

            for key in part_data.keys():
                if ((key[0] == 's' or key[0] == 'e') and len(key) == 3) or \
                        key[0] == 'n' or key == 'bc':
                    continue
                pa_type.add_property(
                    key,
                    default=0.0,
                    data=part_data[key][part_types[pa_type.name]]
                )

            # Miscellaneous operations
            pa_type.align_particles()
            num_parts = pa_type.get_number_of_particles()
            tot_num_parts += num_parts

            # Assign smoothing length to all particles
            h_arr = self.h0 * np.ones(num_parts)
            pa_type.h = h_arr

            # Add kernel sum property
            pa_type.add_property('wsum', default=1.0)

            # Add density rate
            pa_type.add_property('arho', default=0.0)

            # Add Cauchy stress tensor
            sigma = np.zeros(9*num_parts)
            sig_keys = ['sxx', 'sxy', 'sxz', 'sxy', 'syy', 'syz', 'sxz', 'syz',
                        'szz']

            for i, key in enumerate(sig_keys):
                if key in part_data.keys():
                    sigma[i::9] = part_data[key][part_types[pa_type.name]]
                else:
                    print("Key \'%s\' not found!" % key)
                    exit()

            pa_type.add_property('sigma', default=0.0, data=sigma, stride=9)

            # Bulk and shear elastic moduli
            young = part_data['young'][part_types[pa_type.name]]
            poisson = part_data['poisson'][part_types[pa_type.name]]
            bulk = young / (3.0 * (1.0 - 2.0 * poisson))
            shear = young / (2 * (1 + poisson))
            pa_type.add_property('bulk', default=1.0, data=bulk)
            pa_type.add_property('shear', default=1.0, data=shear)

            # Sound speed
            rho0 = part_data['rho'][part_types[pa_type.name]]
            sv0 = np.full(num_parts, self.c0)
            cs0 = np.maximum(np.sqrt(bulk / rho0), np.sqrt(young / rho0))
            cs0 = np.maximum(cs0, sv0)
            pa_type.add_property('rho0', default=1.0, data=rho0)
            pa_type.add_property('cs', default=self.c0, data=cs0)

            # Artificial Stress
            pa_type.add_property('asig', default=0.0, stride=9)

            # Add body forces
            body = np.zeros(3 * num_parts)
            bf_keys = ['fx', 'fy', 'fz']

            for i, key in enumerate(bf_keys):
                if key in part_data.keys():
                    body[i::3] = part_data[key][part_types[pa_type.name]]
                else:
                    print("Key \'%s\' not found!" % key)
                    exit()

            pa_type.add_property('f', default=0.0, data=body, stride=3)

            # Accumulated displacement
            pa_type.add_property('disp', default=0.0, stride=3)

            # Porosity
            nv = part_data['nw'][part_types[pa_type.name]]
            pa_type.add_property('nv', default=0.0, data=nv)

            # =================================================================
            # Add properties to domain particles only
            if pa_type.name == 'domain':

                # Inverse of the L matrix to correct the kernel gradient
                pa_type.add_property('l_mat', default=0.0, stride=9)

                # Deviatoric stress tensor
                pa_type.add_property('sigma_dev', default=0.0, stride=9)

                # Trial stress tensor
                pa_type.add_property('sigma_tr', default=0.0, data=sigma,
                                     stride=9)

                # Stress rate
                pa_type.add_property('sigma_dot', default=0.0, stride=9)

                # Stress diffusion rate
                pa_type.add_property('sig_diff', default=0.0, stride=9)

                # Deviatoric stress invariant, q
                pa_type.add_property('q', default=0.0)

                # Artificial viscosity acceleration, av_a
                pa_type.add_property('av_a', default=0.0, stride=3)

                # Artificial stress acceleration, as_a
                pa_type.add_property('as_a', default=0.0, stride=3)

                # Kernel value for dp
                # TODO: Check if can be converted to a constant, then do it!
                wdp = self.kernel(self.sim_dim).kernel([0,0], self.dp, self.h0)
                pa_type.add_constant('wdp', wdp)

                # Add elastic strain tensors
                eps_e = np.zeros(9*num_parts)
                e_keys = ['exx', 'exy', 'exz', 'exy', 'eyy', 'eyz', 'exz',
                          'eyz', 'ezz']
                for i, key in enumerate(e_keys):
                    if key in part_data.keys():
                        eps_e[i::9] = part_data[key][part_types[pa_type.name]]
                    else:
                        print("Key \'%s\' not found!" % key)
                        exit()

                # Initialize plastic and total strain tensors
                eps_p = np.zeros(9*num_parts)
                eps = np.copy(eps_e)

                # Make sure that Eps_zz = 0 if plane-strain
                if self.sim_dim == 2:
                    eps_p[8::9] = -eps_e[8::9]
                    eps[8::9] = 0.0

                # Deformation vectors
                pa_type.add_property('eps', default=0.0, data=eps, stride=9)
                pa_type.add_property('eps_e', default=0.0, data=eps_e,
                                     stride=9)
                pa_type.add_property('eps_p', default=0.0, data=eps_p,
                                     stride=9)
                pa_type.add_property('eps_dot', default=0.0, stride=9)

                # TODO: With the new way you are calculating eps_p in the
                #  integrators, you can potentially eliminate this property and
                #  calculations of it in the constitutive update equations.
                pa_type.add_property('eps_p_dot', default=0.0, stride=9)

                pa_type.add_property('spin_dot', default=0.0, stride=9)
                pa_type.add_property('ep_acc', default=0.0)
                pa_type.add_property('ep_eff', default=0.0)

                # Flag property
                pa_type.add_property('flag', default=0, type='int')

                # Kernel derivative correction matrix
                m_mat = np.zeros(9 * num_parts)
                m_mat[0::9] = 1.0
                m_mat[4::9] = 1.0
                m_mat[8::9] = 1.0
                pa_type.add_property('m_mat', default=0.0, data=m_mat,
                                     stride=9)

                # =============================================================
                # ==================== CONSTITUTIVE PARAMS ====================
                phi = part_data['phi'][part_types[pa_type.name]] * pi / 180
                psi = part_data['psi'][part_types[pa_type.name]] * pi / 180
                cohesion = part_data['cohesion'][part_types[pa_type.name]]
                ac = sqrt(2.0 / 3.0)
                sy = sqrt(3.0) * cohesion

                # Von Mises
                if self.y_criterion == 1:
                    if self.sim_dim == 3:
                        sy = 2*cohesion

                    # Add property to particles
                    pa_type.add_property('aphi', default=0.0)
                    pa_type.add_property('apsi', default=0.0)
                    pa_type.add_property('ac', default=0.0, data=ac)
                    pa_type.add_property('sy', default=0.0, data=sy)

                # Drucker-Prager parameters. sy is the uniaxial yield stress.
                #  This ensures consistency between VM and DP models for
                #  phi = 0.
                elif self.y_criterion == 2:
                    if self.sim_dim == 2:
                        aphi = sqrt(6)*np.tan(phi) / \
                            np.sqrt(3 + 4*np.power(np.tan(phi), 2))
                        apsi = sqrt(6)*np.tan(psi) / \
                            np.sqrt(3 + 4*np.power(np.tan(psi), 2))
                        ac = sqrt(2) / np.sqrt(3 + 4*np.power(np.tan(phi), 2))

                    else:
                        sy = 2*cohesion

                        # Fit to the MCYC through the outer edges
                        #  ("-" compression)
                        aphi = 2*sqrt(6)*np.sin(phi) / (3 - np.sin(phi))
                        apsi = 2*sqrt(6)*np.sin(psi) / (3 - np.sin(psi))
                        ac = sqrt(6)*np.cos(phi) / (3 - np.sin(phi))

                    pa_type.add_property('aphi', default=0.0, data=aphi)
                    pa_type.add_property('apsi', default=0.0, data=apsi)
                    pa_type.add_property('ac', default=0.0, data=ac)
                    pa_type.add_property('sy', default=0.0, data=sy)

                # Modified Cam-Clay model
                elif self.y_criterion == 3:

                    # Void ratio
                    e = nv / (1 - nv)
                    pa_type.add_property('void_ratio', default=0.0, data=e)


                # =================== Integrator properties ===================
                # Add tracking parameters at previous step for leapfrog
                # integrator
                if self.integrator == 1:

                    # Velocity acceleration at step n
                    pa_type.add_property('an', default=0.0, stride=3)

                    # Mass density rate at step "n"
                    pa_type.add_property('arhon', default=0.0)

                    # Stress rate at step n
                    pa_type.add_property('sigma_dotn', default=0.0, stride=9)

                    # Strain rate at step n
                    pa_type.add_property('eps_dotn', default=0.0, stride=9)

                # =============================================================

                # ============= Properties for undrained simulations ===========
                if self.sim_type == 2:
                    pa_type.add_property('pwdt', default=0.0)
                # =============================================================

                # ================ Particle shifting properties ===============
                pa_type.add_property('divx', default=0.0)  # div(position)
                pa_type.add_property('pv', default=0.0)  # vij.xij/rij
                pa_type.add_property('pstype', default=0, type="int")
                pa_type.add_property('phips', default=0.0) # Reduce normal du
                pa_type.add_property('gradc', default=0.0, stride=3)  # Sum(DW)
                pa_type.add_property('gfick', default=0.0, stride=3)  # D^C
                pa_type.add_property('vps', default=0.0, stride=3) # PS vel.

                # Normal vectors used for free surface particles
                n_vec = np.zeros(3 * num_parts)
                keys = ['nx', 'ny', 'nz']

                for i, key in enumerate(keys):
                    if key in part_data.keys():
                        n_vec[i::3] = \
                            part_data[key][part_types[pa_type.name]]
                    else:
                        print("Key \'%s\' not found!" % key)
                        exit()

                pa_type.add_property('n', default=0.0, data=n_vec, stride=3)
                # =============================================================

                # Set output arrays for domain PAs
                if self.y_criterion == 3:  # MCC
                    pa_type.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w',
                                               'rho', 'eps', 'eps_e', 'eps_p',
                                               'ep_acc', 'p', 'q', 'sigma',
                                               'gid', 'type', 'disp', 'ep_eff',
                                               'pw', 'nv', 'pc', 'void_ratio'])

                else:
                    pa_type.set_output_arrays(['x', 'y', 'z', 'u', 'v', 'w',
                                               'rho', 'eps', 'eps_e', 'eps_p',
                                               'ep_acc', 'p', 'q', 'sigma',
                                               'gid', 'type', 'disp', 'ep_eff',
                                               'pw', 'nv'])

            # =================================================================
            # Add Properties to boundary particles
            if pa_type.name == 'boundary':

                # ==== For Vane shear simulations ====

                # Initial x-coordinate
                pa_type.add_property(
                    'x0',
                    default=0.0,
                    data=part_data['x'][part_types[pa_type.name]],
                )

                # Initial y-coordinate
                pa_type.add_property(
                    'y0',
                    default=0.0,
                    data=part_data['y'][part_types[pa_type.name]],
                )
                # ====================================

                # Add boundary condition type (0/1 - Fixed/Slip, 2 - Adami)
                bc = np.zeros(num_parts, dtype='i')
                key = 'bc'
                if key in part_data.keys():
                    bc[::] = part_data[key][part_types[pa_type.name]]
                else:
                    print("Key \'%s\' not found!" % key)
                    exit()

                pa_type.add_property('bc', default=0, data=bc, type='int')

                # ==== ADD NORMAL VECTORS TO THE BOUNDARY PARTICLES ==== #
                n_vec = np.zeros(3 * num_parts)
                keys = ['nx', 'ny', 'nz']

                for i, key in enumerate(keys):
                    if key in part_data.keys():
                        n_vec[i::3] = \
                            part_data[key][part_types[pa_type.name]]
                    else:
                        print("Key \'%s\' not found!" % key)
                        exit()

                pa_type.add_property('n', default=0.0, data=n_vec, stride=3)

                # ====================================================== #

                # Add prescribed boundary velocity
                vb = np.zeros(3*num_parts)
                keys = ['u', 'v', 'w']

                for i, key in enumerate(keys):
                    if key in part_data.keys():
                        vb[i::3] = part_data[key][part_types[pa_type.name]]
                    else:
                        print("Key \'%s\' not found!" % key)
                        exit()

                pa_type.add_property('vb', default=0.0, data=vb, stride=3)

                pa_type.set_output_arrays(['x', 'y', 'z', 'vb', 'rho', 'disp',
                                           'sigma', 'gid', 'type', 'pw', 'n'])

            # load balancing properties
            pa_type.set_lb_props(list(pa_type.properties.keys()))

        return pas

    # Integrator and integrator stepper choices
    def create_solver(self):
        kernel = self.kernel(self.sim_dim)

        # Forward Euler
        if self.integrator == 0:
            domain_stepper = DomainSingleEulerStep(
                sim_type=self.sim_type,
                damp_eps=self.damp_eps,
                damp_time=self.damp_time,
            )
            boundary_stepper = BoundaryEulerStep()

            integrator = MyEulerIntegrator(
                domain=domain_stepper,
                boundary=boundary_stepper,
            )

        # Leap-Frog
        elif self.integrator == 1:
            domain_stepper = DomainSingleLFStep(
                damp_eps=self.damp_eps,
                damp_time=self.damp_time,
            )

            boundary_stepper = BoundaryLFStep()

            integrator = MyLeapFrogIntegrator(
                domain=domain_stepper,
                boundary=boundary_stepper,
            )

        # Default (Euler)
        else:
            domain_stepper = DomainSingleEulerStep(
                sim_type=self.sim_type,
                damp_eps=self.damp_eps,
                damp_time=self.damp_time,
            )

            boundary_stepper = BoundaryEulerStep()

            integrator = MyEulerIntegrator(
                domain=domain_stepper,
                boundary=boundary_stepper,
            )

        solver = Solver(
            kernel=kernel,
            dim=self.sim_dim,
            integrator=integrator,
            dt=self.time_step,
            tf=self.tf,
            fixed_h=True,
            pfreq=self.out_freq
        )
        return solver

    # This method must be overloaded if not using a Scheme instance.
    def create_equations(self):

        # This enables the list of equations to take the correct YC
        if self.y_criterion == 1 or self.y_criterion == 2:
            # DP or VonMises
            ConstitutiveSolver = DPSolver
            constitutive_solver_kwargs_single = {
                "dest": 'domain',
                "sources": None,
                "c_model": self.c_model,
                "debug": self.debug,
            }
        elif self.y_criterion == 3:
            # MCC
            ConstitutiveSolver = MCCSolver
            constitutive_solver_kwargs_single = {
                "dest": 'domain',
                "sources": None,
                "nu": 0.3,
                "c_model": self.c_model,
                "tol": 1e-5,
                "max_iter": 100,
                "debug": self.debug,
            }
        else:
            raise NotImplementedError("Unknown yield criterion during equation"
                                      " building")

        # Single phase "dry" (total stress) simulation
        if self.sim_type == 1:
            equations = [

                # Group 1
                Group(
                    equations=[
                        KernelSum(
                            dest='boundary',
                            sources=['domain'],
                            debug=self.debug,
                            bp=self.bp,
                        ),
                        KernelSum(
                            dest='domain',
                            sources=['domain', 'boundary'],
                            debug=self.debug,
                            bp=self.bp,
                        ),
                        KernelGradCorrect(
                            dest='domain',
                            sources=['domain', 'boundary'],
                            kgc=self.kgc,
                            sim_dim=self.sim_dim,
                            debug=self.debug,
                        ),
                    ],
                    real=True
                ),

                # Group 2
                Group(
                    equations=[
                        BoundaryStress(
                            dest='boundary',
                            sources=['domain'],
                            sim_dim=self.sim_dim,
                            debug_bound=self.bp,
                        ),
                        PSPreCalcs(
                            dest='domain',
                            sources=['domain', 'boundary'],
                            sim_dim=self.sim_dim,
                        ),
                    ],
                    real=True
                ),

                # Group 3
                Group(
                    equations=[
                        DeformationRates(
                            dest='domain',
                            sources=['domain', 'boundary'],
                        ),
                        DensityEquation(
                            dest='domain',
                            sources=['domain', 'boundary'],
                            debug=self.debug,
                        ),
                    ],
                    real=True
                ),

                # Group 4
                Group(
                    equations=[
                        TrialStress(
                            dest='domain',
                            sources=['domain'],
                            debug=self.debug,
                        ),
                        TrialStressDecomposition(
                            dest='domain',
                            sources=['domain'],
                            debug=self.debug,
                        ),
                        ConstitutiveSolver(
                            **constitutive_solver_kwargs_single
                        ),
                        MomentumEquation(
                            dest='domain',
                            sources=['domain', 'boundary'],
                            sigma_c=self.sigma_c,
                        ),
                        ArtVisc(
                            dest='domain',
                            sources=['domain', 'boundary'],
                            alpha=self.alpha,
                            beta=self.beta,
                        ),
                        ParticleShift(
                            dest='domain',
                            sources=['domain'],
                            h0=self.h0,
                            simdim=self.sim_dim,
                        ),
                    ],
                    real=True
                ),
            ]

        # Single phase (total stress) undrained simulation
        elif self.sim_type == 2:
            equations = [

                # Group 1
                Group(
                    equations=[
                        KernelSum(
                            dest='boundary',
                            sources=['domain'],
                            debug=self.debug,
                            bp=self.bp,
                        ),
                        KernelSum(
                            dest='domain',
                            sources=['domain', 'boundary'],
                            debug=self.debug,
                            bp=self.bp,
                        ),
                        KernelGradCorrect(
                            dest='domain',
                            sources=['domain', 'boundary'],
                            kgc=self.kgc,
                            sim_dim=self.sim_dim,
                            debug=self.debug,
                        ),
                    ],
                    real=True
                ),

                # Group 2
                Group(
                    equations=[
                        BoundaryStress(
                            dest='boundary',
                            sources=['domain'],
                            sim_dim=self.sim_dim,
                            debug_bound=self.bp,
                        ),
                        BoundaryPressure(
                            dest='boundary',
                            sources=['domain'],
                            gravity=self.gravity,
                            sim_dim=self.sim_dim,
                        ),
                        PSPreCalcs(
                            dest='domain',
                            sources=['domain', 'boundary'],
                            sim_dim=self.sim_dim,
                        ),
                    ],
                    real=True
                ),

                # Group 3
                Group(
                    equations=[
                        DeformationRates(
                            dest='domain',
                            sources=['domain', 'boundary'],
                        ),
                        DensityEquation(
                            dest='domain',
                            sources=['domain', 'boundary'],
                            debug=self.debug,
                        ),
                    ],
                    real=True
                ),

                # Group 4
                Group(
                    equations=[
                        TrialStress(
                            dest='domain',
                            sources=['domain'],
                            debug=self.debug,
                        ),
                        TrialStressDecomposition(
                            dest='domain',
                            sources=['domain'],
                            debug=self.debug,
                        ),
                        ConstitutiveSolver(
                            **constitutive_solver_kwargs_single
                        ),
                        MomentumEquationPw(
                            dest='domain',
                            sources=['domain', 'boundary'],
                            sigma_c=self.sigma_c,
                        ),
                        ArtVisc(
                            dest='domain',
                            sources=['domain', 'boundary'],
                            alpha=self.alpha,
                            beta=self.beta,
                        ),
                        UndrainedPressure(
                            dest='domain',
                            sources=['domain', 'boundary'],
                            bulkw=self.bulkw,
                            sim_dim=self.sim_dim,
                            drained=self.drained,
                        ),
                        ParticleShift(
                            dest='domain',
                            sources=['domain'],
                            h0=self.h0,
                            simdim=self.sim_dim,
                        ),
                    ],
                    real=True
                ),
            ]

        # Not implemented
        else:
            raise NotImplementedError("Unknown simulation type during equation")

        return equations

    def create_nnps(self):
        if self.pbc == 1:
            domain = DomainManager(
                xmin=self.xmin, xmax=self.xmax,
                ymin=self.ymin, ymax=self.ymax,
                zmin=self.zmin, zmax=self.zmax,
                periodic_in_x=self.pbcx,
                periodic_in_y=self.pbcy,
                periodic_in_z=self.pbcz
            )
        else:
            domain = None

        if self.nnps.upper() == 'LL':  # Linked-list NNPS
            nps = LinkedListNNPS(
                dim=self.sim_dim,
                particles=self.particles,
                radius_scale=self.kw,
                cache=False,
                domain=domain
            )
        else:
            nps = LinkedListNNPS(
                dim=self.sim_dim,
                particles=self.particles,
                radius_scale=self.kw,
                cache=False,
                domain=domain
            )
        return nps

    # Post-process output and other results management methods
    def post_process(self):
        in_path = os.path.abspath(self.options.output_dir)
        dir_path = os.path.join(in_path, 'VTU')

        # Create a vtu directory if not there yet.
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        # Save output files to the VTU directory with VTU format.
        convert_pysph_output(in_path, dir_path, ftype=1, version='10')


# After setting up the framework, instantiate the class (which includes the
#  solver) and run it to get the solution.
if __name__ == '__main__':
    app = SDPySPHApplication()
    app.run()

    # Post-processing is added here
    app.post_process()
