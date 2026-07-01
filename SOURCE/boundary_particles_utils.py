import numpy as np
from pysph.base import nnps
from pysph.base.kernels import WendlandQuintic, CubicSpline
# from cyarray.carray import DoubleArray  # For HPC simulations
# from pyzoltan.core.carray import UIntArray  # For laptop simulations
from cyarray.carray import UIntArray  # For HPC simulations
from math import *

r"""
    ** Pre-processing of particles to determine mirror (ghost) particles **
    
    The methods implemented below are to determine unique mirror (or ghost
    particles) for each boundary particle. The mirror particles are positioned 
    inside the problem domain and used to extrapolate the properties of fluid 
    or sediment particles to the dummy boundaries. In particular, they are used
    to extrapolate the pore-water pressure field for the coupled u-p or u-w-p 
    formulations implemented.
    
    For more information about the mirroring process, refer to the paper of 
    Chow et al. (2018).
    
    Note: The main difference of the implementation here and that of the paper
    is that instead of using boundary particles on the actual position of the 
    physical boundary and using those as the base point for the mirroring 
    process, here, we use the closest fluid/sediment particle as the base point     

    References
    ----------
    .. [Chowetal2018]
    A.D. Chow, B.D. Rogers, S.J. Lind, P.K. Stansby, "Incompressible SPH (ISPH)
    with fast Poisson solver on a GPU", "Computer Physics Communications", 226, 
    81–103, https://doi.org/10.1016/j.cpc.2018.01.005.
"""

def boundary_normals(particles, boundary, kernel, dim, radius):
    r"""
    This function calculates the normals to the boundary particles using the
    kernel gradient of the particles. As a result, particles near edges have
    normals that are not aligned with the orthogonal Cartesian axes, which is
    equivalent to having more-or-less round corners and non-planar surfaces.

    @param array particles: List of particle types with positions 0 = Boundary
        and 1 = Barrier (Assumed to be always in this order!)
    @param ParticleArray boundary: Particle array of boundary or barrier
        particles
    @param KernelObject kernel: PySPH kernel object
    @param int dim: Spatial dimension of the simulation (2 or 3D)
    @param double radius: Radius of the kernel (between 1.0 and 4.0)
    @return: Array of normalized normal vectors to boundary particles.
    """

    # Identify which type of boundary particle was passed to the caller and
    #  assign the index for the destination when computing neighbors.
    pos = 0
    if boundary.name == 'barrier':
        pos = 1

    # Create a kernel
    kernel = kernel(dim)

    # Number of boundary particles
    nparts = boundary.get_number_of_particles()

    # NumPy array to hold the normal vectors
    normals = np.empty((nparts, 3))

    # Special c-array that holds the list of neighbors' indices
    # nbrs = DoubleArray()
    nbrs = UIntArray()

    # Find nearest neighbors for all particles
    nps = nnps.LinkedListNNPS(dim=dim, particles=particles,
                              radius_scale=radius)

    # Loop over all types of boundaries in particles and then over all
    #  particles of that kind to the neighbors
    for i in range(nparts):

        # Initialize normal vector
        n = [0, 0, 0]

        # Position vector of current boundary particle
        rb = np.asarray(
            [[boundary.x[i]], [boundary.y[i]], [boundary.z[i]]]
        )

        for j in range(len(particles)):

            # Find neighbors for current source (j) and particle (i)
            nps.get_nearest_particles(j, pos, i, nbrs)
            neighbors = nbrs.get_npy_array()  # Neighbor particles index array

            # Position vectors of all neighbors
            r = np.asarray(
                [(particles[j].x)[neighbors],
                (particles[j].y)[neighbors],
                (particles[j].z)[neighbors]]
            )

            # Masses of neighbors
            m = np.asarray((particles[j].m)[neighbors])

            # Mass densities of neighbors
            rho = np.asarray((particles[j].rho)[neighbors])

            # Smoothing length of neighbors
            h = np.asarray((particles[j].h)[neighbors])

            # Vectors of relative position between current particle and
            #   neighbors
            r_ij = r - rb

            # Scalar distances between current particle and neighbors
            rij = np.linalg.norm(r_ij,axis=0)

            # ====================== NORMAL CALCULATIONS ======================

            dwij = np.zeros(3)
            for k in range(neighbors.size):

                # Calculate kernel gradients for all neighbors
                kernel.gradient(r_ij[:,k],rij[k],h[k],dwij)

                # Update normal vector
                n += m[k] * dwij / rho[k]

        # Normalize the normal vector
        n /= np.linalg.norm(n) + 1e-12

        # Eliminate very small deviations
        # TODO: This has seemed to increase the accuracy of the normal
        #  prediction. However, it has not been tested extensively.
        #  Additionally, before this change, some random particles were not
        #  getting correct normals. Perhaps the base formulation already
        #  contains errors. Need to check in the future for robustness. This
        #  was added (08/29/2023).
        idx = np.where(np.abs(n) > 0.9)[0]
        sign = 1 * np.sign(n[idx])
        if idx.size > 0:
            n[:] = 0.0
            n[idx] = sign * 1.0

        # Add normal vector to resultant array
        normals[i] = n

    return normals.ravel()

def find_pivot_particles(boundary, dp, dim):
    r"""
        Find whether a boundary particle is on the outer layer of the boundary
        or if it is an inside boundary particle. If it is an inside
        boundary, finds the index of the closest outer boundary particle to
        serve as the pivot point to calculate the position of its mirror
        particle. If the particle is on the outer layer, the neighbor index
        is assigned -1.

        @return: List of outer neighbor index for each boundary particle.
    """

    # List of PySPH Particle Arrays
    particles = [boundary]

    # Find nearest neighbors for all particles
    nps = nnps.LinkedListNNPS(dim=dim, particles=particles, radius_scale=1.5)

    # Special c-array that holds the list of neighbors' indices
    # nbrs = DoubleArray()
    nbrs = UIntArray()

    # Number of boundary particles
    nparts = boundary.num_real_particles

    # Dictionary to keep track of groups of boundary particles for each closest
    #  fluid/sediment
    gpart = {}

    # Loop over each boundary particle and get the list of fluid/sediment
    #  neighbors
    for i in range(nparts):
        nps.get_nearest_particles(0, 0, i, nbrs)  # 0 - Source, 1 - Destination
        neighbors = nbrs.get_npy_array()  # Boundary/Boundary

        # Exclude self from neighbor list
        self_idx = np.where(boundary.gid[neighbors] == boundary.gid[i])
        neighbors = np.delete(neighbors, self_idx)

        # Normal vector of current boundary particle
        idx = 3*i
        n = np.asarray([[boundary.n[idx]], [boundary.n[idx + 1]],
                        [boundary.n[idx + 2]]])

        # Position vector of current boundary particle
        rb = np.asarray([[boundary.x[i]], [boundary.y[i]], [boundary.z[i]]])

        # Position vectors of neighbors
        r = np.asarray([boundary.x[neighbors], boundary.y[neighbors],
                        boundary.z[neighbors]])

        # Calculate projection point in the normal direction
        rp = rb + 2 * dp * n

        # Indices of neighbor particles that are close enough to the projection
        #  point but also in the direction of the normal passing through the
        #  particle
        idx1 = np.where(np.linalg.norm(n * (rp - r), axis=0) > 0.01 * dp)[0]
        idx2 = np.where(np.linalg.norm(rp - r, axis=0) < 1.01 * dp)[0]

        # Find closest neighbor closest to the projection point making sure
        #  that if the particle is not in the direction of the normal,
        #  then there is no neighbor and the particle is an outer particle
        if idx1.size > 0 and idx2.size > 0:
            idx3 = np.intersect1d(idx1, idx2)

            if idx3.size == 1:
                f_idx = idx3[0]

            # For particles that will have multiple reference neighbors (normal
            #  points inside the boundary or more than one outer particle)
            elif idx3.size > 1:
                f_idx = idx3[np.argmin(np.linalg.norm(rb - r, axis=0)[idx3])]
            else:
                f_idx = -1

            if f_idx >= 0:
                jdx = neighbors[f_idx]
            else:
                jdx = -1

        else:
            jdx = -1

        # Update dictionary of reference neighbors
        gpart[i] = jdx

    # Make sure middle particles have the right reference neighbor by
    #  checking if their reference has a reference itself. If that is the
    #  case, assign that latter particle as the final reference.
    for i in range(nparts):
        nbr = gpart[i]

        # Check if particle is an inner particle (has neighbor)
        if nbr >= 0:
            nbr_nbr = gpart[nbr]

            # Check that neighbor's neighbor is also inner particle (has
            #  neighbor)
            if nbr_nbr >= 0:
                nbr_nbr_nbr = gpart[nbr_nbr]

                # Check one more time for particles in the middle layer
                if nbr_nbr_nbr >= 0:

                    # Assign neighbor's neighbor's neighbor as reference
                    #  neighbor
                    gpart[i] = nbr_nbr_nbr

                # Assign neighbor's neighbor as reference neighbor
                else:
                    gpart[i] = nbr_nbr

    return gpart

def mirror_particles_positions(boundary, dp, dim):
    r'''
        Determine a mirror particle and its position within the problem domain
        for each boundary particle

        Note: Main function, called within the code.
    '''

    # Identify the reference neighbor particle to use as "pivot"
    parts_dict = find_pivot_particles(boundary, dp, dim)

    # Dictionary of boundary particles to mirrored point coordinate
    mirror_dict = {}

    # List of keys from the dictionary of the closest fluid/sediment particles
    keys = parts_dict.keys()

    # Loop over all boundary particles in the dictionary
    for key in keys:

        # Normal vector to the boundary particle to be mirrored
        idx = 3 * key
        nb = np.asarray([boundary.n[idx], boundary.n[idx + 1],
                        boundary.n[idx + 2]])

        # Position vector of the boundary particle to be mirrored
        rb = np.asarray([boundary.x[key],boundary.y[key],boundary.z[key]])

        # Get boundary particle neighbor (pivot particle) of the current part
        nbr = parts_dict[key]

        # Position vector of the pivot particle
        r = np.asarray([boundary.x[nbr], boundary.y[nbr], boundary.z[nbr]])
        if nbr < 0:
            r = rb

        # Normal vector to the pivot particle
        idx = 3 * nbr
        n = np.asarray([boundary.n[idx], boundary.n[idx + 1],
                        boundary.n[idx + 2]])

        # Check whether it is outer or inner particle, and calculate mirror
        #  particle position accordingly
        if nbr < 0:
            # Coefficient to correct position of mirror particles of outer
            #  particles with diagonal normals
            mult = 1

            # Check whther particle is diagonal
            if np.where(np.abs(nb) > 0)[0].size > 1:
                mult = sqrt(2)

            rm = rb + mult * dp * nb
        else:
            # Calculate a combined normal vector. This is to avoid mirror
            #  particles within the boundary for some of the boundary particles
            #  that have normals pointing inwards or very different from their
            #  pivot particle's normal. Should not affect the outer particles
            #  or inner particles that have "well-behaved" normals.
            nv = (n + nb) / np.linalg.norm(n + nb)

            rm = r + (np.linalg.norm(nv * (r - rb)) + dp) * nv

        # Add mirror points and boundary particle indices to dictionary
        mirror_dict[key] = rm

    return mirror_dict
