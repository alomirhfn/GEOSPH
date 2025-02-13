TXT INPUT FILE:
  - The only parameters you really control from the TXT file are the initial particle spacing and the speed of sound.
  - The other parameters need to be input into the pysph_run.sh file.

CSV INPUT FILE:
  - X, Y, Z: Particle positions
  - u, v, w: Particle velocity components
  - nw: Porosity (not used in current version)
  - mass: Particle mass (kept constant)
  - rho: Particle mass density
  - energy: Not used (total particle energy)
  - fx, fy, fz: Body force components (including gravity)
  - sxx, syy, ..., syz: Cauchy stress tensor components
  - exx, eyy, ..., eyz: Total deformation tensor components
  - nx, ny, nz: Normal vector components (used in particle shifting and for boundary particles)
  - bc: Type of boundary condition if using DummyParticle() | 0 = No-slip, 1 = Free slip
  - young: Elastic modulus of the material
  - poisson: Coefficient of Poisson
  - phi: Mohr-Coulomb internal friction angle
  - cohesion: Mohr-Coulomb cohesion
  - psi: Dilatancy angle
  - h_mod: Deprecated (keep zero)
  - To use MCC constitutive model other variables are needed in the CSV. See implementation and sd_pysph.py for what is needed