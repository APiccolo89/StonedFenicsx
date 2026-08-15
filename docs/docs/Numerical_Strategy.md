# How StonedFEniCSx works? 

**StonedFEniCSx** is a numerical code that is written in Python, and it highly relies on the FEniCSx library. The code is conceived to solve *problems*. A *problem* is a set of differential equations that are solved in a specific domain. For example: the stokes problem is solved only in the `subducting_plate_domain` and in the `wedge_domain`; the lithostatic pressure problem and energy problem are solved in the `Global_domain`. Thus, the problem are small solvers that accept the geometry of the domain, and solves the differential equation as a function of the specific of the problem. 

**StonedFEniCSx** can solve the steady-state and time-dependent problems. 

**StonedFEniCSx** reads the input, generate a user-layer in which the user can modify the *input.yml* file, then thanks to a top-level function, the simulation is configured and run it. 

The `subducting_plate_domain` stokes problem is solved once per numerical simulation, unless the velocity of the slab changes over-time in the time-dependent solution. This saves computation time, and it has been designed in the following manner, because the kinematic boundary condition over-constrain the velocity field of the slab, and the deviatoric stress-strain within the slab domain are not reliable. `wedge_domain` is solved only one time if the rheology is linear, otherwise is solved every-iteration. 

The lithostatic-pressure problem is solved once per timestep in case of time-dependent problem, while, in steady-state problem is solved at every iteration. The variation of lithostatic pressure in the time-dependent problems is negligible between time-step, and it is necessary to compute once. The energy problem is solved each time-step. 

The information between the domains are interpolated back-forth between the meshes of each sub-domains. 

