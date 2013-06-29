__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2011-12-19"
__copyright__ = "Copyright (C) 2011 " + __author__
__license__  = "GNU Lesser GPL version 3 or any later version"

"""
This is a highly tuned and stripped down Navier-Stokes solver optimized
for both speed and memory. The algorithm used is a second order in time 
fractional step method (incremental pressure correction).

Crank-Nicolson discretization is used in time of the Laplacian and 
the convected velocity. The convecting velocity is computed with an 
Adams-Bashforth projection. The fractional step method can be used
both non-iteratively or with iterations over the pressure velocity 
system.

The velocity vector is segregated, and we use three scalar velocity 
components

V = FunctionSpace(mesh, 'CG', 1)
u_components = ['u0', 'u1', 'u2'] in 3D, ['u0', 'u1'] in 2D
q_[ui] = Function(V) for ui = u_components
u_ = as_vector(q_['u0'], q_['u1'], q_['u2'])

A single coefficient matrix is assembled and used by all velocity 
componenets. It is built by preassembling as much as possible. 

The system of momentum equations solved are
u = TrialFunction(V)
v = TestFunction(V)
U = 0.5*(u+q_1['u0'])     # Scalar
U1 = 1.5*u_1 - 0.5*u_2    # Vector
F = (1/dt)*inner(u - u_1, v)*dx + inner(grad(U)*U1, v)*dx + inner(p_.dx(0), v)*dx \
     nu*inner(grad(U), grad(v))*dx + inner(f[0], v)*dx

where (q_['u0'], p_.dx(0), f[0]) is replaced by (q_1['u1'], p_.dx(1), f[1]) and 
(q_1['u2'], p_.dx(2), f[3]) for the two other velocity components.
We solve an equation corresponding to lhs(F) == rhs(F) for all ui.
     
The variables u_1 and u_2 are velocity vectors at time steps k-1 and k-2. We 
are solving for u, which is the velocity at time step k. p_ is the latest 
approximation for the pressure.

The matrix corresponding to assemble(lhs(F)) is the same for all velocity
components and it is computed as:

    A  = 1/dt*M + 0.5*Ac + 0.5*nu*K
    
where

    M  = assemble(inner(u, v)*dx)
    Ac = assemble(inner(grad(u)*U1, v)*dx)
    K  = assemble(inner(grad(u), grad(v))*dx)

However, we start by assembling a coefficient matrix (A_rhs) that is used 
to compute parts of the rhs vector corresponding to mass, convection
and diffusion:

    A_rhs = 1/dt*M - 0.5*Ac - 0.5*nu*K
    b[ui]  = A_rhs*q_1[ui].vector()

The pressure gradient and body force needs to be added to b as well. Three
matrices are preassembled for the computation of the pressure gradient:

  P = dict((ui, assemble(v*p.dx(i)*dx)) for i, ui in enumerate(u_components))

and the pressure gradient for each component of the momentum equation is 
then computed as

  assemble(p_.dx(i)*v*dx) = P[ui] * p_.vector()

Ac needs to be reassembled each new timestep. Ac is assembled into A to 
save memory. A and A_rhs are recreated each new timestep by assembling Ac, 
setting up A_rhs and then using the following to create A:

   A = -A_rhs + 2/dt*M

We then solve the linear system A * u = b[ui] for all q_[ui].vector()

Pressure is solved through

  inner(grad(p), grad(q))*dx == inner(grad(p_), grad(q))*dx - 1/dt*inner(div(u_), q)*dx

Here we assemble the rhs by:

  Ap = assemble(inner(grad(p), grad(q))*dx)
  bp = Ap * p_.vector()
  for ui in u_components:
    bp.axpy(-1./dt, Rx[ui]*x_[ui])
  where the preassemble Rx is:
    Rx = dict((ui, assemble(q*u.dx(i)*dx)) for i, ui in  enumerate(u_components))

We then solve Ap * p = bp for p_.vector().
  
Velocity update is computed through:

  inner(u, v)*dx == inner(q_[ui], v)*dx - dt*inner(dp_.dx(i), v)*dx

where each component on the rhs of the equation is computed effectively as
  inner(q_[ui], v)*dx = M * q_[ui].vector()
  dt*inner(dp_.dx(i), v)*dx = dt * P[ui] * dp_.vector()

where dp_ is the pressure correction, i.e., th newly computed pressure 
at the new timestep minus the pressure at previous timestep.

The lhs mass matrix is either the regular M, or the lumped and diagonal
mass matrix ML computed as
  ones = Function(V)
  ones.vector()[:] = 1.
  ML = M * ones.vector()

"""
################### Problem dependent parameters ####################

#from DrivenCavity import *
from Channel import *
#from LaminarChannel import *
#from Lshape import *
#from TaylorGreen2D import *
#from TaylorGreen3D import *

#####################################################################
### Should import a mesh and a dictionary called NS_parameters    ###
assert(isinstance(NS_parameters, dict))
assert(isinstance(mesh, Mesh))

if NS_parameters['velocity_degree'] > 1:
    NS_parameters['use_lumping_of_mass_matrix'] = False
vars().update(NS_parameters)  # Put NS_parameters in global namespace
parameters['krylov_solver'].update(krylov_solvers)

# Check how much memory is actually used by dolfin before we allocate anything
dolfin_memory_use = getMyMemoryUsage()
info_red('Memory use of plain dolfin = ' + dolfin_memory_use)

# Declare solution Functions and FunctionSpaces
V = FunctionSpace(mesh, 'CG', velocity_degree, constrained_domain=constrained_domain)
Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree, constrained_domain=constrained_domain)
if velocity_degree == pressure_degree:
    Q = V
else:
    Q = FunctionSpace(mesh, 'CG', pressure_degree, constrained_domain=constrained_domain)
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

dim = mesh.geometry().dim()
u_components = map(lambda x: 'u'+str(x), range(dim))
sys_comp =  u_components + ['p']

# Use dictionaries to hold all Functions and FunctionSpaces
VV = dict((ui, V) for ui in u_components); VV['p'] = Q

# Start from previous solution if restart_folder is given
if restart_folder:
    q_  = dict((ui, Function(VV[ui], path.join(restart_folder, ui + '.xml.gz'))) for ui in sys_comp)
    q_1 = dict((ui, Function(V, path.join(restart_folder, ui + '.xml.gz'))) for ui in u_components)
    try: # Check if there's a previous solution stored as well
        q_2 = dict((ui, Function(V, path.join(restart_folder, ui + '_1.xml.gz'))) for ui in u_components)
    except:
        q_2 = dict((ui, Function(V, path.join(restart_folder, ui + '.xml.gz'))) for ui in u_components)
else:
    q_  = dict((ui, Function(VV[ui])) for ui in sys_comp)
    q_1 = dict((ui, Function(V)) for ui in u_components)
    q_2 = dict((ui, Function(V)) for ui in u_components)
    
u_  = as_vector([q_[ui]  for ui in u_components]) # Velocity vector at t
u_1 = as_vector([q_1[ui] for ui in u_components]) # Velocity vector at t - dt
u_2 = as_vector([q_2[ui] for ui in u_components]) # Velocity vector at t - 2*dt

x_  = dict((ui, q_ [ui].vector()) for ui in sys_comp)     # Solution vectors t
x_1 = dict((ui, q_1[ui].vector()) for ui in u_components) # Solution vectors t - dt
x_2 = dict((ui, q_2[ui].vector()) for ui in u_components) # Solution vectors t - 2*dt

p_  = q_['p']               # pressure at t - dt/2
dp_ = Function(Q)           # pressure correction

###################  Boundary conditions  ###########################

bcs = create_bcs(**vars())

###################  Initialize solution ############################

initialize(**vars())

################### Fetch linear solvers  ###########################

u_sol, p_sol, du_sol = get_solvers(**vars())

#####################################################################

# Preassemble constant pressure gradient matrix
P = dict((ui, assemble(v*p.dx(i)*dx)) for i, ui in enumerate(u_components))

# Preassemble velocity divergence matrix
if velocity_degree == pressure_degree:
    Rx = P
else:
    Rx = dict((ui, assemble(q*u.dx(i)*dx)) for i, ui in  enumerate(u_components))

# Preassemble some constant in time matrices
M = assemble(inner(u, v)*dx)                    # Mass matrix
K = assemble(inner(grad(u), grad(v))*dx)        # Diffusion matrix without viscosity coefficient
if velocity_degree == pressure_degree and bcs['p'] == []:
    Ap = K
else:
    Ap = assemble(inner(grad(q), grad(p))*dx)   # Pressure Laplacian
A = Matrix()                                    # Coefficient matrix (needs reassembling)

# Velocity update uses lumping of the mass matrix for P1-elements
# Compute inverse of the lumped diagonal mass matrix 
if use_lumping_of_mass_matrix:
    # Create vectors used for lumping mass matrix
    ones = Function(V)
    ones.vector()[:] = 1.
    ML = M * ones.vector()
    ML.set_local(1. / ML.array())
else:
    # Use regular mass matrix for velocity update
    [bc.apply(M) for bc in bcs['u0']]

# Apply boundary conditions on Ap that is used directly in solve
if bcs['p']:
    [bc.apply(Ap) for bc in bcs['p']]
    Ap.compress()
    
# Adams Bashforth projection of velocity at t - dt/2
U_ = 1.5*u_1 - 0.5*u_2

# Convection form
a  = 0.5*inner(v, dot(U_, nabla_grad(u)))*dx

#### Get any constant body forces ####
f = body_force(**vars())
######################################

# Preassemble constant body force
assert(isinstance(f, Constant))
b0 = dict((ui, assemble(v*f[i]*dx)) for i, ui in enumerate(u_components))

b   = dict((ui, Vector(x_[ui])) for ui in sys_comp)       # rhs vectors
bold= dict((ui, Vector(x_[ui])) for ui in sys_comp)       # rhs temp storage vectors
work = Vector(x_['u0'])

stop = False
reset_sparsity = True
t1 = time.time(); old_tstep = tstep
print_solve_info = use_krylov_solvers and krylov_solvers['monitor_convergence']

############ Do something problem specific ####
vars().update(pre_solve_hook(**vars()))
###############################################
tin = time.time()
while t < (T - tstep*DOLFIN_EPS) and not stop:
    t += dt
    tstep += 1
    inner_iter = 0
    err = 1e8
    num_iter = max(iters_on_first_timestep, max_iter) if tstep == 1 else max_iter
    #############################
    start_timestep_hook(**vars())
    #############################
    while err > max_error and inner_iter < num_iter:
        err = 0
        inner_iter += 1
        ### Assemble matrices and compute rhs vector for tentative velocity ###
        if inner_iter == 1:
            # Set up A only on the first iteration because nothing here is changing in time
            # Set up coefficient matrix for computing the rhs:
            A = assemble(a, tensor=A, reset_sparsity=reset_sparsity) 
            reset_sparsity = False   
            A._scale(-1.)               # Negative convection on the rhs 
            A.axpy(1./dt, M, True)      # Add mass
            A.axpy(-0.5*nu, K, True)    # Add diffusion
            
            # Compute rhs for all velocity components
            for ui in u_components:
                b[ui][:] = b0[ui][:]
                b[ui].axpy(1., A*x_1[ui])

            # Reset matrix for lhs
            A._scale(-1.)
            A.axpy(2./dt, M, True)
            [bc.apply(A) for bc in bcs['u0']]
        
        t0 = time.time()
        for ui in u_components:
            bold[ui][:] = b[ui][:] 
            b[ui].axpy(-1., P[ui]*x_['p'])       # Add pressure gradient
            [bc.apply(b[ui]) for bc in bcs[ui]]
            work[:] = x_[ui][:]
            info_blue('Solving tentative velocity '+ui, inner_iter == 1 and print_solve_info)
            #################################
            velocity_tentative_hook(**vars())
            #################################
            u_sol.solve(A, x_[ui], b[ui])
            b[ui][:] = bold[ui][:]  # store preassemble part
            err += norm(work - x_[ui])
        u_sol.t += (time.time()-t0)
        
        ### Solve pressure ###
        t0 = time.time()
        dp_.vector()[:] = x_['p'][:]
        b['p'][:] = 0.
        for ui in u_components:
            b['p'].axpy(-1./dt, Rx[ui]*x_[ui]) # Divergence of u_
        b['p'].axpy(1., Ap*x_['p'])
        [bc.apply(b['p']) for bc in bcs['p']]
        rp = residual(Ap, x_['p'], b['p'])
        info_blue('Solving pressure', inner_iter == 1 and print_solve_info)
        #######################
        pressure_hook(**vars())
        #######################
        p_sol.solve(Ap, x_['p'], b['p'])
        if normalize: normalize(x_['p'])
        dp_.vector()[:] = x_['p'][:] - dp_.vector()[:]
        if num_iter > 1:
            if inner_iter == 1: 
                info_blue('  Inner iterations velocity pressure:')
                info_blue('                 error u  error p')
            info_blue('    Iter = {0:4d}, {1:2.2e} {2:2.2e}'.format(inner_iter, err, rp))
        p_sol.t += (time.time()-t0)
        
    # Update velocity
    if use_lumping_of_mass_matrix:
        for ui in u_components:
            x_[ui].axpy(-dt, (P[ui] * dp_.vector()) * ML)
            [bc.apply(x_[ui]) for bc in bcs[ui]]
    else:
        t0 = time.time()
        for ui in u_components:
            b[ui][:] = M*x_[ui][:]        
            b[ui].axpy(-dt, P[ui]*dp_.vector())
            [bc.apply(b[ui]) for bc in bcs[ui]]        
            info_blue('Updating velocity '+ui, print_solve_info)
            ##############################
            velocity_update_hook(**vars())
            ##############################
            du_sol.solve(M, x_[ui], b[ui])
        du_sol.t += (time.time()-t0)
    
    # Update to a new timestep
    for ui in u_components:
        x_2[ui][:] = x_1[ui][:]
        x_1[ui][:] = x_ [ui][:]
        
    # Print some information and save solution
    if tstep % save_step == 0 or tstep % checkpoint == 0:
        info_green('Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}'.format(t, tstep, T)) 
        tottime = time.time() - t1    
        info_red('Total computing time on previous {0:d} timesteps = {1:f}'.format(tstep - old_tstep, tottime))
        save_solution(**vars())
        t1 = time.time(); old_tstep = tstep
    #####################################################
    temporal_hook(**vars())
    stop = check_if_kill(tstep, t, q_, q_1, NS_parameters)
    ######################################################
        
info_red('Total computing time = {0:f}'.format(time.time() - tin))
print 'Additional memory use of processor = {0}'.format(eval(getMyMemoryUsage()) - eval(dolfin_memory_use))
mymem = eval(getMyMemoryUsage())-eval(dolfin_memory_use)
info_red('Total memory use of solver = ' + str(MPI.sum(mymem)))
list_timings()

###### Final hook ######        
theend(**vars())
