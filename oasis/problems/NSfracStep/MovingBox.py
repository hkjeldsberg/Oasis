from oasis.common.utilities import AssignedVectorFunction
from oasis.problems.NSfracStep import *

set_log_level(99)


def problem_parameters(commandline_kwargs, NS_parameters, **NS_namespace):
    if "restart_folder" in commandline_kwargs.keys():
        restart_folder = commandline_kwargs["restart_folder"]
        restart_folder = path.join(getcwd(), restart_folder)

        f = open(path.join(path.dirname(path.abspath(__file__)), restart_folder,
                           'params.dat'), 'r')
        NS_parameters.update(cPickle.load(f))
        NS_parameters['T'] = 10
        NS_parameters['restart_folder'] = restart_folder
        globals().update(NS_parameters)

    else:
        T = 20
        dt = 0.01
        nu = 0.01
        NS_parameters.update(
            # Mesh parameters
            Nx=10,
            Ny=10,
            # Fluid properties
            nu=nu,
            # Simulation properties
            T=T,
            dt=dt,
            checkpoint=1000,
            save_step=10,
            print_intermediate_info=100,
            velocity_degree=1,
            pressure_degree=1,
            max_iter=2,
            max_error=1e-8,
            folder="moving_box_results",
            use_krylov_solvers=True)


def mesh(Nx, Ny, **NS_namespace):
    mesh = UnitSquareMesh(Nx, Ny)

    return mesh


def pre_boundary_condition(mesh, **NS_namespace):
    # Mark geometry
    inlet = AutoSubDomain(lambda x, b: b and x[0] <= DOLFIN_EPS)
    outlet = AutoSubDomain(lambda x, b: b and x[0] > 1 - DOLFIN_EPS)
    walls = AutoSubDomain(lambda x, b: b and (near(x[1], 0) or near(x[1], 1)))

    boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary.set_all(0)

    walls.mark(boundary, 1)
    inlet.mark(boundary, 2)
    outlet.mark(boundary, 3)

    return dict(boundary=boundary)


def create_bcs(V, Q, sys_comp, boundary, NS_expressions, **NS_namespace):
    info_red("Creating boundary conditions")
    f = 1 / 10
    A = 1 / 20
    outlet = Constant(0.0)

    inlet = Constant(0.0)
    outlet = Expression("f * 2 * pi * A * sin(2 * pi * f * t)", degree=2, t=0, f=f, A=A)
    noslip = Constant(0.0)

    NS_expressions["inlet"] = inlet
    NS_expressions["outlet"] = outlet

    # Velocity
    bcu_wall = DirichletBC(V, noslip, boundary, 1)

    bcu_in_x = DirichletBC(V, NS_expressions["inlet"], boundary, 2)
    bcu_in_y = DirichletBC(V, Constant(0), boundary, 2)

    bcu_out_x = DirichletBC(V, NS_expressions["outlet"], boundary, 3)
    bcu_out_y = DirichletBC(V, Constant(0), boundary, 3)

    # Pressure
    bcp = DirichletBC(Q, Constant(0), boundary, 2)

    bcs = dict((ui, []) for ui in sys_comp)
    # bcs['u0'] = [bcu_out_x, bcu_wall, bcu_in_x]
    # bcs['u1'] = [bcu_out_y, bcu_wall, bcu_in_y]
    bcs['u0'] = [bcu_out_x, bcu_wall]
    bcs['u1'] = [bcu_out_y, bcu_wall]
    bcs["p"] = [bcp]

    return bcs


def pre_solve_hook(V, mesh, newfolder, velocity_degree, wu_, x_, u_components, boundary, **NS_namespace):
    """Called prior to time loop"""
    viz_u = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "velocity.xdmf"))
    viz_p = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "pressure.xdmf"))

    for viz in [viz_u, viz_p]:
        viz.parameters["rewrite_function_mesh"] = True
        viz.parameters["flush_output"] = True
        viz.parameters["functions_share_mesh"] = True

    Vv = VectorFunctionSpace(mesh, "CG", velocity_degree)
    DG = FunctionSpace(mesh, "DG", velocity_degree)
    u_vec = Function(Vv, name="u")

    # Set up mesh solver
    u_mesh, v_mesh = TrialFunction(V), TestFunction(V)

    A = CellDiameter(mesh)
    D = project(A, DG)
    D_arr = D.vector().get_local()

    # Note: Double check if alfa changes when mesh moves. I do not think so as long as
    # project is here and not inside time loop
    # Set alfa to something 1/distace, computed by laplace.
    alfa = Constant((D_arr.max() - D_arr.min()) ** 4) / D ** 4
    a_mesh = inner(alfa * grad(u_mesh), grad(v_mesh)) * dx

    L_mesh = dict((ui, Vector(x_[ui])) for ui in u_components)

    # Inlet, walls and outlet
    rigid_bc_walls = DirichletBC(V, Constant(0), boundary, 1)

    bc_in_x = DirichletBC(V, NS_expressions["inlet"], boundary, 2)
    bc_in_y = DirichletBC(V, Constant(0.0), boundary, 2)

    bc_out_x = DirichletBC(V, NS_expressions["outlet"], boundary, 3)
    bc_out_y = DirichletBC(V, Constant(0.0), boundary, 3)

    # For moving the mesh
    position = AssignedVectorFunction(wu_)

    bc_mesh = dict((ui, []) for ui in u_components)
    rigid_bc = [rigid_bc_walls]
    bc_mesh["u0"] = [bc_in_x, bc_out_x] + rigid_bc
    bc_mesh["u1"] = [bc_in_y, bc_out_y] + rigid_bc

    mesh_prec = PETScPreconditioner("hypre_amg")
    mesh_sol = PETScKrylovSolver("gmres", mesh_prec)

    krylov_solvers = dict(monitor_convergence=False,
                          report=False,
                          error_on_nonconvergence=False,
                          nonzero_initial_guess=True,
                          maximum_iterations=20,
                          relative_tolerance=1e-8,
                          absolute_tolerance=1e-8)

    mesh_sol.parameters.update(krylov_solvers)
    coordinates = mesh.coordinates()
    dof_map = vertex_to_dof_map(V)

    return dict(viz_p=viz_p, viz_u=viz_u,
                u_vec=u_vec, mesh_sol=mesh_sol, bc_mesh=bc_mesh,
                dof_map=dof_map, a_mesh=a_mesh, L_mesh=L_mesh,
                coordinates=coordinates, position=position, alfa=alfa)


def update_prescribed_motion(t, dt, w_, dof_map, coordinates, wx_, u_components, mesh_sol, bc_mesh, NS_expressions,
                             A_cache, position, a_mesh,
                             L_mesh, mesh, **NS_namespace):
    # Update time
    for key, value in NS_expressions.items():
        if key == "outlet":
            NS_expressions[key].t = t

    move = False
    for ui in u_components:
        # Solve for d and w
        A_mesh = A_cache[(a_mesh, tuple(bc_mesh[ui]))]
        [bc.apply(L_mesh[ui]) for bc in bc_mesh[ui]]
        mesh_sol.solve(A_mesh, wx_[ui], L_mesh[ui])

        # Move mesh
        arr = w_[ui].vector().get_local()
        mesh_tolerance = 1e-15
        if mesh_tolerance < abs(arr.min()) + abs(arr.max()):
            coordinates[:, int(ui[-1])] += (arr * dt)[dof_map]
            move = True

    return move


def temporal_hook(t, q_, viz_u, viz_p, u_vec, **NS_namespace):
    assign(u_vec.sub(0), q_["u0"])
    assign(u_vec.sub(1), q_["u1"])

    viz_u.write(u_vec, t)
    viz_p.write(q_["p"], t)
