import pickle

import numpy as np

from ..NSfracStep import *

set_log_level(99)


def problem_parameters(commandline_kwargs, NS_parameters, NS_expressions, **NS_namespace):
    if "restart_folder" in commandline_kwargs.keys():
        restart_folder = commandline_kwargs["restart_folder"]
        restart_folder = path.join(getcwd(), restart_folder)

        f = open(path.join(path.dirname(path.abspath(__file__)), restart_folder,
                           'params.dat'), 'r')
        NS_parameters.update(pickle.load(f))
        NS_parameters['T'] = 10
        NS_parameters['restart_folder'] = restart_folder
        globals().update(NS_parameters)

    else:
        NS_parameters.update(
            checkpoint=1000,
            save_step=10e10,
            print_intermediate_info=100,

            # Geometrical parameters
            diam=0.02,  # Diameter
            nu=4e-05,  # Kinematic viscosity
            U=1.0,  # Free-stream flow speed
            A_ratio=0.25,  # Amplitude ratio
            St=0.2280,  # Strouhal number
            F=1.00,  # Frequency ratio
            # Simulation parameters
            T=1,
            dt=5e-05,
            # dt=0.001,
            velocity_degree=1,
            pressure_degree=1,
            # mesh_path="mesh/cylinder_original.xml",
            mesh_path="mesh/cylinder_refined.xml",
            folder="oscillating_circle_results",
            use_krylov_solvers=True,
            max_iter=2,  # 20 # Vary this
            # max_error=1E-6,
            plot_mesh=False,
        )


def mesh(mesh_path, plot_mesh, **NS_namespace):
    """
    Import and plot mesh
    """
    mesh = Mesh(mesh_path)

    if plot_mesh:
        import matplotlib.pyplot as plt
        plot(mesh)
        plt.show()

    return mesh


def pre_boundary_condition(diam, mesh, **NS_namespace):
    H = 30 * diam / 2  # Height
    L = 63 * diam  # Length

    # Mark geometry
    inlet = AutoSubDomain(lambda x, b: b and x[0] <= DOLFIN_EPS)
    walls = AutoSubDomain(lambda x, b: b and (near(x[1], -H) or near(x[1], H)))
    circle = AutoSubDomain(lambda x, b: b and (-H / 2 <= x[1] <= H / 2) and (L / 16 <= x[0] <= L / 3))
    outlet = AutoSubDomain(lambda x, b: b and (x[0] > L - DOLFIN_EPS * 1000))

    boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary.set_all(0)

    inlet.mark(boundary, 1)
    walls.mark(boundary, 2)
    circle.mark(boundary, 3)
    outlet.mark(boundary, 4)

    return dict(boundary=boundary)


def create_bcs(V, Q, U, St, F, diam, A_ratio, sys_comp, boundary, NS_expressions, **NS_namespace):
    info_red("Creating boundary conditions")

    f_v = St * U / diam  # Fixed-cylinder vortex shredding frequency
    f_o = F * f_v  # Frequency of harmonic oscillation
    y_max = A_ratio * diam  # Max displacement (Amplitude)
    print("Frequency is %.4f" % f_o)
    print("Amplitude %.4f " % y_max)
    NS_expressions["circle_x"] = Constant(0)
    NS_expressions["circle_y"] = Expression('2 * pi * f_o * y_max* cos(2 * pi * f_o * t)', degree=2, t=0, y_max=y_max,
                                            f_o=f_o)
    # NS_expressions["circle_y"] = Constant(0)

    bcu_in_x = DirichletBC(V, Constant(U), boundary, 1)
    bcu_in_y = DirichletBC(V, Constant(0), boundary, 1)

    bcu_wall = DirichletBC(V, Constant(0), boundary, 2)
    bcu_circle_x = DirichletBC(V, NS_expressions["circle_x"], boundary, 3)
    bcu_circle_y = DirichletBC(V, NS_expressions["circle_y"], boundary, 3)

    bcp_out = DirichletBC(Q, Constant(0), boundary, 4)

    bcs = dict((ui, []) for ui in sys_comp)
    bcs['u0'] = [bcu_circle_x, bcu_in_x, bcu_wall]
    bcs['u1'] = [bcu_circle_y, bcu_in_y, bcu_wall]
    bcs["p"] = [bcp_out]

    return bcs


def pre_solve_hook(V, p_, u_, nu, mesh, newfolder, velocity_degree, u_components, boundary, **NS_namespace):
    """Called prior to time loop"""
    viz_u = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "velocity.xdmf"))
    viz_p = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "pressure.xdmf"))
    viz_w = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "mesh_velocity.xdmf"))

    for viz in [viz_u, viz_p, viz_w]:
        viz.parameters["rewrite_function_mesh"] = True
        viz.parameters["flush_output"] = True
        viz.parameters["functions_share_mesh"] = True

    Vv = VectorFunctionSpace(mesh, "CG", velocity_degree)
    DG = FunctionSpace(mesh, "DG", velocity_degree)
    u_vec = Function(Vv, name="u")

    # Facet normals
    n = FacetNormal(mesh)

    # Set up mesh solver
    u_mesh, v_mesh = TrialFunction(V), TestFunction(V)

    A = CellDiameter(mesh)
    D = project(A, DG)
    D_arr = D.vector().get_local()

    # Note: Set alfa to something 1/distace, computed by laplace.
    alfa = Constant(D_arr.max() ** 3 - D_arr.min() ** 3) / D ** 3
    a_mesh = inner(alfa * grad(u_mesh), grad(v_mesh)) * dx
    L_mesh = dict((ui, assemble(Function(V) * v_mesh * dx)) for ui in u_components)

    # Inlet, walls
    rigid_bc_in = DirichletBC(V, Constant(0), boundary, 1)
    rigid_bc_walls = DirichletBC(V, Constant(0), boundary, 2)
    circle_bc_x = DirichletBC(V, NS_expressions["circle_x"], boundary, 3)
    circle_bc_y = DirichletBC(V, NS_expressions["circle_y"], boundary, 3)
    rigid_bc_out = DirichletBC(V, Constant(0), boundary, 4)

    bc_mesh = dict((ui, []) for ui in u_components)
    rigid_bc = [rigid_bc_in, rigid_bc_walls, rigid_bc_out]
    bc_mesh["u0"] = [circle_bc_x] + rigid_bc
    bc_mesh["u1"] = [circle_bc_y] + rigid_bc

    # TODO: Profiling on this?
    mesh_prec = PETScPreconditioner("hypre_amg")  # In tests "sor" is faster. ilu
    mesh_sol = PETScKrylovSolver("gmres", mesh_prec)

    ds = Measure("ds", subdomain_data=boundary)

    R = VectorFunctionSpace(mesh, 'R', 0)
    c = TestFunction(R)
    tau = -p_ * Identity(2) + nu * (grad(u_) + grad(u_).T)
    forces = dot(dot(tau, n), c) * ds(3)

    # TODO: Increase tolerance?
    # TODO: less iterations?
    krylov_solvers = dict(monitor_convergence=False,
                          report=False,
                          error_on_nonconvergence=False,
                          nonzero_initial_guess=True,
                          maximum_iterations=20,
                          relative_tolerance=1e-8,
                          absolute_tolerance=1e-8)

    mesh_sol.parameters.update(krylov_solvers)
    coordinates = mesh.coordinates()
    if velocity_degree == 1:
        dof_map = vertex_to_dof_map(V)
    else:
        dof_map = V.dofmap().entity_dofs(mesh, 0)

    return dict(viz_p=viz_p, viz_u=viz_u, viz_w=viz_w, u_vec=u_vec, mesh_sol=mesh_sol, bc_mesh=bc_mesh, dof_map=dof_map,
                a_mesh=a_mesh, L_mesh=L_mesh, n=n, forces=forces, coordinates=coordinates)


def start_timestep_hook(t, NS_expressions, **NS_namespace):
    # TODO: Is this needed?
    NS_expressions["circle_y"].t = t


def update_prescribed_motion(t, dt, St, U, diam, F, A_ratio, wx_, w_, u_components, tstep, mesh_sol,
                             bc_mesh, NS_expressions, dof_map, A_cache,
                             a_mesh, L_mesh, mesh, coordinates, **NS_namespace):
    # f_v = St * U / diam  # Fixed-cylinder vortex shredding frequency
    # f_o = F * f_v  # Frequency of harmonic oscillation
    # y_max = A_ratio * diam  # Max displacement (Ampliture)

    # Update time
    NS_expressions["circle_y"].t = t
    # print(St, U, D, F, f_v, A_ratio, D)

    # print(y_max * cosine(2 * pii * f_o * t))

    move = False
    for ui in u_components:
        # Solve for d and w
        A_mesh = A_cache[(a_mesh, tuple(bc_mesh[ui]))]  # Assemble mesh form, and add bcs
        [bc.apply(L_mesh[ui]) for bc in bc_mesh[ui]]  # Apply bcs
        mesh_sol.solve(A_mesh, wx_[ui], L_mesh[ui])  # Solve for mesh

        # Move mesh
        arr = w_[ui].vector().get_local()
        # TODO: Increase tolerance?
        mesh_tolerance = 1e-15
        if mesh_tolerance < abs(arr.min()) + abs(arr.max()):
            coordinates[:, int(ui[-1])] += (arr * dt)[dof_map]
            move = True

    return move


def temporal_hook(t, tstep, V, w_, q_, max_iter, forces, f, U, diam, nu, viz_u, viz_p, viz_w, u_vec, newfolder,
                  **NS_namespace):
    viz_p.write(q_["p"], t)
    assign(u_vec.sub(0), q_["u0"])
    assign(u_vec.sub(1), q_["u1"])

    viz_u.write(u_vec, t)
    # viz_p.write(q_["p"], t)
    viz_w.write(w_["u0"], t)
    viz_w.write(w_["u1"], t)

    rho = 1000
    factor = 0.5 * rho * U ** 2 * diam
    Dr = assemble(forces).get_local() * rho  # Times constant fluid density
    print("Time:", round(t, 4), "Drag", Dr[0], "Lift", Dr[1], "Drag Coeff", Dr[0] / factor)

    # Write forces to file
    fluid_properties = np.asarray([round(t, 4), Dr[0], Dr[0] / factor, Dr[1], Dr[1] / factor])
    forces_path = path.join(newfolder, "VTK", "forces.txt")
    with open(forces_path, 'a') as filename:
        filename.write("{:.4f} {:.3f} {:.3f} {:.3f} {:.3f}\n".format(*fluid_properties))
