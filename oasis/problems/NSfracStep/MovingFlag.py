from pathlib import Path

import meshio
import numpy as np
import pygmsh

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
        dt = 0.05
        nu = 0.01
        NS_parameters.update(
            checkpoint=1000,
            save_step=10,
            print_intermediate_info=100,

            # Geometrical parameters
            H=12,
            L=23,
            b_dist=4.5,
            b_h=1,
            b_l=1,
            f_l=4.5,
            f_h=0.06,
            nu=nu,
            T=T,
            dt=dt,
            velocity_degree=1,
            pressure_degree=1,
            max_iter=2,
            max_error=1e-8,
            folder="moving_flag_results",
            lc=0.1,
            mesh_path="mesh/moving_flag.xdmf",
            use_krylov_solvers=True)


def mesh(mesh_path, lc, H, L, b_dist, b_h, b_l, f_l, f_h, **NS_namespace):
    # Name mesh after local cell length
    mesh_path = Path(mesh_path.format(lc))
    mesh_path.parent.mkdir(exist_ok=True, parents=True)

    if not mesh_path.exists():
        # Initialize geometry
        geom = pygmsh.built_in.Geometry()

        # Surounding box
        b0 = geom.add_point([0, 0, 0], lcar=lc * 2)
        b1 = geom.add_point([L, 0, 0], lcar=lc * 2)
        b2 = geom.add_point([L, H, 0], lcar=lc * 2)
        b3 = geom.add_point([0, H, 0], lcar=lc * 2)

        # Inner geometry
        f0 = geom.add_point([b_dist, H / 2 - b_h / 2, 0], lcar=lc / 3)
        f1 = geom.add_point([b_dist + b_l, H / 2 - b_h / 2, 0], lcar=lc / 3)
        f2 = geom.add_point([b_dist + b_l, H / 2 - f_h / 2, 0], lcar=lc / 3)
        f3 = geom.add_point([b_dist + b_l + f_l, H / 2 - f_h / 2, 0], lcar=lc / 3)
        f4 = geom.add_point([b_dist + b_l + f_l, H / 2 + f_h / 2, 0], lcar=lc / 3)
        f5 = geom.add_point([b_dist + b_l, H / 2 + f_h / 2, 0], lcar=lc / 3)
        f6 = geom.add_point([b_dist + b_l, H / 2 + b_h / 2, 0], lcar=lc / 3)
        f7 = geom.add_point([b_dist, H / 2 + b_h / 2, 0], lcar=lc / 3)

        # Draw lines
        l0 = geom.add_line(b0, b1)
        l1 = geom.add_line(b1, b2)
        l2 = geom.add_line(b2, b3)
        l3 = geom.add_line(b3, b0)

        l4 = geom.add_line(f0, f1)
        l5 = geom.add_line(f1, f2)
        l6 = geom.add_line(f2, f3)
        l7 = geom.add_line(f3, f4)
        l8 = geom.add_line(f4, f5)
        l9 = geom.add_line(f5, f6)
        l10 = geom.add_line(f6, f7)
        l11 = geom.add_line(f7, f0)

        # Gather lines
        ll_outer = geom.add_line_loop(lines=[l0, l1, l2, l3])
        ll_inner = geom.add_line_loop(lines=[l4, l5, l6, l7, l8, l9, l10, l11])

        # Set surface
        ps = geom.add_plane_surface(ll_outer, [ll_inner])

        # Mesh surface
        msh = pygmsh.generate_mesh(geom)

        for cell in msh.cells:
            if cell.type == "triangle":
                triangle_cells = cell.data

        for key in msh.cell_data_dict["gmsh:geometrical"].keys():
            if key == "triangle":
                triangle_data = msh.cell_data_dict["gmsh:geometrical"][key]

        triangle_mesh = meshio.Mesh(points=msh.points,
                                    cells={"triangle": triangle_cells},
                                    cell_data={"name_to_read": [triangle_data]})

        meshio.write(mesh_path.__str__(), triangle_mesh)

    mesh = Mesh()
    mesh_path = mesh_path.__str__()
    with XDMFFile(MPI.comm_world, mesh_path) as infile:
        infile.read(mesh)

    return mesh


class Flag_counter(UserExpression):
    def __init__(self, **kwargs):
        self.map = {}
        self.counter = -1
        super().__init__(**kwargs)

    def get_map(self):
        return self.map

    def eval(self, values, x):
        self.counter += 1
        self.map[self.counter] = [x[0], x[1]]


class Flag_y(UserExpression):
    def __init__(self, t, T_end, factor, x_hat_map, counter_max, **kwargs):
        self.factor = factor
        self.T_end = T_end
        self.t = t
        self.t_prev = 0
        self.A = 0
        self.A_t = 0
        self.map = x_hat_map
        self.max = counter_max
        self.counter = -1

        super().__init__(**kwargs)

    def get_A(self):
        if self.t < 2:
            return 0
        else:
            return self.factor * np.sin(2 * np.pi * (self.t - 2) / self.T_end)

    def update(self, t):
        # Set t
        self.t_prev = self.t
        self.t = t

        # Find velocity
        tmp = self.A_t
        self.A_t = self.get_A()
        diff = self.A_t - tmp
        self.A = diff / (self.t - self.t_prev)

    def eval(self, values, x):
        self.counter += 1
        values[:] = (self.map[self.counter][0] - 5.5) ** 2 * self.A
        if self.counter == self.max:
            self.counter = -1


class Flag_x(UserExpression):
    def __init__(self, t, T_end, factor, x_hat_map, counter_max, **kwargs):
        self.factor = factor
        self.T_end = T_end
        self.t = t
        self.A = 0
        self.A_t = 0
        self.map = x_hat_map
        self.max = counter_max
        self.counter = -1
        super().__init__(**kwargs)

    def get_A(self):
        if self.t < 2:
            return 0
        else:
            return self.factor * np.sin(2 * np.pi * (self.t - 2) / self.T_end) ** 2

    def update(self, t):
        # Set t
        self.t_prev = self.t
        self.t = t

        # Find velocity
        tmp = self.A_t
        self.A_t = self.get_A()
        diff = self.A_t - tmp
        self.A = diff / (self.t - self.t_prev)

    def eval(self, values, x, **kwargs):
        self.counter += 1
        values[:] = self.A * (self.map[self.counter][0] - 5.5) / 4.5
        if self.counter == self.max:
            self.counter = -1


def pre_boundary_condition(H, b_dist, b_l, f_h, L, mesh, **NS_namespace):
    # Mark geometry
    inlet = AutoSubDomain(lambda x, b: b and x[0] <= DOLFIN_EPS)
    walls = AutoSubDomain(lambda x, b: b and (near(x[1], 0) or near(x[1], H)))
    box = AutoSubDomain(lambda x, b: b and (1 <= x[1] <= 3 * H / 4) and (1 <= x[0] <= b_dist + b_l + DOLFIN_EPS * 1000))
    flag = AutoSubDomain(lambda x, b: b and (H / 2 - f_h - DOLFIN_EPS <= x[1] <= H / 2 + f_h + DOLFIN_EPS) and (
            b_dist + b_l - DOLFIN_EPS * 1000 <= x[0] <= L - 1))
    outlet = AutoSubDomain(lambda x, b: b and (x[0] > L - DOLFIN_EPS * 1000))

    boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary.set_all(0)

    inlet.mark(boundary, 1)
    walls.mark(boundary, 2)
    box.mark(boundary, 3)
    flag.mark(boundary, 4)
    outlet.mark(boundary, 5)

    return dict(boundary=boundary)


def create_bcs(V, Q, w_, sys_comp, boundary, NS_expressions, **NS_namespace):
    info_red("Creating boundary conditions")

    # Get x_hat (reference domain)
    flag_counter = Flag_counter(element=V.ufl_element())
    bc_tmp = DirichletBC(V, flag_counter, boundary, 4)
    bc_tmp.apply(w_["u0"].vector())
    w_["u0"].vector().zero()
    counter_max = flag_counter.counter
    x_hat_map = flag_counter.get_map()

    flag_x = Flag_x(0, 10, -0.344, x_hat_map, counter_max, element=V.ufl_element())
    flag_y = Flag_y(0, 10, 0.075, x_hat_map, counter_max, element=V.ufl_element())

    NS_expressions["flag_x"] = flag_x
    NS_expressions["flag_y"] = flag_y

    bcu_in_x = DirichletBC(V, Constant(1), boundary, 1)
    bcu_in_y = DirichletBC(V, Constant(0), boundary, 1)

    bcu_wall = DirichletBC(V, Constant(0), boundary, 2)
    bcu_box = DirichletBC(V, Constant(0), boundary, 3)

    bcu_flag_x = DirichletBC(V, flag_x, boundary, 4)
    bcu_flag_y = DirichletBC(V, flag_y, boundary, 4)

    bcp_out = DirichletBC(Q, Constant(0), boundary, 5)

    bcs = dict((ui, []) for ui in sys_comp)
    bcs['u0'] = [bcu_in_x, bcu_wall, bcu_box, bcu_flag_x]
    bcs['u1'] = [bcu_in_y, bcu_wall, bcu_box, bcu_flag_y]
    bcs["p"] = [bcp_out]

    return bcs


def pre_solve_hook(V, mesh, newfolder, velocity_degree, wu_, x_, u_components, boundary, **NS_namespace):
    """Called prior to time loop"""
    viz_d = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "deformation.xdmf"))
    viz_u = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "velocity.xdmf"))
    viz_p = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "pressure.xdmf"))
    viz_w = XDMFFile(MPI.comm_world, path.join(newfolder, "VTK", "mesh_velocity.xdmf"))

    for viz in [viz_d, viz_u, viz_p, viz_w]:
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

    # Inlet, walls and box
    rigid_bc_in = DirichletBC(V, Constant(0), boundary, 1)
    rigid_bc_walls = DirichletBC(V, Constant(0), boundary, 2)
    rigid_bc_out = DirichletBC(V, Constant(0), boundary, 3)

    # Flag
    flag_bc_x = DirichletBC(V, NS_expressions["flag_x"], boundary, 4)
    flag_bc_y = DirichletBC(V, NS_expressions["flag_y"], boundary, 4)

    # Outlet
    rigid_bc_box = DirichletBC(V, Constant(0), boundary, 7)

    # For moving the mesh
    position = AssignedVectorFunction(wu_)

    bc_mesh = dict((ui, []) for ui in u_components)
    rigid_bc = [rigid_bc_in, rigid_bc_walls, rigid_bc_out, rigid_bc_box]
    bc_mesh["u0"] = [flag_bc_x] + rigid_bc
    bc_mesh["u1"] = [flag_bc_y] + rigid_bc

    mesh_prec = PETScPreconditioner("hypre_amg")  # In tests "sor" is faster. ilu
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

    return dict(viz_p=viz_p, viz_u=viz_u, viz_d=viz_d, viz_w=viz_w,
                u_vec=u_vec, mesh_sol=mesh_sol, bc_mesh=bc_mesh,
                dof_map=dof_map, a_mesh=a_mesh, L_mesh=L_mesh,
                coordinates=coordinates, position=position, alfa=alfa)


def update_prescribed_motion(t, dt, wx_, u_components, mesh_sol, bc_mesh, NS_expressions, A_cache, position, a_mesh,
                             L_mesh, mesh, **NS_namespace):
    # Update time
    for key, value in NS_expressions.items():
        if key != "uv":
            value.update(t)

    move = False
    for ui in u_components:
        # Solve for d and w
        A_mesh = A_cache[(a_mesh, tuple(bc_mesh[ui]))]

        [bc.apply(L_mesh[ui]) for bc in bc_mesh[ui]]

        t1 = OasisTimer("Mesh solver")
        mesh_sol.solve(A_mesh, wx_[ui], L_mesh[ui])
        t1.stop()

    position()
    position.vector()[:] *= dt

    # Move mesh
    arr = position.vector().get_local()
    if 1e-15 < abs(arr.min()) + abs(arr.max()):
        log_level = get_log_level()
        set_log_level(99)
        ALE.move(mesh, position)
        mesh.bounding_box_tree().build(mesh)
        set_log_level(log_level)
        move = True

    return move


def temporal_hook(t, w_, q_, viz_u, viz_p, u_vec, viz_w, **NS_namespace):
    assign(u_vec.sub(0), q_["u0"])
    assign(u_vec.sub(1), q_["u1"])

    viz_u.write(u_vec, t)
    viz_p.write(q_["p"], t)
    viz_w.write(w_["u0"], t)
    viz_w.write(w_["u1"], t)
