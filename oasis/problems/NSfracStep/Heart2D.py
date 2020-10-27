from pathlib import Path

import meshio
import numpy as np
import pygmsh

from ..NSfracStep import *

set_log_level(99)
parameters["allow_extrapolation"] = True


def get_H_and_D(N):
    size_factor = 1.0

    def gaussian(t, sigma=0.4, mu=1.068 / 2):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (t - mu) ** 2 / sigma ** 4)

    def ratio(t):
        # return 2 - 0.4 * np.sin(np.pi * t / 1.068)
        # return (2 / 2 - 0.1 * gaussian(t))
        return 2 - 0.1 * gaussian(t)

    def area(t):
        # return 18 + 20 * np.sin(np.pi * t / 1.068)
        # return 18 / 2 + 5 / 2 * gaussian(t)
        return 18 + 20 * gaussian(t)

    # 10 000 timesteps per cycle
    t = np.linspace(0, 1.068, N)
    ratios = ratio(t)
    areas = area(t)

    H = np.sqrt(4 * areas * ratios / np.pi)
    D = H / ratios
    H = H / H[0]
    D = D / D[0]

    return H, D


def problem_parameters(commandline_kwargs, NS_parameters, NS_expressions, **NS_namespace):
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
        T = 1.068  # Should simulate 10 heartbeats
        dt = 6.14 * 10 ** (-3)  # Recommended in article
        N = int(T / dt)
        T = 10 * T
        nu = 0.04
        NS_parameters.update(
            checkpoint=1000,
            save_step=10,
            check_flux=2,
            print_intermediate_info=1000,
            nu=nu,
            T=T,
            N=N,
            dt=dt,
            # Problem specific:
            D_0=3.4,  # in cm
            H_0=3.4 * 2,  # in cm
            velocity_degree=1,
            pressure_degree=1,
            folder="heart_results",
            lc=0.08,
            max_iter=2,
            mesh_path="mesh/ventricle_fine.xdmf",
            use_krylov_solvers=True)


def mesh(mesh_path, lc, D_0, H_0, **NS_namespace):
    # Inspired from
    # https://gist.ogithub.com/michalhabera/bbe8a17f788192e53fd758a67cbf3bed
    mesh_path = Path(mesh_path)
    mesh_path.parent.mkdir(exist_ok=True, parents=True)

    if not mesh_path.exists():
        geom = pygmsh.built_in.Geometry()

        # Create geometry
        p0 = geom.add_point([0, 0, 0], lcar=lc)
        p1 = geom.add_point([D_0 / 2, 0, 0], lcar=lc)
        p2 = geom.add_point([0, - H_0, 0], lcar=lc)
        p3 = geom.add_point([-D_0 / 2, 0, 0], lcar=lc)

        p4 = geom.add_point([1.615, 0, 0], lcar=lc)
        p5 = geom.add_point([-0.085, 0, 0], lcar=lc)
        p6 = geom.add_point([-0.272, 0, 0], lcar=lc)
        p7 = geom.add_point([-1.632, 0, 0], lcar=lc)

        l0 = geom.add_ellipse_arc(p1, p0, p2, p2)
        l1 = geom.add_ellipse_arc(p2, p0, p3, p3)
        # l2 = geom.add_line(p3, p1)
        l3 = geom.add_line(p3, p7)
        l4 = geom.add_line(p7, p6)
        l5 = geom.add_line(p6, p5)
        l6 = geom.add_line(p5, p4)
        l7 = geom.add_line(p4, p1)

        # ll = geom.add_line_loop(lines=[l0, l1, l2])
        ll = geom.add_line_loop(lines=[l7, l0, l1, l3, l4, l5, l6])
        ps = geom.add_plane_surface(ll)

        # Tag line and surface
        geom.add_physical_line(lines=l4, label=1)
        geom.add_physical_line(lines=l6, label=2)
        geom.add_physical_line(lines=[l7, l0, l1, l3, l5], label=3)
        geom.add_physical_surface(surfaces=ps, label=4)

        # Mesh surface
        msh = pygmsh.generate_mesh(geom)

        # Write, then read mesh and MeshFunction
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


class Wall(UserExpression):
    def __init__(self, tstep, comp, DH, DH_0, dt, N, **kwargs):
        self.tstep = tstep
        self.comp = comp
        self.DH = DH
        self.DH_0 = DH_0
        self.dt = dt
        self.N = N
        self.ds = ds
        super().__init__(**kwargs)

    def eval(self, values, x):
        tstep = self.tstep % self.N
        values[:] = (self.DH[tstep] - self.DH[tstep - 1]) / (self.dt * self.DH_0) * x[self.comp]


class Inlet(UserExpression):
    def __init__(self, tstep, D, H, dt, ds, N, newfolder, **kwargs):
        self.tstep = tstep
        self.D = D
        self.H = H
        self.dt = dt
        self.ds = ds
        self.N = N

        self.newfolder = newfolder

        self.u = 0
        self.U = 0

        super().__init__(**kwargs)

    def update(self, tstep):
        self.tstep = tstep
        tstep = self.tstep % self.N

        Q = (np.pi / 4 * self.D[tstep] * self.H[tstep]
             * ((self.D[tstep] - self.D[tstep - 1]) / self.dt / self.D[tstep]
                + (self.H[tstep] - self.H[tstep - 1]) / self.dt / self.H[tstep]))

        length = assemble(Constant(1) * self.ds(2))
        self.U = Q / length

        # Write forces to file
        flux_properties = np.asarray([tstep, Q, self.u])
        flux_path = path.join(self.newfolder, "VTK", "flux.txt")
        with open(flux_path, 'a') as filename:
            filename.write("{:.4f} {:.3f} {:.3f} \n".format(*flux_properties))
        # if Q < 0:
        #    self.u = -self.u
        # else:
        #    self.u = 0
        self.U = self.logistic(self.dt * self.tstep)
        # self.U = self.gaussian(self.dt * tstep)
        print("t={:.6f} u={}".format(self.dt * self.tstep, self.U))

    def gaussian(self, t, sigma=0.4, mu=1.068 / 2):
        return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (t - mu) ** 2 / sigma ** 4)

    def logistic(self, t, k=5, t0=1.068, L=1):
        return L / (1 + np.exp(-k * (t - t0)))

    def parabolic(self, x, x0=-0.085, x1=1.615):
        return 4 * (x[0] - x1) * (x[0] - x0) / ((x0 - x1) ** 2)

    def eval(self, values, x):
        values[:] = self.parabolic(x) * self.U


class Outlet(UserExpression):
    def __init__(self, tstep, D, H, dt, ds, N, newfolder, **kwargs):
        self.tstep = tstep % N
        self.D = D
        self.H = H
        self.dt = dt
        self.ds = ds
        self.N = N
        self.newfolder = newfolder

        self.u = 0

        super().__init__(**kwargs)

    def update(self, tstep):
        tstep = tstep % self.N
        Q = (np.pi / 4 * self.D[tstep] * self.H[tstep]
             * ((self.D[tstep] - self.D[tstep - 1]) / self.dt / self.D[tstep]
                + (self.H[tstep] - self.H[tstep - 1]) / self.dt / self.H[tstep]))

        if Q < 0:
            length = assemble(Constant(1) * self.ds(2))
            self.u = Q / length

            # Write forces to file
            flux_properties = np.asarray([tstep, Q, self.u])
            flux_path = path.join(self.newfolder, "VTK", "flux.txt")
            with open(flux_path, 'a') as filename:
                filename.write("{:.4f} {:.3f} {:.3f} \n".format(*flux_properties))
        else:
            self.u = 0

    def eval(self, values, x):
        values[:] = self.u


class ResistenceCondition(UserExpression):
    def __init__(self, tstep, D, H, dt, ds, N, n, newfolder, **kwargs):
        self.tstep = tstep % N
        self.D = D
        self.H = H
        self.dt = dt
        self.ds = ds
        self.N = N
        self.newfolder = newfolder
        self.C_out = 0  # 5.97 * 10 ** (-3)  # Resistance constant
        self.P_V = 11332.0 * 10 ** (-6)  # Bloodpressure , 85 mmHg in Pa
        self.r_AO = 1.36 / 2.  # Radius of aortic valve
        self.n = n

        self.p = 0

        super().__init__(**kwargs)

    def update(self, tstep, u_):
        tstep = tstep % self.N
        Q = (np.pi / 4 * self.D[tstep] * self.H[tstep]
             * ((self.D[tstep] - self.D[tstep - 1]) / self.dt / self.D[tstep]
                + (self.H[tstep] - self.H[tstep - 1]) / self.dt / self.H[tstep]))

        if Q < 0:
            Q_out = np.pi / 2 * self.r_AO * assemble(dot(u_, self.n) * self.ds(3))
            self.p = -(self.C_out * Q_out + self.P_V)
        else:
            self.p = 0

    def eval(self, values, x):
        values[:] = self.p


def pre_boundary_condition(mesh, **NS_namespace):
    inlet = AutoSubDomain(lambda x, b: b and -0.085 - DOLFIN_EPS <= x[0] <= 1.615 +
                                       DOLFIN_EPS and x[1] > 0 - DOLFIN_EPS)
    outlet = AutoSubDomain(lambda x, b: b and -1.632 - DOLFIN_EPS <= x[0] <= -0.272 +
                                        DOLFIN_EPS and x[1] > 0 - DOLFIN_EPS)
    walls = AutoSubDomain(lambda x, b: b and not ((-1.632 + DOLFIN_EPS <= x[0] <= -0.272 -
                                                   DOLFIN_EPS and x[1] > 0 - DOLFIN_EPS)
                                                  or (-0.085 + DOLFIN_EPS <= x[0] <=
                                                      -1.615 - DOLFIN_EPS
                                                      and x[1] < 0 - DOLFIN_EPS)))

    boundary = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary.set_all(0)

    walls.mark(boundary, 1)
    inlet.mark(boundary, 2)
    outlet.mark(boundary, 3)

    return dict(boundary=boundary)


def create_bcs(V, Q, D_0, H_0, sys_comp, mesh, NS_expressions, tstep, dt, N, boundary, newfolder, **NS_namespace):
    info_red("Creating boundary conditions")

    H, D = get_H_and_D(N)

    n = FacetNormal(mesh)
    ds = Measure('ds', domain=mesh, subdomain_data=boundary)

    # Resistance boundary condition at aortic valve
    wall0 = Wall(tstep, 0, D, D_0, dt, N, element=V.ufl_element())
    wall1 = Wall(tstep, 1, H, H_0, dt, N, element=V.ufl_element())
    #wall0 = wall1 = Constant(0.0)
    inlet = Inlet(tstep, D, H, dt, ds, N, newfolder, element=V.ufl_element())
    outlet = Outlet(tstep, D, H, dt, ds, N, newfolder, element=V.ufl_element())
    p_outlet = ResistenceCondition(tstep, D, H, dt, ds, N, n, newfolder, element=V.ufl_element())

    NS_expressions["wall0"] = wall0
    NS_expressions["wall1"] = wall1
    NS_expressions["inlet"] = inlet
    NS_expressions["outlet"] = outlet
    NS_expressions["resistance"] = p_outlet

    wall0 = wall1 = Constant(0.0)
    bcu_wall_x = DirichletBC(V, wall0, boundary, 1)
    bcu_wall_y = DirichletBC(V, wall1, boundary, 1)

    bcu_in_x = DirichletBC(V, Constant(0), boundary, 2)
    bcu_in_y = DirichletBC(V, NS_expressions["inlet"], boundary, 2)

    bcu_out_x = DirichletBC(V, Constant(0), boundary, 3)
    bcu_out_y = DirichletBC(V, NS_expressions["outlet"], boundary, 3)

    # bcp_out = DirichletBC(Q, NS_expressions["resistance"], boundary, 3)
    bcp_out = DirichletBC(Q, Constant(0), boundary, 3)
    bcp_in = DirichletBC(Q, Constant(0), boundary, 2)

    bcs = dict((ui, []) for ui in sys_comp)
    bcs['u0'] = [bcu_in_x, bcu_wall_x]
    bcs['u1'] = [bcu_in_y, bcu_wall_y]
    bcs["p"] = [bcp_out]

    return bcs


def pre_solve_hook(V, u_components, mesh, newfolder, boundary, velocity_degree, **NS_namespace):
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
    D_ = project(A, DG)
    D_arr = D_.vector().get_local()

    # Note: Set alfa to something 1/distace, computed by laplace.
    # alfa = Constant(D_arr.max() ** 3 - D_arr.min() ** 3) / D_ ** 3
    alfa = Constant((D_arr.max() - D_arr.min()) ** 4) / D_ ** 4
    a_mesh = inner(alfa * grad(u_mesh), grad(v_mesh)) * dx
    L_mesh = dict((ui, assemble(Function(V) * v_mesh * dx)) for ui in u_components)

    # Walls, inlet, outlet
    walls_x = DirichletBC(V, NS_expressions["wall0"], boundary, 1)
    walls_y = DirichletBC(V, NS_expressions["wall1"], boundary, 1)
    inlet_bc = DirichletBC(V, Constant(0), boundary, 2)
    outlet_bc = DirichletBC(V, Constant(0), boundary, 3)

    bc_mesh = dict((ui, []) for ui in u_components)
    # rigid_bc = [inlet_bc]
    rigid_bc = [inlet_bc, outlet_bc]
    bc_mesh["u0"] = [walls_x] + rigid_bc
    bc_mesh["u1"] = [walls_y] + rigid_bc

    mesh_prec = PETScPreconditioner("hypre_amg")  # in tests sor are faster ..
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
    # dof_map = V.dofmap().entity_dofs(mesh, 0)

    return dict(viz_p=viz_p, viz_u=viz_u, viz_w=viz_w, u_vec=u_vec, mesh_sol=mesh_sol, bc_mesh=bc_mesh, dof_map=dof_map,
                a_mesh=a_mesh, L_mesh=L_mesh, coordinates=coordinates)


def update_prescribed_motion(dt, w_, u_, u_components, tstep, mesh_sol, bc_mesh, NS_expressions, a_mesh, L_mesh, wx_,
                             coordinates, dof_map, A_cache, **NS_namespace):
    for key, value in NS_expressions.items():

        if "let" in key:
            value.update(tstep)
        elif "resistance" in key:
            value.update(tstep, u_)
        else:
            value.tstep = tstep

    move = False
    testing = True

    for ui in u_components:  # and not testing:
        # Solve for d and w
        A_mesh = A_cache[(a_mesh, tuple(bc_mesh[ui]))]  # Assemble mesh form, and add bcs
        [bc.apply(L_mesh[ui]) for bc in bc_mesh[ui]]  # Apply bcs
        mesh_sol.solve(A_mesh, wx_[ui], L_mesh[ui])  # Solve for mesh

        # Move mesh
        arr = w_[ui].vector().get_local()
        mesh_tolerance = 1e-15
        if mesh_tolerance < abs(arr.min()) + abs(arr.max()):
            coordinates[:, int(ui[-1])] += (arr * dt)[dof_map]
            move = True

    return move


def temporal_hook(t, w_, p_, u_, tstep, viz_u, viz_p, u_vec, viz_w, **NS_namespace):
    assign(u_vec.sub(0), u_[0])
    assign(u_vec.sub(1), u_[1])

    viz_p.write(p_, t)
    viz_u.write(u_vec, t)
