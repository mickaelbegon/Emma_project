import numpy as np
import pandas as pd
from casadi import MX, vertcat
import os
import pickle

import matplotlib
matplotlib.use("Qt5Agg")

from bioptim import (
    Node,
    ConstraintList,
    InterpolationType,
    ConstraintFcn,
    ObjectiveList,
    DynamicsFunctions,
    OptimalControlProgram,
    BoundsList,
    InitialGuessList,
    ObjectiveFcn,
    OdeSolver,
    OdeSolverBase,
    CostType,
    Solver,
    NonLinearProgram,
    ControlType,
    PhaseDynamics,
    DynamicsEvaluation,
    SolutionMerge,
    Axis,
    DynamicsOptionsList,
    DynamicsOptions,
    TorqueBiorbdModel,
    TorqueDerivativeBiorbdModel,
    ObjectiveWeight,
    DefectType
)

#Define the stiffness and damping of the bar as global variables
stiffness = 14160
damping = 91

class DynamicModel(TorqueDerivativeBiorbdModel):
    def __init__(self, biorbd_model_path):
        super().__init__(biorbd_model_path)

    def dynamics(
            self,
            time: MX,
            states: MX,
            controls: MX,
            parameters: MX,
            algebraic_states: MX,
            numerical_timeseries: MX,
            nlp: NonLinearProgram,
    ) -> DynamicsEvaluation:
        """
        Modify the system dynamics to model the bar as a spring

        Parameters
        ----------
        time: MX
            The current time of the system
        states: MX
            The current states of the system
        controls: MX
            The current controls of the system
        parameters: MX
            The current parameters of the system
        algebraic_states: MX
            The current algebraic states of the system
        numerical_timeseries: MX
            The current numerical timeseries of the system
        nlp: NonLinearProgram
            A reference to the phase of the ocp

        Returns
        -------
        The state derivative
        """

        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tau = DynamicsFunctions.get(nlp.states["tau"], states)
        taudot = DynamicsFunctions.get(nlp.controls["taudot"], controls)

        tau[0] = -stiffness * q[0] + damping * qdot[0]  # x
        tau[1] = -stiffness * q[1] + damping * qdot[1]  # z

        # todo: add passive joint torques?
        qddot = nlp.model.forward_dynamics(with_contact=False)(q, qdot, tau, [], [])

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            slope_q = DynamicsFunctions.get(nlp.states_dot["q"], nlp.states_dot.scaled.cx)
            slope_qdot = DynamicsFunctions.get(nlp.states_dot["qdot"], nlp.states_dot.scaled.cx)
            slope_tau = DynamicsFunctions.get(nlp.states_dot["tau"], nlp.states_dot.scaled.cx)

            if nlp.dynamics_type.ode_solver.defects_type == DefectType.QDDOT_EQUALS_FORWARD_DYNAMICS:
                defects = vertcat(slope_q, slope_qdot, slope_tau)  - vertcat(qdot, qddot, taudot)


            elif nlp.dynamics_type.ode_solver.defects_type == DefectType.TAU_EQUALS_INVERSE_DYNAMICS:
                tau_id = nlp.model.inverse_dynamics(with_contact=False)(q, qdot, slope_qdot, [], [])
                defects = vertcat(slope_q, tau_id, slope_tau) - vertcat(qdot, tau, taudot)


        return DynamicsEvaluation(dxdt=vertcat(qdot, qddot), defects=defects)


def prepare_ocp(
        biorbd_model_path: str,
        final_time: tuple,
        n_shooting: tuple,
        min_time: float,
        max_time: float,
        total_mass: float,
        init_sol: bool,
        final_state_bound: bool,
        coef_fig : int,
        weight_tau: float,
        weight_time: float = 1,
        mode: str="",         ode_solver: OdeSolverBase = OdeSolver.RK4(),
        use_sx: bool = False,
        n_threads: int = 1,
        phase_dynamics: PhaseDynamics = PhaseDynamics.SHARED_DURING_THE_PHASE,
        expand_dynamics: bool = True,
        control_type: ControlType = ControlType.CONSTANT,
        x_init: InitialGuessList = None,
        u_init: InitialGuessList = None,
) -> OptimalControlProgram:
    """
    The initialization of an ocp

    Parameters
    ----------
    biorbd_model_path: str
        The path to the biorbd model
    final_time: tuple
        The initial guess for the final time of each phase
    n_shooting: tuple
        The number of shooting points (one number per phase) to define in the direct multiple shooting program
     min_time: float
        The minimum time allowed for the final node
    max_time: float
        The maximum time allowed for the final node
    total_mass : float
        The mass of the athlete
    init_sol : bool
        If it computes an initial solution
    final_state_bound : bool
        If the final state is with bound (false means it's with constraints)
    coef_fig : int
        Weighting coefficient for objectives that implement FIG code specifications
    weight_tau: float
        Weight for the torque minimization objective
    weight_time: float
        Weight for the time minimization objective
    mode : str
        Specifies the orientation of the pelvic during the descent phase
    ode_solver: OdeSolverBase = OdeSolver.RK4()
        Which type of OdeSolver to use
    use_sx: bool
        If the SX variable should be used instead of MX (can be extensive on RAM)
    n_threads: int
        The number of threads to use in the paralleling (1 = no parallel computing)
    phase_dynamics: PhaseDynamics
        If the dynamics equation within a phase is unique or changes at each node.
        PhaseDynamics.SHARED_DURING_THE_PHASE is much faster, but lacks the capability to have changing dynamics within
        a phase. A good example of when PhaseDynamics.ONE_PER_NODE should be used is when different external forces
        are applied at each node
    expand_dynamics: bool
        If the dynamics function should be expanded. Please note, this will solve the problem faster, but will slow down
        the declaration of the OCP, so it is a trade-off. Also depending on the solver, it may or may not work
        (for instance IRK is not compatible with expanded dynamics)
    control_type: ControlType
        The type of the controls
    x_init : InitialGuessList
        Initial values of the states
    u_init : InitialGuessList
        Initial values of the controls

    Returns
    -------
    The OptimalControlProgram ready to be solved
    """
    bio_model = (DynamicModel(biorbd_model_path), DynamicModel(biorbd_model_path), DynamicModel(biorbd_model_path))

    n_tau = bio_model[0].nb_tau
    n_q = bio_model[0].nb_q

    # Index of useful degrees of freedom
    names = ["TxHands", "TzHands", "RyHands",
             "Elbow", "Shoulder", "Back", "LowBack",
             "RxThighR", "RyThighR", "KneeR", "FootR",
             "RxThighL", "RyThighL", "KneeL", "FootL"]
    idx = {name: int(i) for i, name in enumerate(names) if name}
    idx_joints = np.arange(idx["RyHands"] + 1, idx["FootL"] + 1)  # index to constraint to 0 in the final state (all except those of the hands)

    weights = {"Elbow": 5, "KneeR": 10, "FootR": 2}

    # Add objective functions
    objective_functions = ObjectiveList()
    for phase in range(3):
#        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", quadratic=True, weight=weight_tau, phase=phase)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau", quadratic=True, weight=weight_tau, phase=phase)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", quadratic=True, weight=0.1, phase=phase)
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=weight_time, min_bound=min_time, max_bound=max_time,phase=phase)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=1, derivative=True, phase=phase)

        # FIG code specifications (knees, elbows and ankles flexion and thighs abduction)
        for name, w in weights.items():
            objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE,
                key="q", index=idx[name],target=0,weight=w * coef_fig,phase=phase)


        leg_weight = np.zeros(n_shooting[0]+1)
        leg_weight[:int(n_shooting[0]/2)] = 3

        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_STATE, key="q", index=idx["RxThighR"], phase=0,  node=Node.ALL,
                                weight=ObjectiveWeight(leg_weight, interpolation=InterpolationType.EACH_FRAME))
        objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx["RxThighR"], target=0, weight=3*coef_fig, phase=2)




    # Dynamics
    dynamics = DynamicsOptionsList()
    for phase in range(3):
        dynamics.add(DynamicsOptions(
            expand_dynamics=expand_dynamics,
            phase_dynamics=phase_dynamics,
            ode_solver=OdeSolver.COLLOCATION(method="radau", polynomial_degree=5,
                                             defects_type=DefectType.TAU_EQUALS_INVERSE_DYNAMICS),
        ))


    # Constraints
    constraint_list = ConstraintList()
    #if init_sol is False:
        # avoid the lower bar when min_bound=0.02
    constraint_list.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.ALL, first_marker="LowerBarMarker",
                            second_marker="MarkerR", min_bound=-1234, max_bound=np.inf, axes=Axis.X, phase=1)



    # impose the orientation of the pelvic during the descent phase
    # anteversion: min_bound=0, max_bound= np.inf,
    # retroversion: min_bound=0, max_bound= np.inf,
        #if mode == "anteversion":
    constraint_list.add(ConstraintFcn.TRACK_MARKERS, phase=0, node=Node.ALL, reference_jcs=idx["Back"],
                        marker_index=3, axes=Axis.X, min_bound=-2345, max_bound=3456, )
        #elif mode == "retroversion":
        #    constraint_list.add(ConstraintFcn.TRACK_MARKERS, phase=0, node=Node.ALL, reference_jcs=idx["Back"],
        #                        marker_index=3, axes=Axis.X, min_bound=0, max_bound= np.inf,)

    # end of  phase[0] when the feet reach the height of the lower bar
    constraint_list.add(ConstraintFcn.SUPERIMPOSE_MARKERS, node=Node.END, first_marker="LowerBarMarker",
                        second_marker="MarkerR", axes=Axis.Z, phase=0, min_bound=0.02, max_bound=0.02)

    #  symmetry of the thighs
    pairs = [
        ("RxThighR", "RxThighL", -1),
        ("RyThighR", "RyThighL", 1),
        ("KneeR", "KneeL", 1),
        ("FootR", "FootL", 1),
    ]
    for phase in range(3):
        for a, b, c in pairs:
            constraint_list.add(ConstraintFcn.PROPORTIONAL_STATE,
                key="q", phase=phase, node=Node.ALL,
                first_dof=idx[a], second_dof=idx[b],coef=c )


    # BOUNDS
    rot_start =  -2 * np.pi / 45 # hands tilted by 8° at the start
    rot_end = -2 * np.pi   # ends with hands 360° rotated

    x_bounds = BoundsList()
    for phase in range(3):
        x_bounds.add("q", bounds=bio_model[0].bounds_from_ranges("q"), phase=phase)
        x_bounds.add("qdot", bounds=bio_model[0].bounds_from_ranges("qdot"), phase=phase)

    x_bounds[0]["q"][:, 0] = 0
    x_bounds[0]["q"][idx["RyHands"], 0] = rot_start
    x_bounds[0]["q"][1, 0] = - total_mass*9.81 / stiffness # equilibrium position
    x_bounds[0]["qdot"][:, 0] = 0  # speeds start at 0
    x_bounds[1]["q"][idx["RyHands"], -1] = -np.pi # end of second phase with hands under the upper bar


    if final_state_bound:
        x_bounds[2]["q"][idx_joints, -1] = 0
        x_bounds[2]["q"][idx["RyHands"], -1] = rot_end
        x_bounds[2]["qdot"][idx["RyHands"], -1] = - np.pi  # ends with hands speed of pi rad/s
    else :
        constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", phase=2, node=Node.END, index=idx["RyHands"], min_bound= rot_end, max_bound= rot_end)
        constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", phase=2, node=Node.END, index=idx_joints, min_bound=0, max_bound=0)
        constraint_list.add(ConstraintFcn.BOUND_STATE, key="qdot", phase=2, node=Node.END, index=idx["RyHands"], min_bound= -np.pi, max_bound= -np.pi)


    # Define control path bounds
    tau_min, tau_max = (-200, 200)
    u_min = [0] * 3 + [tau_min] * (n_tau-3)
    u_max = [0] * 3 + [tau_max] * (n_tau-3)

    if init_sol is False:
        u_min[idx["Shoulder"]] = -3.11 * total_mass
        u_max[idx["Shoulder"]] = 2.15 * total_mass
        u_min[idx["RyThighR"]] = -4.20 * total_mass
        u_max[idx["RyThighR"]] = 9.36 * total_mass
        u_min[idx["RyThighL"]] = -4.20 * total_mass
        u_max[idx["RyThighL"]] = 9.36 * total_mass

    u_bounds = BoundsList()
    # for phase in range(3):
    #     u_bounds.add("tau", min_bound=u_min, max_bound=u_max, phase=phase)


    for phase in range(3):
        x_bounds.add("tau", min_bound=u_min, max_bound=u_max, phase=phase)

    if x_init is None:
        rotations = [rot_start, -np.pi / 4, -np.pi, -2 * np.pi]
        x_init = InitialGuessList()
        for phase in range(3):
            init_q = np.zeros((n_q, 2))
            init_q[idx["RyHands"],0] = rotations[phase]
            init_q[idx["RyHands"],1] = rotations[phase] + (rotations[phase+1]-rotations[phase])*6 #ordre DC+1
            init_qdot = np.zeros((n_q,1))
            init_qdot[idx["RyHands"],:] = -np.pi

            x_init.add("q", init_q, phase=phase, interpolation=InterpolationType.LINEAR)
            x_init.add("qdot", init_qdot, phase=phase)
            x_init.add("tau", [0] * n_tau, phase=phase)

    if u_init is None:
        u_init = InitialGuessList()
        for phase in range(3):
        #     u_init.add("tau", [0] * n_tau, phase=phase)
            u_init.add("taudot", [0] * n_tau, phase=phase)

    return OptimalControlProgram(
        bio_model,
        n_shooting,
        final_time,
        dynamics=dynamics,
        x_init=x_init,
        u_init=u_init,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
        use_sx=use_sx,
        n_threads=n_threads,
        control_type=control_type,
        constraints=constraint_list,
    )


def main():

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(CURRENT_DIR, "applied_examples/results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    use_pkl = False

    n_shooting = (25, 25, 50)

    for num in range(100,102):    #range(576):

        filename = f"/applied_examples/athlete_{num}_deleva.bioMod"
        print("model : ", filename)
        masses = pd.read_csv(CURRENT_DIR+ "/applied_examples/masses.csv")
        masse = masses["total_mass"][num-1]

        # initial solution
        if use_pkl is False or not os.path.exists(os.path.join(RESULTS_DIR, f"athlete{num}_base.pkl")):

            ocp = prepare_ocp(biorbd_model_path=CURRENT_DIR + filename, final_time=(1, 0.5, 1),
                              n_shooting=n_shooting, min_time=0.2, max_time=2, coef_fig=1, total_mass=masse,
                              init_sol=True, weight_tau=1, weight_time=0.1,final_state_bound=True, n_threads=os.cpu_count()-2,
                              use_sx=True)
            #todo compare final_state_bound=True vs False ... False should be faster
            # --- Live plots --- #
            ocp.add_plot_penalty(CostType.ALL)  # This will display the objectives and constraints at the current iteration

            # --- Print ocp structure --- #
            ocp.print(to_console=False, to_graph=False)

            # --- Solve the ocp --- #
            solver = Solver.IPOPT()#show_online_optim=True
            solver.set_linear_solver("ma57")
            solver.set_maximum_iterations(1000)
            solver.set_bound_frac(1e-8)
            solver.set_bound_push(1e-8)
            solver.set_acceptable_tol(1e-3)

            print("start solving")
            sol = ocp.solve(solver)
            print("solving finished")

            parts_s = sol.decision_states(to_merge=[SolutionMerge.NODES])  # list per phase
            parts_u = sol.decision_controls(to_merge=[SolutionMerge.NODES])  # list per phase
            x_init = InitialGuessList()
            u_init = InitialGuessList()
            for p, (ps, pu) in enumerate(zip(parts_s, parts_u)):
                x_init.add("q", ps["q"], InterpolationType.ALL_POINTS, phase=p)
                x_init.add("qdot", ps["qdot"], InterpolationType.ALL_POINTS, phase=p)
                x_init.add("tau", ps["tau"], InterpolationType.ALL_POINTS, phase=p)
                u_init.add("taudot", pu["taudot"], InterpolationType.EACH_FRAME, phase=p)

            ocp.update_initial_guess(x_init=x_init, u_init=u_init, )
            #todo: correct ValueError: show_online_optim and online_optim cannot be simultaneous set
            solver.set_maximum_iterations(2000)
            solver.set_acceptable_tol(1e-3)
            #solver.show_online_optim = None
            #solver.online_optim = 0
            sol2 = ocp.solve(solver)


            parts_s = sol.decision_states(to_merge=[SolutionMerge.NODES])
            qs = np.hstack([p["q"] for p in parts_s])
            qdots = np.hstack([p["qdot"] for p in parts_s])
            taus = np.hstack([p["tau"] for p in parts_s])

            #taus = sol.decision_controls(to_merge=[SolutionMerge.NODES])[0]['tau']
            taudots = np.hstack([p["taudot"] for p in sol.decision_controls(to_merge=[SolutionMerge.NODES])])
            time = sol.stepwise_time(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES]).T[0]

            # # --- Save the solution --- #
            with open(os.path.join(RESULTS_DIR, f"athlete{num}_base.pkl"), "wb") as file:
                data = {"q": qs, "qdot": qdots, "tau": taus, "taudot": taudots, "time": time, }
                pickle.dump(data, file)

            print("initial solution saved")


        ######################################################################################################################################################
        print("load initial solution")
        with open(os.path.join(RESULTS_DIR, f"athlete{num}_base.pkl"), "rb") as file:
            prev_sol_data = pickle.load(file)

        qs = prev_sol_data["q"]
        qdots = prev_sol_data["qdot"]
        taus = prev_sol_data["tau"]
        taudots = prev_sol_data["taudot"]

        # Create the initial solution (warm start)
        ns = np.array(n_shooting)
        S_ctrl = np.concatenate(([0], np.cumsum(ns)))  # tailles ni
        S_state = np.concatenate(([0], np.cumsum(ns + 1)))  # tailles ni+1

        x_init = InitialGuessList()
        u_init = InitialGuessList()

        for p, (sa, sb, ua, ub) in enumerate(zip(S_state[:-1], S_state[1:], S_ctrl[:-1], S_ctrl[1:])):
            x_init.add("q", qs[:, sa:sb], InterpolationType.EACH_FRAME, phase=p)
            x_init.add("qdot", qdots[:, sa:sb], InterpolationType.EACH_FRAME, phase=p)
            x_init.add("tau", taus[:, sa:sb], InterpolationType.EACH_FRAME, phase=p)

            u_init.add("taudot", taudots[:, ua:ub], InterpolationType.EACH_FRAME, phase=p)


        for mode in ['anteversion', 'retroversion']:

            if use_pkl is False or not os.path.exists(os.path.join(RESULTS_DIR, f"athlete{num}_complet_{mode}.pkl")):

                # solution complete
                ocp = prepare_ocp(biorbd_model_path=CURRENT_DIR + filename, final_time=(1, 0.5, 1), n_shooting=n_shooting,
                                  min_time=0.01, max_time=2, total_mass=masse, init_sol=False, weight_tau=0.0001, weight_time=1,
                                  coef_fig=1,final_state_bound=True, mode=mode, n_threads=os.cpu_count()-2, use_sx=True)

                # --- Live plots --- #
                ocp.add_plot_penalty(CostType.ALL)  # This will display the objectives and constraints at the current iteration

                # --- Print ocp structure --- #
                ocp.print(to_console=False, to_graph=False)

                # --- Solve the ocp --- #
                solver = Solver.IPOPT()
                solver.set_linear_solver("ma57")
                solver.set_maximum_iterations(20000)
                sol = ocp.solve(solver)

                qs = sol.decision_states(to_merge=[SolutionMerge.NODES])[0]['q']
                qdots = sol.decision_states(to_merge=[SolutionMerge.NODES])[0]['qdot']
                taus = sol.decision_states(to_merge=[SolutionMerge.NODES])[0]['tau']

                for i in range(1, len(sol.decision_states(to_merge=[SolutionMerge.NODES]))):
                    qs = np.hstack((qs, sol.decision_states(to_merge=[SolutionMerge.NODES])[i]['q']))
                    qdots = np.hstack((qdots, sol.decision_states(to_merge=[SolutionMerge.NODES])[i]['qdot']))
                    taus = np.hstack((taus, sol.decision_states(to_merge=[SolutionMerge.NODES])[i]['taus']))

                taudots = sol.decision_controls(to_merge=[SolutionMerge.NODES])[0]['taudot']
                for i in range(1, len(sol.decision_controls(to_merge=[SolutionMerge.NODES]))):
                    taudots = np.hstack((taudots, sol.decision_controls(to_merge=[SolutionMerge.NODES])[i]['taudot']))

                time = sol.stepwise_time(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES]).T[0]

                # # --- Save the solution --- #
                with open(os.path.join(RESULTS_DIR, f"athlete{num}_complet_{mode}.pkl"), "wb") as file:
                    data = {"q": qs, "qdot": qdots, "tau": taus, "taudot": taudots, "time": time, }
                    pickle.dump(data, file)

                    print("data of full solution saved")


                # --- Show the results graph --- #
                sol.print_cost()
                #sol.graphs(show_bounds=True, show_now=True)

                with open(os.path.join(RESULTS_DIR, f"Sol_athlete{num}_complet_{mode}.pkl"), "wb") as file:
                    del sol.ocp
                    pickle.dump(sol, file)
                print("object solution of full solution saved")

            # --- Animate the solution --- #
            if num == 502:
                viewer = "pyorerun"
                sol.animate(n_frames=0, viewer=viewer, show_now=True)





if __name__ == "__main__":
    main()


