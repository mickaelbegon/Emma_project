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
    DefectType,
    OrderingStrategy,
)

from bioptim.limits.path_conditions import PathCondition
import time

#Define the stiffness and damping of the bar as global variables
stiffness = 16000
damping = 100

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


def save_sol_states_controls(sol, filename):
    parts_s = sol.decision_states(to_merge=[SolutionMerge.NODES])
    qs = np.hstack([p["q"] for p in parts_s])
    qdots = np.hstack([p["qdot"] for p in parts_s])
    taus = np.hstack([p["tau"] for p in parts_s])

    # taus = sol.decision_controls(to_merge=[SolutionMerge.NODES])[0]['tau']
    taudots = np.hstack([p["taudot"] for p in sol.decision_controls(to_merge=[SolutionMerge.NODES])])
    time = sol.stepwise_time(to_merge=[SolutionMerge.NODES, SolutionMerge.PHASES]).T[0]

    iter = sol.iterations
    conv = sol.real_time_to_optimize
    cost = sol.cost

    # # --- Save the solution --- #

    with open(filename + ".pkl", "wb") as file:
        data = {"q": qs, "qdot": qdots, "tau": taus, "taudot": taudots, "time": time,
                "convergence": conv,  "cost": cost, "iter": iter,
                } ,
        pickle.dump(data, file)
    print("states and controls saved:" + filename)

    with open(filename + "_sol.pkl", "wb") as file:
        del sol.ocp
        pickle.dump(sol, file)
    print("object solution of full solution saved" + filename)


def prepare_ocp(
        biorbd_model_path: str,
        final_time: tuple,
        n_shooting: tuple,
        min_time: float,
        max_time: float,
        total_mass: float,
        final_state_bound: bool,
        coef_fig : int,
        weight_tau: float,
        weight_time: float = 1,
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
    polynomial_degree = 5
    bio_model = (DynamicModel(biorbd_model_path), DynamicModel(biorbd_model_path), DynamicModel(biorbd_model_path))

    n_tau = bio_model[0].nb_tau
    n_q = bio_model[0].nb_q

    # Index of useful degrees of freedom
    names = ["TxHands", "TzHands", "RyHands",
             "Elbow", "Shoulder", "Back", #"Neck",
             "HipAbdR", "HipFlexR", "KneeR", "AnkleR",
             "HipAbdL", "HipFlexL", "KneeL", "AnkleL"]
    idx = {name: int(i) for i, name in enumerate(names) if name}
    idx_joints = np.arange(idx["RyHands"] + 1, idx["AnkleR"] + 1)  # index to constraint to 0 in the final state (all except those of the hands)

    #  symmetry of the lower limb
    pairs = [
        ("HipAbdR", "HipAbdL", -1),
        ("HipFlexR", "HipFlexL", 1),
        ("KneeR", "KneeL", 1),
        ("AnkleR", "AnkleL", 1),
    ]

    weights = {"Elbow": 5, "KneeR": 10, "AnkleR": 2}
    weight_abd = np.zeros(n_shooting[0])
    weight_abd[int(n_shooting[0]/2):] = 6

    # Add objective functions
    objective_functions = ObjectiveList()
    dynamics = DynamicsOptionsList()
    constraint_list = ConstraintList()

    for phase in range(3):
#        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="tau", quadratic=True, weight=weight_tau, phase=phase)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="tau", quadratic=True, weight=weight_tau, phase=phase)
        objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_CONTROL, key="taudot", quadratic=True, weight=weight_tau/1000, phase=phase)
        objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=weight_time, min_bound=min_time, max_bound=max_time,phase=phase)
        #objective_functions.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, key="qdot", weight=weight_tau/10, derivative=True, phase=phase)

        # FIG code specifications (knees, elbows and ankles flexion and thighs abduction)
        for name, w in weights.items():
            objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE,
                key="q", phase=phase, index=idx[name],target=0, weight=w*coef_fig, )

        dynamics.add(DynamicsOptions(
            expand_dynamics=expand_dynamics,
            phase_dynamics=phase_dynamics,
            ode_solver=OdeSolver.COLLOCATION(method="radau", polynomial_degree=polynomial_degree,
                                             defects_type=DefectType.TAU_EQUALS_INVERSE_DYNAMICS),
        ))

        for dof1, dof2, coef in pairs:
            constraint_list.add(ConstraintFcn.PROPORTIONAL_STATE,
                                key="q", phase=phase, node=Node.ALL,
                                first_dof=idx[dof1], second_dof=idx[dof2], coef=coef)


    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE,
                            key="q", index=idx["HipAbdR"], phase=0,
                            target=0,#node=Node.ALL,
                            weight=ObjectiveWeight(weight_abd, interpolation=InterpolationType.EACH_FRAME))
    objective_functions.add(ObjectiveFcn.Lagrange.TRACK_STATE, key="q", index=idx["HipAbdR"], target=0, weight=6*coef_fig, phase=2)

    # impose the orientation of the pelvic during the descent phase
    constraint_list.add(ConstraintFcn.TRACK_MARKERS,
                        list_index=22,
                        phase=0, node=Node.ALL,
                        reference_jcs=bio_model[0].segment_index("UPPER_TRUNK"),
                        marker_index=bio_model[0].marker_index("MarkerR"), axes=Axis.X,
                        min_bound= -2345, #0 if mode=="anteversion" else -2345,
                        max_bound= 3456,)# 0 if mode=="retroversion" else 3456, )

    # end of  phase[0] when the feet reach the height of the lower bar
    constraint_list.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                        node=Node.END, first_marker="LowerBarMarker",
                        second_marker="MarkerR", axes=Axis.Z, phase=0, min_bound=0.02, max_bound=0.02)

    constraint_list.add(ConstraintFcn.SUPERIMPOSE_MARKERS,
                        list_index=11,
                        phase=1, node=Node.ALL,
                        first_marker="LowerBarMarker", second_marker="MarkerR",
                        min_bound=-1234, # if init_sol else 0.02,
                        max_bound=np.inf, axes=Axis.X, )

    # BOUNDS
    rot_start =  -2 * np.pi / 45 # hands tilted by 8° at the start
    rot_end = rot_start - 2 * np.pi  # ends with hands 360° rotated

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
    else:
        constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", phase=2, node=Node.END, index=idx["RyHands"], min_bound= rot_end, max_bound= rot_end)
        constraint_list.add(ConstraintFcn.BOUND_STATE, key="q", phase=2, node=Node.END, index=idx_joints, min_bound=0, max_bound=0)
        constraint_list.add(ConstraintFcn.BOUND_STATE, key="qdot", phase=2, node=Node.END, index=idx["RyHands"], min_bound= -np.pi, max_bound= -np.pi)


    # Define control path bounds
    tau_min, tau_max = (-200, 200)
    u_min = [0] * 3 + [tau_min] * (n_tau-3)
    u_max = [0] * 3 + [tau_max] * (n_tau-3)

    u_min[idx["Shoulder"]] = -3.11 * total_mass
    u_max[idx["Shoulder"]] = 2.15 * total_mass
    u_min[idx["HipFlexR"]] = -4.20 * total_mass
    u_min[idx["HipFlexR"]] = -4.20 * total_mass
    u_max[idx["HipFlexR"]] = 9.36 * total_mass
    u_max[idx["HipFlexR"]] = 9.36 * total_mass

    for phase in range(3):
        x_bounds.add("tau", min_bound=u_min, max_bound=u_max, phase=phase)


    u_bounds = BoundsList()
    # for phase in range(3):
    #     u_bounds.add("tau", min_bound=u_min, max_bound=u_max, phase=phase)



    if x_init is None:
        rotations = [rot_start, -np.pi / 4, -np.pi, -2 * np.pi]
        x_init = InitialGuessList()
        for phase in range(3):
            init_q = np.zeros((n_q, 2))
            init_q[idx["RyHands"],0] = rotations[phase]
            init_q[idx["RyHands"],1] = rotations[phase] + (rotations[phase+1]-rotations[phase])*(polynomial_degree+1) #ordre DC+1
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
        ordering_strategy=OrderingStrategy.TIME_MAJOR,
    )


def main():

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(CURRENT_DIR, "applied_examples/results3")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    n_shooting = (25, 25, 55)
    weight_time = 10
    weight_fig = 1000
    use_sx = True

    time0 = time.perf_counter()

    for num in range(1,577):    #range(576):

        filename = f"/applied_examples/athlete_{num}_deleva.bioMod"
        print("model : ", filename)
        masses = pd.read_csv(CURRENT_DIR+ "/applied_examples/masses.csv")
        masse = masses["total_mass"][num-1]
        weight_tau = 1e-3 * masse

        # initial solution
        ocp = prepare_ocp(biorbd_model_path=CURRENT_DIR + filename, final_time=(1, 0.5, 1),
                          n_shooting=n_shooting, min_time=0.2, max_time=2, coef_fig=weight_fig, total_mass=masse,
                          weight_tau=weight_tau, weight_time=weight_time,
                          final_state_bound=True, n_threads=os.cpu_count()-1,   use_sx=use_sx)
        #todo compare final_state_bound=True vs False ... False should be faster

        ocp.add_plot_penalty(CostType.ALL)  # This will display the objectives and constraints at the current iteration
        ocp.print(to_console=False, to_graph=False)

        # --- Solver options --- #
        solver = Solver.IPOPT()#show_online_optim=True
        #solver = Solver.FATROP()
        solver.set_linear_solver("ma57")
        solver.set_maximum_iterations(5000)
        solver.set_bound_frac(1e-8)
        solver.set_bound_push(1e-8)

        print("solve initial solution")
        sol0 = ocp.solve(solver)
        save_sol_states_controls(sol0, os.path.join(RESULTS_DIR, f"athlete{num}_base"))
        print("initial solution saved")


        mode = "anteversion"
        # change bounds of two constraints:
        # list_index = 0: anteversion: min_bound=0, max_bound= np.inf, min_bound=-1234, #
        # list_index = 1: bar obstacle: 0.02,
        phase = 1
        list_index = 11
        print(ocp.nlp[phase].g[list_index].bounds.min) #should be -1234
        ocp.nlp[phase].g[list_index].bounds.min = PathCondition(0.02)

        phase = 0
        list_index = 22
        print(ocp.nlp[phase].g[list_index].bounds.min, ocp.nlp[phase].g[list_index].bounds.max) #should be -2345 3456
        ocp.nlp[phase].g[list_index].bounds.min = PathCondition(0)
        ocp.nlp[phase].g[list_index].bounds.max = PathCondition(np.inf)
        print("solve ante solution")
        sol1 = ocp.solve(solver, warm_start=sol0)
        save_sol_states_controls(sol1, os.path.join(RESULTS_DIR, f"athlete{num}_{mode}"))

        mode = "retroversion" # retroversion: min_bound=0, max_bound= np.inf,
        phase = 0
        list_index = 22
        print(ocp.nlp[phase].g[list_index].bounds.min, ocp.nlp[phase].g[list_index].bounds.max) #should be -2345 3456
        ocp.nlp[phase].g[list_index].bounds.min = PathCondition(-np.inf)
        ocp.nlp[phase].g[list_index].bounds.max = PathCondition(0)
        print("solve retro solution")
        sol2 = ocp.solve(solver, warm_start=sol0)
        save_sol_states_controls(sol2, os.path.join(RESULTS_DIR, f"athlete{num}_{mode}"))

        time1 = time.perf_counter()
        print(time1-time0)

if __name__ == "__main__":
    main()


