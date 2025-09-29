from __future__ import annotations
import os
import pickle
from pathlib import Path
import matplotlib
matplotlib.use("Qt5Agg")  # GUI backend (adapt to QtAgg if PyQt6)
import numpy as np
import pandas as pd

from bioptim import CostType, Solver
from .ocp import prepare_ocp
from .helpers import stack_states, stack_controls, step_time, save_arrays, build_initial_guesses

def main():
    ROOT = Path(__file__).resolve().parent.parent  # project root (adjust if needed)
    RESULTS_DIR = ROOT / "applied_examples" / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    use_pkl = True
    n_shooting = (50, 50, 50)

    for num in [102]:  # or range(576)
        model_path = (ROOT / f"applied_examples/athlete_{num}_deleva.bioMod").as_posix()
        print("model:", model_path)

        masses = pd.read_csv(ROOT / "applied_examples" / "masses.csv")
        total_mass = float(masses["total_mass"][num - 1])

        base_pkl = RESULTS_DIR / f"athlete{num}_base.pkl"

        # ------------- Initial solution -------------
        if (not use_pkl) or (not base_pkl.exists()):
            ocp = prepare_ocp(
                biorbd_model_path=model_path,
                final_time=(1, 0.5, 1),
                n_shooting=n_shooting,
                min_time=0.2,
                max_time=2.0,
                coef_fig=1.0,
                total_mass=total_mass,
                init_sol=True,
                weight_control=1.0,
                weight_time=0.1,
                final_state_bound=True,
                n_threads=max(1, (os.cpu_count() or 4) - 2),
                use_sx=False,
            )

            ocp.add_plot_penalty(CostType.ALL)
            ocp.print(to_console=False, to_graph=False)

            solver = Solver.IPOPT(show_online_optim=False)
            solver.set_linear_solver("ma57")
            solver.set_maximum_iterations(5000)

            print("start solving (initial)")
            sol = ocp.solve(solver)
            print("solving finished (initial)")

            q, qd = stack_states(sol)
            tau = stack_controls(sol)
            t = step_time(sol)
            save_arrays(base_pkl, q, qd, tau, t)
            print("initial solution saved")

        # ------------- Load initial solution -------------
        print("load initial solution")
        with open(base_pkl, "rb") as f:
            prev = pickle.load(f)
        qs, qdots, taus = prev["q"], prev["qdot"], prev["tau"]

        # Warm start (ready if you want to pass x_init/u_init to the next ocp)
        _x_init, _u_init = build_initial_guesses(qs, qdots, taus, n_shooting)

        # ------------- Full solutions -------------
        for mode in ["anteversion", "retroversion"]:
            full_pkl = RESULTS_DIR / f"athlete{num}_complet_{mode}.pkl"
            sol_pkl  = RESULTS_DIR / f"Sol_athlete{num}_complet_{mode}.pkl"

            if (not use_pkl) or (not full_pkl.exists()):
                ocp = prepare_ocp(
                    biorbd_model_path=model_path,
                    final_time=(1, 0.5, 1),
                    n_shooting=n_shooting,
                    min_time=0.01,
                    max_time=2.0,
                    total_mass=total_mass,
                    init_sol=False,
                    weight_control=0.0001,
                    weight_time=1.0,
                    coef_fig=1.0,
                    final_state_bound=True,
                    mode=mode,
                    n_threads=max(1, (os.cpu_count() or 4) - 2),
                    use_sx=False,
                )

                ocp.add_plot_penalty(CostType.ALL)
                ocp.print(to_console=False, to_graph=False)

                solver = Solver.IPOPT()
                solver.set_linear_solver("ma57")
                solver.set_maximum_iterations(20000)

                print(f"start solving (full, {mode})")
                sol = ocp.solve(solver)
                print(f"solving finished (full, {mode})")

                q, qd = stack_states(sol)
                tau   = stack_controls(sol)
                t     = step_time(sol)
                save_arrays(full_pkl, q, qd, tau, t)
                print("data of full solution saved")

                sol.print_cost()
                sol.graphs(show_bounds=True, show_now=True)

                with open(sol_pkl, "wb") as f:
                    del sol.ocp
                    pickle.dump(sol, f)
                print("object solution of full solution saved")

            # Optional animation example
            # if num == 102:
            #     viewer = "pyorerun"
            #     sol.animate(n_frames=0, viewer=viewer, show_now=True)

if __name__ == "__main__":
    main()
