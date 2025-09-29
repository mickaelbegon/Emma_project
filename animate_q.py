import matplotlib
matplotlib.use("Qt5Agg")
import numpy as np
import math

import matplotlib.pyplot as plt
from pyorerun import BiorbdModel, PhaseRerun

import os
import pickle

athlete = 1
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CURRENT_DIR, "applied_examples", "results3")  # <- bon dossier
modelname = os.path.join(CURRENT_DIR, "applied_examples", f"athlete_{athlete}_deleva.bioMod")
model = BiorbdModel(modelname)


file_pkl = os.path.join(RESULTS_DIR, f"athlete{athlete}_base.pkl")#
# file_pkl = os.path.join(RESULTS_DIR, f"athlete{athlete}_retroversion.pkl")
# file_pkl = os.path.join(RESULTS_DIR, f"athlete{athlete}_anteversion.pkl")

if not os.path.exists(file_pkl) or os.path.getsize(file_pkl) == 0:
    raise FileNotFoundError(f"Fichier absent ou vide: {file_pkl}")

with open(file_pkl, "rb") as f:
    data = pickle.load(f)

qs    = data[0]["q"]
qdots = data[0]["qdot"]
taus  = data[0]["tau"]
time  = data[0]["time"]         # <- tu sauvegardes 'time' dans tes pkl
# taudots = data["taudot"]   # <- supprimé: cette clé n'existe pas dans tes pkl

n_shooting = (25, 25, 60)


def plot_all_dofs_states(modelname: str, time: np.ndarray, states: np.ndarray,
                    title: str = "Generalized coordinates (q)"):
    """
    time: shape (T,)       — temps
    q:    shape (n_q, T)   — états q
    biomod: chemin vers le .bioMod (pour lire les noms de DoF)
    """
    model = BiorbdModel(modelname)
    names = [model.nameDof(i).to_string() for i in range(model.nb_q())]
    n_q = states.shape[0]

    # grille carrée ~√n

    ncols = math.ceil(math.sqrt(n_q))
    nrows = math.ceil(n_q / ncols)

    fig, axes = plt.subplots(nrows, ncols, sharex=True, figsize=(3.8*ncols, 2.6*nrows))
    axes = np.atleast_1d(axes).ravel()

    for i in range(n_q):
        ax = axes[i]
        ax.plot(time, states[i, :])
        ax.set_title(names[i], fontsize=9)
        ax.grid(True, alpha=0.3)

    # masquer les axes vides si la grille n’est pas pleine
    for j in range(n_q, nrows*ncols):
        axes[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    axes[max(0, n_q-1)].set_xlabel("Time [s]")
    plt.show()

plot_all_dofs_states(modelname, time,qs,"q")
plot_all_dofs_states(modelname, time,qdots,"qdot")
plot_all_dofs_states(modelname, time,taus,"tau")


viz = PhaseRerun(time)
viz.add_animated_model(model, qs)
viz.rerun("swing")


