import matplotlib.pyplot as plt
import numpy as np

def plot_utility_gap(k_values, eps_values, util_gap_k, util_gap_eps):
    # --- k-band ---
    plt.figure(figsize=(10,6), dpi=140)
    means = [util_gap_k[k][0] for k in k_values]
    ses   = [util_gap_k[k][1] for k in k_values]

    plt.errorbar(k_values, means, yerr=1.96*np.array(ses),
                 fmt="-o", color="#E69F00", capsize=5, label="k-band SD")
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("k-band size k")
    plt.ylabel("Utility gap vs RSD")
    plt.title("Total Utility Gap vs k-band SD")
    plt.tight_layout()
    plt.show()

    # --- epsilon ---
    plt.figure(figsize=(10,6), dpi=140)
    means = [util_gap_eps[e][0] for e in eps_values]
    ses   = [util_gap_eps[e][1] for e in eps_values]

    plt.errorbar(eps_values, means, yerr=1.96*np.array(ses),
                 fmt="--s", color="#0072B2", capsize=5, label="ε-SD")
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("ε (noise)")
    plt.ylabel("Utility gap vs RSD")
    plt.title("Total Utility Gap vs ε-Serial Dictatorship")
    plt.tight_layout()
    plt.show()


def plot_fairness_k(k_values, mean_gap_k, se_gap_k=None):
    plt.figure(figsize=(10,6), dpi=140)

    means = [mean_gap_k[k] for k in k_values]

    if se_gap_k is not None:
        ses = [se_gap_k[k] for k in k_values]
        plt.errorbar(
            k_values, means,
            yerr=1.96*np.array(ses),
            fmt='-o', capsize=5,
            color="#E69F00", linewidth=2.5,
            label="k-band SD"
        )
    else:
        plt.plot(k_values, means, '-o',
                 color="#E69F00", linewidth=2.5,
                 label="k-band SD")

    plt.axhline(0, linestyle='--', color='gray')

    plt.xlabel("Band size k", fontsize=15)
    plt.ylabel("Fairness gap (Gini − Gini(RSD))", fontsize=15)
    plt.title("Fairness vs k-band Serial Dictatorship", fontsize=17)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_fairness_eps(eps_values, mean_gap_eps, se_gap_eps=None):
    plt.figure(figsize=(10,6), dpi=140)

    means = [mean_gap_eps[e] for e in eps_values]

    if se_gap_eps is not None:
        ses = [se_gap_eps[e] for e in eps_values]
        plt.errorbar(
            eps_values, means,
            yerr=1.96*np.array(ses),
            fmt='--s', capsize=5,
            color="#0072B2", linewidth=2.5,
            label="ε-SD"
        )
    else:
        plt.plot(eps_values, means, '--s',
                 color="#0072B2", linewidth=2.5,
                 label="ε-SD")

    plt.axhline(0, linestyle='--', color='gray')

    plt.xlabel("ε (noise magnitude)", fontsize=15)
    plt.ylabel("Fairness gap (Gini − Gini(RSD))", fontsize=15)
    plt.title("Fairness vs ε-Serial Dictatorship", fontsize=17)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_fairness_displacement_frontier(rank_i, k_values, eps_values,
                  mean_gap_k, mean_gap_eps,
                  disruption, signed=True):

    plt.figure(figsize=(10,7), dpi=140)

    if signed:
        disp_k = disruption["k_signed"]
        disp_e = disruption["eps_signed"]
        ylabel = "Expected signed displacement"
        title = f"Fairness vs signed disruption — rank {rank_i}"
    else:
        disp_k = disruption["k_abs"]
        disp_e = disruption["eps_abs"]
        ylabel = "Expected absolute displacement"
        title = f"Fairness vs absolute disruption — rank {rank_i}"

    # --- k-band ---
    xs = [mean_gap_k[k] for k in k_values]
    if signed:
        ys = [-disp_k[k][rank_i] for k in k_values]
    else:
        ys = [disp_k[k][rank_i] for k in k_values]

    plt.plot(xs, ys, '-o', color="#E69F00", label="k-band SD")
    for x, y, k in zip(xs, ys, k_values):
        plt.text(x+0.002, y+0.03, f"k={k}", color="#E69F00")

    # --- epsilon ---
    xs = [mean_gap_eps[e] for e in eps_values]
    if signed:
        ys = [-disp_e[e][rank_i] for e in eps_values]
    else:
        ys = [disp_e[e][rank_i] for e in eps_values]

    plt.plot(xs, ys, '--s', color="#0072B2", label="ε-SD")
    for x, y, eps in zip(xs, ys, eps_values):
        plt.text(x+0.002, y-0.05, f"ε={eps}", color="#0072B2")

    plt.xlabel("Fairness gap (Gini − Gini(RSD))", fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title, fontsize=17)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_fairness_envy_frontier(k_values, eps_values,
                  mean_gap_k, mean_gap_eps, mean_envy_k, mean_envy_eps):

    plt.figure(figsize=(10,7), dpi=140)

    ylabel = "Average Justified Envy"
    title = f"Fairness vs average justified envy"

    # --- k-band ---
    xs = [mean_gap_k[k] for k in k_values]
    ys = [mean_envy_k[k] for k in k_values]

    plt.plot(xs, ys, '-o', color="#E69F00", label="k-band SD")
    for x, y, k in zip(xs, ys, k_values):
        plt.text(x+0.002, y+0.03, f"k={k}", color="#E69F00")

    # --- epsilon ---
    xs = [mean_gap_eps[e] for e in eps_values]
    ys = [mean_envy_eps[e] for e in eps_values]

    plt.plot(xs, ys, '--s', color="#0072B2", label="ε-SD")
    for x, y, eps in zip(xs, ys, eps_values):
        plt.text(x+0.002, y-0.05, f"ε={eps}", color="#0072B2")

    plt.xlabel("Fairness gap (Gini − Gini(RSD))", fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.title(title, fontsize=17)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

