import numpy as np
import matplotlib.pyplot as plt

class StepSize:
    def __call__(self, x, h, k, *args, **kwargs):
        pass


class Subgrad_descent_step(StepSize):
    def __init__(self, d=1, alpha0=1, alpha_function=None):
        if alpha_function is not None:
            self.alpha_function = alpha_function
        else:
            self.alpha_function = lambda k, subgrad: alpha0/(k+1)**d
    
    def __call__(self, x, h, k, *args, **kwargs):
        return self.alpha_function(k, h)


class PolyakStepSize(StepSize):
    def __init__(self, f_sol=0, const_div=1.0):
        self.f_sol = f_sol
        self.const_div = const_div

    def __call__(self, x, h, k, gradf, f, *args, **kwargs):
        fx = f(x)
        alpha = (fx - self.f_sol) / (self.const_div * np.linalg.norm(h)**2 + 1e-12)
        return alpha
                

class PolyakBacktrackingStepSize(StepSize):
    def __init__(self, f_sol=0, const_div=1.0, min_div=1e-9, max_div=1e9):
        self.f_sol = f_sol
        self.const_div = const_div
        self.min_div = min_div
        self.max_div = max_div

    def __call__(self, x, h, k, gradf, f, *args, **kwargs):
        fx = f(x)
        gx = gradf(x)

        while True:
            denom = self.const_div * np.linalg.norm(gx) + 1e-12
            alpha = (fx - self.f_sol) / denom
            x_new = x + alpha * h
            fx_new = f(x_new)

            if fx_new < fx:  
                # success → allow slightly larger steps next time
                self.const_div = max(self.const_div / 2.0, self.min_div)
                return alpha
            else:
                # failure → shrink step
                self.const_div *= 2.0
                if self.const_div > self.max_div:
                    # safeguard: if too large, stop backtracking
                    return 0.0


def phase_retrieval(A, b):
    return lambda x: np.mean(np.abs(np.dot(A, x)**2 - b))


def phase_retrieval_subgradient(A, b):
    def g(x):
        Ax = A @ x
        residuals = Ax**2 - b
        s = np.sign(residuals)
        grad_terms = (s * 2 * Ax)[:, None] * A
        return np.mean(grad_terms, axis=0)
    return g


def subgrad_method(f, subgrad_f, step, x0, max_iters=1000, tol=1e-6):
    history = []
    x = x0.copy()

    for i in range(max_iters):
        subgrad = subgrad_f(x)
        if np.allclose(subgrad, 0, atol=tol):
            break
        alpha = step(x, subgrad, i, gradf=subgrad_f, f=f)
        x_new = x - alpha * subgrad
        f_current = f(x)
        if i == 0:
            history.append((f_current, x.copy()))
        elif f_current < history[-1][0]:
            history.append((f_current, x.copy()))
        else:
            history.append(history[-1])
            # history.append((min(history[-1][0], f_current), x.copy()))
        x = x_new

    return x, history

def phase_retrieval_params(observ_nmbr, dim, rng, outliers_fraction=0.1, outliers_magnitude=100.0):
    A = np.random.randn(observ_nmbr, dim)
    x_true = np.random.randn(dim)
    
    b = np.abs(A @ x_true)**2
    indices = rng.choice(observ_nmbr, size=int(outliers_fraction * observ_nmbr), replace=False)
    b[indices] = rng.uniform(0., outliers_magnitude, size=indices.shape)

    return A, b, x_true
    

def main():
    rng = np.random.default_rng()
    dim = 128
    observ_nmbr = 3 * dim

    max_iters = 500
    tol = 1e-6
    nmbr_of_runs = 20
    outliers_magnitude = 100.0
    outliers_fractions = [0.0, 0.1, 0.2]
    x0 = np.ones(dim)

    fig, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(12, 12), sharex=True
    )

    for i, outliers_fraction in enumerate(outliers_fractions):

        all_loss_descent = []
        all_loss_polyak = []
        all_errors_descent = []
        all_errors_polyak = []

        for seed in range(nmbr_of_runs):
            rng = np.random.default_rng(seed)
            A, b, x_true = phase_retrieval_params(
                observ_nmbr, dim, rng,
                outliers_fraction=outliers_fraction,
                outliers_magnitude=outliers_magnitude
            )

            f_target = phase_retrieval(A, b)

            # --- Subgradient descent ---
            _, history_descent = subgrad_method(
                f=f_target,
                subgrad_f=phase_retrieval_subgradient(A, b),
                step=Subgrad_descent_step(
                    d=0.5,
                    alpha_function=lambda k, h: 1.0 / (np.linalg.norm(h) * (k+1)**0.5)
                ),
                x0=x0,
                max_iters=max_iters,
                tol=tol
            )

            all_loss_descent.append([f_best for f_best, x in history_descent])
            all_errors_descent.append([np.linalg.norm(x - x_true) for f_best, x in history_descent])

            # --- Polyak ---
            _, history_polyak = subgrad_method(
                f=f_target,
                subgrad_f=phase_retrieval_subgradient(A, b),
                step=PolyakStepSize(f_sol=0.0, const_div=1.0),
                x0=x0,
                max_iters=max_iters,
                tol=tol
            )

            all_loss_polyak.append([f_best for f_best, x in history_polyak])
            all_errors_polyak.append([np.linalg.norm(x - x_true) for f_best, x in history_polyak])

        # Convert to arrays
        all_loss_descent = np.array(all_loss_descent)
        all_loss_polyak = np.array(all_loss_polyak)
        all_errors_descent = np.array(all_errors_descent)
        all_errors_polyak = np.array(all_errors_polyak)

        # Medians
        median_loss_descent = np.median(all_loss_descent, axis=0)
        median_loss_polyak = np.median(all_loss_polyak, axis=0)
        median_err_descent = np.median(all_errors_descent, axis=0)
        median_err_polyak = np.median(all_errors_polyak, axis=0)

        # -------- Left column: objective --------
        ax_obj = axes[i, 0]
        ax_obj.plot(median_loss_descent, label="Subgradient")
        ax_obj.plot(median_loss_polyak, label="Polyak")
        ax_obj.set_yscale("log")
        ax_obj.set_ylabel("Objective value")
        ax_obj.set_title(f"Outliers: {outliers_fraction:.0%}")
        ax_obj.grid(True)

        if i == 0:
            ax_obj.legend()

        # -------- Right column: error --------
        ax_err = axes[i, 1]
        ax_err.plot(median_err_descent, label="Subgradient")
        ax_err.plot(median_err_polyak, label="Polyak")
        ax_err.set_yscale("log")
        ax_err.set_ylabel("‖x − x⋆‖")
        ax_err.set_title(f"Outliers: {outliers_fraction:.0%}")
        ax_err.grid(True)

    for ax in axes[-1, :]:
        ax.set_xlabel("Iteration")

    plt.tight_layout()
    plt.savefig("convergence.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
