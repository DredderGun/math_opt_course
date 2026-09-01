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
                
def svm_objective(x, W, y, lam):
    margins = 1 - y * (W @ x)
    hinge = np.maximum(0.0, margins)
    loss = hinge.mean()
    reg = (lam / 2.0) * np.dot(x, x)

    return loss + reg


def svm_subgradient(x, W, y, lam):
    margins = 1 - y * (W @ x)
    active = margins > 0

    g_hinge = (-y[active, None] * W[active]).mean(axis=0) if np.any(active) else np.zeros_like(x)

    g = g_hinge + lam * x

    return g


def subgrad_method(f, subgrad_f, step, x0, max_iters=1000, tol=1e-6):
    history = []
    x = x0.copy()

    for i in range(max_iters):
        subgrad = subgrad_f(x)
        if np.allclose(subgrad, 0, atol=tol): # critical point
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
        x = x_new

    return x, history


def svm_params(n, d, rng, noise=0.1):
    W_pos = rng.normal(loc=+1.0, scale=1.0, size=(n // 2, d))
    W_neg = rng.normal(loc=-1.0, scale=1.0, size=(n // 2, d))
    W = np.vstack([W_pos, W_neg])

    y = np.hstack([np.ones(n // 2), -np.ones(n // 2)])

    flip = rng.random(n) < noise
    y[flip] *= -1

    return W, y


def main():
    rng = np.random.default_rng()
    dim = 64
    observ_nmbr = 3 * dim

    max_iters = 500
    tol = 1e-6
    nmbr_of_runs = 15
    reg_params = [0.01, 0.1, 2.0]
    x0 = np.ones(dim)

    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(15, 4), sharex=True
    )

    for i, reg_param in enumerate(reg_params):
        all_loss_descent = []
        all_loss_strong_descent = []
        all_loss_polyak = []

        for seed in range(nmbr_of_runs):
            rng = np.random.default_rng(seed)

            W, y = svm_params(n=observ_nmbr, d=dim, rng=rng, noise=0.05)

            svm_f = lambda x: svm_objective(x, W, y, lam=reg_param)
            svm_subg = lambda x: svm_subgradient(x, W, y, lam=reg_param)

            _, history = subgrad_method(
                f=svm_f,
                subgrad_f=svm_subg,
                step=Subgrad_descent_step(
                    alpha_function=lambda k, h: 1.0 / (np.linalg.norm(h) * (k+1)**0.5)
                ),
                x0=x0,
                max_iters=max_iters,
                tol=tol
            )
            all_loss_descent.append([f_best for f_best, x in history])

            _, history = subgrad_method(
                f=svm_f,
                subgrad_f=svm_subg,
                step=Subgrad_descent_step(
                    alpha_function=lambda k, h: 2.0 / (reg_param * (k+1))
                ),
                x0=x0,
                max_iters=max_iters,
                tol=tol
            )
            all_loss_strong_descent.append([f_best for f_best, x in history])

            _, history = subgrad_method(
                f=svm_f,
                subgrad_f=svm_subg,
                step=PolyakStepSize(f_sol=0.0, const_div=1.0),
                x0=x0,
                max_iters=max_iters,
                tol=tol
            )
            all_loss_polyak.append([f_best for f_best, x in history])

        all_loss_descent = np.array(all_loss_descent)
        all_loss_strong_descent = np.array(all_loss_strong_descent)
        all_loss_polyak = np.array(all_loss_polyak)

        median_loss_descent = np.median(all_loss_descent, axis=0)
        median_loss_strong_descent = np.median(all_loss_strong_descent, axis=0)
        median_loss_polyak = np.median(all_loss_polyak, axis=0)

        min_all_value = min(
            median_loss_descent.min(),
            median_loss_strong_descent.min(),
            median_loss_polyak.min()
        )

        ax_obj = axes[i]
        ax_obj.plot(median_loss_descent - min_all_value, label="Subgradient (1/√k)")
        ax_obj.plot(median_loss_strong_descent - min_all_value, label="Subgradient (1/\lamda k)")
        ax_obj.plot(median_loss_polyak - min_all_value, label="Polyak")
        ax_obj.set_yscale("log")
        ax_obj.set_ylabel("f(x) - f*")
        ax_obj.set_xlabel("Iteration")
        ax_obj.set_title(f"reg param: {reg_param}")
        ax_obj.grid(True)
        ax_obj.legend()

    plt.tight_layout()
    plt.savefig("convergence.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()