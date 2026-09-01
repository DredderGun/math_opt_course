import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


class StepSize:
    def __call__(self, x, h, k, *args, **kwargs):
        pass


class Descent_step(StepSize):
    def __init__(self, d=0.5, alpha0=np.sqrt(2), norm_order=2, alpha_function=None):
        self.norm_ord = norm_order
        if alpha_function is not None:
            self.alpha_function = alpha_function
        else:
            self.alpha_function = lambda k, subgrad: alpha0/(np.linalg.norm(subgrad, ord=self.norm_ord) * (k+1)**d)
    
    def __call__(self, x, h, k, *args, **kwargs):
        return self.alpha_function(k, h)


def euclidean_projection_onto_simplex(v):
    """Project onto the simplex Δ_n = {x : x ≥ 0, Σx = 1}"""
    tol = 1e-12
    if np.sum(v) <= 1 + tol and np.all(v >= -tol):
        return v
    n = len(v)
    u = np.sort(v)[::-1]  # Sort in descending order
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def subgrad_method(f, subgrad_f, proj_op, step, x0, max_iters=1000, tol=1e-6):
    history = []
    x = x0.copy()

    for i in range(max_iters):
        subgrad = subgrad_f(x)
        alpha = step(x, subgrad, i, gradf=subgrad_f, f=f)
        x_new = x - alpha * subgrad
        x_new = proj_op(x_new)
        f_current = f(x)
        if i == 0:
            history.append((f_current, x.copy()))
        else:
            f_best = min(f_current, history[-1][0])
            x_best = x.copy() if f_current < history[-1][0] else history[-1][1].copy()
            history.append((f_best, x_best))
        x = x_new

    return x, history


def mirror_descent_method(f, subgrad_f, step, x0, max_iters=1000, tol=1e-6):
        history = []
        x = x0.copy()

        for i in range(max_iters):
            subgrad = subgrad_f(x)
            alpha = step(x, subgrad, i, gradf=subgrad_f, f=f)

            assert np.all(x >= 0), "Negative x"

            z = -alpha * subgrad
            z -= np.max(z)
            num = x * np.exp(z)
            den = np.sum(num)

            x_new = num / den
            f_current = f(x)
            if i == 0:
                history.append((f_current, x.copy()))
            else:
                f_best = min(f_current, history[-1][0])
                x_best = x.copy() if f_current < history[-1][0] else history[-1][1].copy()
                history.append((f_best, x_best))
            x = x_new

        return x, history


def ax_b_one_norm_objective(A, b, x):
    return np.linalg.norm(np.dot(A, x) - b, ord=1)


def ax_b_one_norm_objective_subgradient(A, b, x, rng):
    """
    Compute a subgradient of ||Ax - b||₁ at x.
    """
    r = np.dot(A, x) - b
    s = np.sign(r)
    
    zero_tol = 1e-12
    zero_mask = np.abs(r) < zero_tol
    
    if np.any(zero_mask):
        s[zero_mask] = rng.uniform(-1, 1, size=np.sum(zero_mask))
    
    return np.dot(A.T, s)


def generate_ax_b_one_norm_params(n, m, rng):
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)

    x = cp.Variable(n, nonneg=True)
    objective = cp.Minimize(cp.norm1(A @ x - b))
    constraints = [cp.sum(x) == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.SCS, verbose=False)
    if problem.status != cp.OPTIMAL and problem.status != cp.OPTIMAL_INACCURATE:
        raise ValueError("CVXPY did not converge")
    x_true = x.value
    
    return A, b, x_true


def main():
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(15, 4), sharex=True
    )

    n = int(100)
    m = int(1.5 * n)

    max_iters = 500
    tol = 1e-6
    nmbr_of_runs = 50
    x0 = np.ones(n) / n

    all_loss_subgrad = []
    all_loss_mirror_desc = []

    for seed in range(nmbr_of_runs):
        rng = np.random.default_rng(seed)

        A, b, x_true = generate_ax_b_one_norm_params(n, m, rng)

        target_f = lambda x: ax_b_one_norm_objective(A, b, x)
        target_f_subg = lambda x: ax_b_one_norm_objective_subgradient(A, b, x, rng)
        f_sol = target_f(x_true)

        _, history = subgrad_method(
            f=target_f,
            subgrad_f=target_f_subg,
            proj_op=euclidean_projection_onto_simplex,
            step=Descent_step(norm_order=2),
            x0=x0,
            max_iters=max_iters,
            tol=tol
        )
        all_loss_subgrad.append([f_best for f_best, x in history])

        _, history = mirror_descent_method(
            f=target_f,
            subgrad_f=target_f_subg,
            step=Descent_step(norm_order=np.inf),
            x0=x0,
            max_iters=max_iters,
            tol=tol
        )
        all_loss_mirror_desc.append([f_best for f_best, x in history])

        all_loss_subgrad[-1] = np.array(all_loss_subgrad[-1]) - f_sol
        all_loss_mirror_desc[-1] = np.array(all_loss_mirror_desc[-1]) - f_sol

    all_loss_subgrad = np.array(all_loss_subgrad)
    all_loss_mirror_desc = np.array(all_loss_mirror_desc)

    median_loss_subgrad = np.median(all_loss_subgrad, axis=0)
    median_loss_mirror_desc = np.median(all_loss_mirror_desc, axis=0)

    axes[0].plot(median_loss_subgrad, label="Subgradient")
    axes[0].plot(median_loss_mirror_desc, label="mirror descent")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("f(x) - f*")
    axes[0].set_xlabel("Iteration")
    axes[0].grid(True)
    axes[0].legend()
    axes[0].set_title("Convergence of subgradient and mirror descent")

    axes[1].plot(median_loss_subgrad * (np.arange(1, len(median_loss_subgrad)+1)**0.5), label="subgradient * sqrt(k)")
    axes[1].plot(median_loss_mirror_desc * (np.arange(1, len(median_loss_mirror_desc)+1)**0.5), label="mirror descent * sqrt(k)")
    axes[1].grid(True)
    axes[1].legend()
    axes[1].set_title("Theoretical convergence rates")

    plt.tight_layout()
    plt.savefig("convergence.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
