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
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def subgrad_method(f, subgrad_f, proj_op, step, x0, max_iters=1000, tol=1e-6, history_track_best=False):
    history = []
    x = x0.copy()

    for i in range(max_iters):
        subgrad = subgrad_f(x)
        alpha = step(x, subgrad, i, gradf=subgrad_f, f=f)
        x_new = x - alpha * subgrad
        x_new = proj_op(x_new)
        f_current = f(x)
        if history_track_best:
            if i == 0:
                history.append((f_current, x.copy()))
            else:
                f_best = min(f_current, history[-1][0])
                x_best = x.copy() if f_current < history[-1][0] else history[-1][1].copy()
                history.append((f_best, x_best))
        else:
            history.append((f_current, x.copy()))
        x = x_new

    return x, history


def mirror_descent_method(f, subgrad_f, step, x0, max_iters=1000, tol=1e-6, history_track_best=False):
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
            if history_track_best:
                if i == 0:
                    history.append((f_current, x.copy()))
                else:
                    f_best = min(f_current, history[-1][0])
                    x_best = x.copy() if f_current < history[-1][0] else history[-1][1].copy()
                    history.append((f_best, x_best))
            else:
                history.append((f_current, x.copy()))
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


def simplex_to_2d(x):
    """
    Map 3D simplex point to 2D barycentric coordinates.
    x must satisfy x1 + x2 + x3 = 1.
    """
    v1 = np.array([0.0, 0.0])
    v2 = np.array([1.0, 0.0])
    v3 = np.array([0.5, np.sqrt(3)/2])

    return x[0]*v1 + x[1]*v2 + x[2]*v3


def main():
    fig, ax = plt.subplots()

    n = 3
    m = 6

    max_iters = 7
    tol = 1e-6
    x0 = np.ones(n) / n

    track_subgrad = []
    track_mirror_desc = []

    rng = np.random.default_rng(1234)
    A, b, x_true = generate_ax_b_one_norm_params(n, m, rng)

    target_f = lambda x: ax_b_one_norm_objective(A, b, x)
    target_f_subg = lambda x: ax_b_one_norm_objective_subgradient(A, b, x, rng)

    _, history = subgrad_method(
        f=target_f,
        subgrad_f=target_f_subg,
        proj_op=euclidean_projection_onto_simplex,
        step=Descent_step(norm_order=2),
        x0=x0,
        max_iters=max_iters,
        tol=tol,
        history_track_best=False
    )
    track_subgrad = [x for f_best, x in history]

    _, history = mirror_descent_method(
        f=target_f,
        subgrad_f=target_f_subg,
        step=Descent_step(norm_order=np.inf),
        x0=x0,
        max_iters=max_iters,
        tol=tol,
        history_track_best=False
    )
    track_mirror_desc = [x for f_best, x in history]

    subgrad_2d = [simplex_to_2d(x) for x in track_subgrad]
    mirror_desc_2d = [simplex_to_2d(x) for x in track_mirror_desc]
    
    subgrad_2d = np.array(subgrad_2d)
    mirror_desc_2d = np.array(mirror_desc_2d)
    opt_2d = simplex_to_2d(x_true)
    x0_2d = simplex_to_2d(x0)

    fig, ax = plt.subplots(figsize=(6,6))

    triangle = np.array([
        simplex_to_2d(np.array([1,0,0])),
        simplex_to_2d(np.array([0,1,0])),
        simplex_to_2d(np.array([0,0,1])),
        simplex_to_2d(np.array([1,0,0]))
    ])
    ax.plot(triangle[:,0], triangle[:,1], 'k-')

    ax.plot(subgrad_2d[:,0], subgrad_2d[:,1], c='blue', label='Subgradient Method', marker='o')
    ax.plot(mirror_desc_2d[:,0], mirror_desc_2d[:,1], c='red', label='Mirror Descent', marker='o')

    ax.scatter(x0_2d[0], x0_2d[1], c='gray', s=150, marker='o', label='Start')
    ax.scatter(opt_2d[0], opt_2d[1], c='gray', s=150, marker='*', label='Optimal')
    
    ax.legend()
    ax.set_aspect('equal')
    ax.set_title("Optimization tracks in 2D projection of simplex")
    plt.tight_layout()
    plt.savefig("tracks.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
