import numpy as np

def myQP(x, W, df, g, dg):
    # min   (1/2) * sT * W * s + cT * s
    # s.t.  A*s+b <= 0
    A0 = dg(x)
    b0 = g(x)
    c = df(x)
    mu0 = np.zeros((b0.shape[0], 1))  #TODO check where mu0 and mu should be used
    mu = []
    active = []
    while True:
        if len(active) == 0:
            M = W
            smu = np.matmul(np.linalg.inv(M), -c)
            s = smu[:len(x), :]
            mu = []

        if len(active) != 0:
            A = A0[active, :]
            b = b0[active]
            s, mu = solve_activeset(x, W, c, A, b)

        contstraints_ceeck = np.round((np.matmul(A0, s.reshape(-1, 1)) + b0))
        mu_check = 0
        if len(mu) == 0:
            mu_check = 1
        elif min(mu) > 0:
            mu_check = 1
        else:
            id_mu = np.argmin(np.array(mu))
            mu.remove(min(mu))
            active.pop(id_mu)

        if np.max(contstraints_ceeck) <= 0:
            if mu_check == 1:
                return s, mu
        else:
            index = np.argmax(contstraints_ceeck)
            active.append(index)
            active = np.unique(np.array(active)).tolist()




def solve_activeset(x, W, c, A, b):
    M = np.vstack((np.hstack((W, np.transpose(A))), np.hstack((A, np.zeros((np.size(A, 0), np.size(A, 0)))))))
    U = np.vstack((-c, -b))
    sol = np.matmul(np.linalg.inv(M), U)

    s = sol[: len(x)]
    mu = sol[len(x):]
    return s, mu
