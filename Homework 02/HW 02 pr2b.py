# %%
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as ppl


# defines the distance squared function
def distance_sq(x):
    eq = (2 - 2 * x[0] - 3 * x[1]) ** 2 + x[0] ** 2 + (x[1] - 1) ** 2
    return eq


# defines the gradient of the distance squared function
def distance_sq_g(x):
    der = np.zeros_like(x)
    der[0] = -8 + 10 * x[0] + 12 * x[1]
    der[1] = -14 + 12 * x[0] + 20 * x[1]
    return der


def distance_sq_H():
    H = np.zeros((2, 2))
    H = H + np.diag([10, 20])
    H[0, 1] = 12
    H[1, 0] = H[0, 1]
    return H


def inexact_line_search(function, g0, x0, t):
    alpha = 1
    counter = 0
    func_eval = function(x0 - alpha * g0)
    phi_eval = function(x0) - t * g0.T @ g0 * alpha
    while func_eval > phi_eval and counter < 100:
        alpha = alpha / 2
        counter += 1
        func_eval = function(x0 - alpha * g0)
        phi_eval = function(x0) - t * g0.T @ g0 * alpha
    xnew = x0 - alpha * g0
    return xnew


def gradient_decent_inexact_line_search(function, gradient, x0, t, tollerance):
    counter = 0
    g0 = gradient(x0)
    g0norm = np.linalg.norm(g0)
    x0track = x0
    distance_track = np.array([function(x0)])
    while g0norm > tollerance and counter < 100:
        counter += 1
        x0 = inexact_line_search(function, g0, x0, t)
        g0 = gradient(x0)
        g0norm = np.linalg.norm(g0)
        x0track = np.vstack((x0track, x0))
        distance0 = np.array([function(x0)])
        distance_track = np.concatenate((distance_track, distance0))
    success = g0norm < tollerance
    distance = function(x0)
    print("was a success: " + str(success) + "\nx values = "), print(x0), print("\nMinimum distance: "), print(distance)
    return x0, x0track, distance_track


x0 = np.array([1, 1])
(x_final, x_track, distance_track) = gradient_decent_inexact_line_search(distance_sq, distance_sq_g, x0, 0.5, 1e-3)
iteration_count = np.count_nonzero(distance_track, axis=0)
print('completed in ' + str(iteration_count) + ' iterations')

x_axis = np.arange(iteration_count - 1)
y_axis = distance_track[:iteration_count - 1] - distance_track[-1]

fig, ax = ppl.subplots()
ax.set_yscale('log')
ax.plot(x_axis, y_axis, label='Gradient Decent')
plt.xlabel('iteration')
plt.ylim(bottom=10 ** -8)
plt.ylabel('Error')
plt.grid(True)
plt.title('Convergence Plot')
plt.show()
