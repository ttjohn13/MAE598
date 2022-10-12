import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

X1 = np.arange(0, 1.1, 0.1)
X2 = np.flip(X1).copy()
X1 = torch.tensor(X1, requires_grad=False)
X2 = torch.tensor(X2, requires_grad=False)
a = np.array([[8.07131, 1730.63, 233.426], [7.43155, 1554.679, 240.337]])
T = 20
p_water = 10 ** (a[0, 0] - a[0, 1] / (T + a[0, 2]))
p_dioxane = 10 ** (a[1, 0] - a[1, 1] / (T + a[1, 2]))
p_data = np.array([[28.1, 34.4, 36.7, 36.9, 36.8, 36.7, 36.5, 35.4, 32.9, 27.7, 17.5]])

p_data = torch.tensor(p_data, requires_grad=False)

A = Variable(torch.tensor([1.0, 1.0]), requires_grad=True)


def pressure(A, x1, x2, p_w, p_dio):
    pressure_predicted = x1 * torch.exp(A[0] * (A[1] * x2 / (A[0] * x1 + A[1] * x2)) ** 2) * p_w + \
                         x2 * torch.exp(A[1] * (A[0] * x1 / (A[0] * x1 + A[1] * x2)) ** 2) * p_dio
    return pressure_predicted


def loss(predicted_pres, measured_pressure):
    error = (predicted_pres - measured_pressure) ** 2
    errorsum = error.sum()
    return errorsum


def inexact_line_search(function, gradient, initialpoint, slope):
    alpha = 1
    counterlinesearch = 0
    func_eval = function(pressure((initialpoint - alpha * gradient), X1, X2, p_water, p_dioxane), p_data)
    transposedgrad = torch.transpose(gradient, -1, 0)
    phi_eval = function(pressure(initialpoint, X1, X2, p_water, p_dioxane), p_data) - slope * torch.matmul(transposedgrad, gradient) * alpha
    while func_eval> phi_eval and counterlinesearch < 100:
        alpha = alpha / 2
        counterlinesearch += 1
        func_eval = function(pressure((initialpoint - alpha * gradient), X1, X2, p_water, p_dioxane), p_data)
        transposedgrad = torch.transpose(gradient, -1, 0)
        phi_eval = function(pressure(initialpoint, X1, X2, p_water, p_dioxane), p_data) - slope * torch.matmul(
            transposedgrad, gradient) * alpha
    xnew = initialpoint - alpha * gradient
    return alpha



tollerance = 1e-3
counter = 0
keepgoing = True

while counter < 100 and keepgoing == True:
    predicted = pressure(A, X1, X2, p_water, p_dioxane)
    squarederror = loss(predicted, p_data)

    squarederror.backward()
    alpha = inexact_line_search(loss, A.grad, A, 0.5)
    gradientnorm = torch.norm(A.grad)
    if gradientnorm < tollerance:
        keepgoing = False
    with torch.no_grad():
        A -= alpha * A.grad

        A.grad.zero_()
    counter += 1

print('estimation A12 and A21 is:', A)
print('final loss is:', squarederror.data.numpy())

x1predicted = np.linspace(0, 1, 100)
x2predicted = np.flip(x1predicted).copy()
ypredicted = pressure( A, torch.tensor(x1predicted), torch.tensor(x2predicted), p_water, p_dioxane)
ypredicted = ypredicted.detach().numpy()

fig, ax = plt.subplots()
ax.plot(x1predicted, ypredicted)
ax.scatter(X1.numpy(), p_data.numpy(), color='red')
plt.ylabel('Pressure')
plt.xlabel('Composition of Water')
plt.show()
