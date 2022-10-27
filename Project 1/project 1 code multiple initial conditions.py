import logging
import math
import random
import numpy as np
import time

import torch
import torch as t
import torch.nn as nn
from torch import optim
from torch.nn import utils
import matplotlib.pyplot as plt

t.manual_seed(13)

logger = logging.getLogger(__name__)

FRAME_TIME = 1.5
GRAVITY_ACCEL = 9.81 / 1000
BOOST_ACCEL = 14.715 / 1000
L_center_of_gravity = 5 / 1000

PLATFORM_WIDTH = 25 / 1000


class Dynmaics(nn.Module):

    def __init__(self):
        super(Dynmaics, self).__init__()

    @staticmethod
    def forward(state, action):
        """
        action[0]: thrust
        action[1]: phi of thrust
        state[0] = y
        state[1] = y_dot
        state[2] = x
        state[3] = x_dot
        state[4] = theta
        state[5] = omega
        """
        delta_state_gravity = t.tensor([0., -GRAVITY_ACCEL * FRAME_TIME, 0., 0., 0., 0])
        N = len(state)
        # Thrust Calculations
        state_tensor = torch.zeros((N, 6))
        state_tensor[:, 1] = torch.cos(state[:, 4] + action[:, 1])
        state_tensor[:, 3] = torch.sin(state[:, 4] + action[:, 1])
        delta_state_thrust = BOOST_ACCEL * FRAME_TIME * t.mul(state_tensor, action[:, 0].reshape(-1, 1))

        state_tensor_angle = t.zeros((N, 6))
        state_tensor_angle[:, 5] = 4 / L_center_of_gravity * t.sin(action[:, 1]) * action[:, 0]
        delta_state_angle = BOOST_ACCEL * FRAME_TIME * state_tensor_angle

        state = state + delta_state_thrust + delta_state_gravity + delta_state_angle

        step_mat = t.tensor([[1., FRAME_TIME, 0., 0., 0., 0.],
                             [0., 1., 0., 0., 0., 0.],
                             [0., 0., 1., FRAME_TIME, 0., 0.],
                             [0., 0., 0., 1., 0., 0.],
                             [0., 0., 0., 0., 1., FRAME_TIME],
                             [0., 0., 0., 0., 0., 1]])
        state = t.t(t.matmul(step_mat, state.t()))

        return state


class Controller(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        dim_input: # of system states
        dim_output: # of actions
        dim_hidden: up to me
        """

        super(Controller, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output),
            nn.Sigmoid(),
            nn.Linear(dim_output, dim_input),
            nn.Tanh(),
            nn.Linear(dim_input, dim_output),
            nn.Tanh()
        )

    def forward(self, state):
        action = self.network(state)
        action = action / 2 + t.tensor([0.5, 0.])
        return action


class Simulation(nn.Module):

    def __init__(self, controller, dynamics, T):
        super(Simulation, self).__init__()
        self.state = self.initialize_state()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.action_trajectory = []
        self.state_trajectory = []

    def forward(self, state):
        self.action_trajectory = []
        self.state_trajectory = []
        for _ in range(T):
            action = self.controller.forward(state)
            state = self.dynamics.forward(state, action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state)
        return self.error(state, self.state_trajectory)

    @staticmethod
    def initialize_state():
        y = np.random.normal(1.0, 0.05, 6)
        y_dot = np.random.rand(8) * -0.005
        x = np.random.normal(0., 0.05, 8)
        x_dot = np.random.normal(0., 0.005, 8)
        theta = np.random.normal(0., 0.1, 8)
        omega = np.random.normal(0., 0.05, 8)
        state = np.zeros((len(y), 6))
        for i in range(len(y)):
            state[i, :] = [y[i], y_dot[i], x[i], x_dot[i], theta[i], omega[i]]# need initial conditions
        return t.tensor(state, requires_grad=False).float()

    def error(self, state, state_trajectory):
        termination_error = 10 * (state[:, 0] - L_center_of_gravity) ** 2 + 10 * state[:, 1] ** 2 + state[:, 2] ** 2 + 2 * state[:, 3] ** 2 + state[:, 4] ** 2 + state[:, 5] ** 2
        termination_error = t.sum(termination_error)
        stack_state_traj = t.stack(state_trajectory)
        x_location = stack_state_traj[:-1, :, 2]
        x_location_error_squared = t.sum(x_location**2, (1, 0))
        angle = stack_state_traj[:-1, :, 4]
        angle_error_squared = t.sum(angle**2, (0, 1))
        y_speed = stack_state_traj[-4:-1, :, 1]
        y_speed_error_squared = t.sum(y_speed**2, (1, 0))
        y = stack_state_traj[:-1, :, 0]
        y = t.nn.functional.relu(-(y - L_center_of_gravity))
        y_squared_error = t.sum(y**2, (1, 0))
        omega = stack_state_traj[-6:-1, :, 5]
        omega_error_sq = t.sum(omega**2, (1, 0))
        transition_error = 0.03 * x_location_error_squared + 0.07 * angle_error_squared + 0.8 * y_speed_error_squared + 0.06 * omega_error_sq  + 0.8 * y_squared_error

        return termination_error + transition_error


class Optimize:
    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.012)

    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.backward()
            return loss

        self.optimizer.step(closure)
        return closure()

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.step()
            print('[%d] loss: %.5f' % (epoch + 1, loss))
            self.visualize()


    def visualize(self):
        data = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        N_cond = len(data[0, :, 0])
        fig = plt.figure(tight_layout=True)
        import matplotlib.gridspec as grsp
        gs = grsp.GridSpec(3, 2)
        axyydot = fig.add_subplot(gs[0, :])
        axx = fig.add_subplot(gs[1, 0])
        axxdot = fig.add_subplot(gs[1, 1])
        axthet = fig.add_subplot(gs[2, 0])
        axomeg = fig.add_subplot(gs[2, 1])
        for i in range(N_cond):
            y = data[:, i, 0]
            y_dot = data[:, i, 1]
            x = data[:, i, 2]
            x_dot = data[:, i, 3]
            theta = data[:, i, 4]
            omega = data[:, i, 5]

            axyydot.plot(y, y_dot)
            axx.plot(y, x)
            axxdot.plot(y, x_dot)
            axthet.plot(y, theta)
            axomeg.plot(y, omega)

        axyydot.set_xlabel('Normalized Vertical Distance')
        axx.set_xlabel('Normalized Vertical Distance')
        axxdot.set_xlabel('Normalized Vertical Distance')
        axthet.set_xlabel('Normalized Vertical Distance')
        axomeg.set_xlabel('Normalized Vertical Distance')
        axyydot.set_ylabel('Normalized y_dot')
        axx.set_ylabel('Normalized x')
        axxdot.set_ylabel('Normalized x_dot')
        axthet.set_ylabel('Theta of Rocket')
        axomeg.set_ylabel('Omega of Rocket')
        fig.align_labels()
        plt.show()


T = 25
dim_input = 6
dim_hidden = 12
dim_output = 2
d = Dynmaics()
c = Controller(dim_input, dim_hidden, dim_output)
s = Simulation(c, d, T)
o = Optimize(s)
o.train(230)
