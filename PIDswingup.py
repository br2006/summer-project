import pygame
import numpy as np
import time
import sys
from gpt import RealTimePlotAPI
from PyQt6 import QtWidgets
import pyqtgraph as pg

WIDTH, HEIGHT = 800, 600
TPF, FPS = 1, 120  # Ticks per frame, frames per second
temp_t = time.time()

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()


class Pendulum:
    def __init__(self, height, I, theta_0, v):
        self.l = height
        self.I = I
        self.m = 1
        self.g = 9.81
        self.theta = theta_0
        self.dtheta = v / self.l
        self.ddtheta = 0

    def apply_forces(self, tt, wheel):
        torque_gravity = self.m * self.g * self.l * np.sin(self.theta)
        torque_control = wheel.balance(self)   # opposite reaction on pendulum

        self.ddtheta = (torque_gravity + torque_control) / self.I

        dt = 1/(TPF*FPS)
        self.dtheta += self.ddtheta * dt
        self.theta += self.dtheta * dt

        return time.time()

    def draw(self):
        screen.fill((255, 255, 255))
        x = int(400 + 150 * np.sin(self.theta))
        y = int(300 - 150 * np.cos(self.theta))
        pygame.draw.line(screen, (100, 100, 100), (400, 300), (x, y))
        pygame.draw.circle(screen, (0, 0, 0), (x, y), 10)


class Reaction:
    def __init__(self, m, r):
        self.m = m
        self.r = r
        self.I = self.m * self.r**2

        self.kp, self.ki, self.kd = 20.0, 0.0, 5.0
        self.dt = 1/(TPF*FPS)

        self.phi = 0
        self.dphi = 0
        self.ddphi = 0

        self.integral = 0
        self.maxtorque = 1.0

    def wrap_angle(self, angle):
        return (angle + np.pi) % (2*np.pi) - np.pi
    
    def swing_up(self, pendulum):
        theta = self.wrap_angle(pendulum.theta)

        Ek = 0.5 * pendulum.I * (pendulum.dtheta)**2
        Ep= pendulum.m * pendulum.g * pendulum.l * (np.cos(theta) - 1.0)
        Etot = Ek + Ep
        E_error = Etot  # target energy 0 upright

        direction = np.tanh(5.0 * dtheta * np.cos(theta))
        torque = swingspeed * E_error * direction
        return torque

    def balance(self, pendulum):
        error = self.wrap_angle(pendulum.theta)
        self.integral += error * self.dt

        torque = -(self.kp * error + self.kd * pendulum.dtheta + self.ki * self.integral)
        if torque > self.maxtorque:
            torque = self.maxtorque
        elif torque < -self.maxtorque:
            torque = -self.maxtorque            

        self.ddphi = torque / self.I
        self.dphi += self.ddphi * self.dt
        self.phi += self.dphi * self.dt

        return torque


# --- Initialise simulation objects ---
pend = Pendulum(1, 1, 0.3, 0)
wheel = Reaction(1, 1)
count = 0

# --- Set up Qt + pyqtgraph manually (no blocking plotter.start()) ---
plotter = RealTimePlotAPI(
    title='Pendulum State',
    target_fps=10,
    x_window=400,
    line_width=2,
    y_range=[-np.pi, np.pi]  # matches wrapped angle range
)

plotter._running = True
plotter._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
plotter._win = pg.GraphicsLayoutWidget(show=True, title=plotter.title)
plotter._win.resize(*plotter.window_size)
plotter._plot = plotter._win.addPlot(title=plotter.title)
plotter._plot.showGrid(x=True, y=True)
plotter._plot.addLegend()
if plotter.y_range:
    plotter._plot.setYRange(*plotter.y_range)


# --- Main loop: pygame + Qt ticked together on the main thread ---
while True:
    count += 1
    clock.tick(TPF * FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_RIGHT]:
                pend.dtheta += 10
            elif keys[pygame.K_LEFT]:
                pend.dtheta -= 10

    pend.apply_forces(temp_t, wheel)
    pend.draw()

    # Plot wrapped angle (continuous error signal), velocity, acceleration
    wrapped_theta = (pend.theta + np.pi) % (2 * np.pi) - np.pi
    plotter.push('θ (wrapped)', count, wrapped_theta)
    plotter.push('dθ/dt', count, pend.dtheta)
    plotter.push('d²θ/dt²', count, pend.ddtheta)

    if not count % TPF:
        pygame.display.update()
        plotter._update()         # manually tick the plot
        plotter._app.processEvents()  # manually tick Qt event loop