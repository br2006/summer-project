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
    def __init__(self, l, I, theta_0, v):
        self.l = l
        self.I = I
        self.m = 0.5
        self.g = 9.81
        self.bp = 0.01     #ADDED DAMPING COEFF   
        self.theta = theta_0
        self.dtheta = v / self.l
        self.ddtheta = 0

    def apply_forces(self, tt, wheel):
        torque_gravity = self.m * self.g * self.l * np.sin(self.theta)
        torque_damping = self.bp * self.dtheta
        torque_control = wheel.balance(self)   # opposite reaction on pendulum
        self.ddtheta = (torque_gravity + torque_damping + torque_control) / self.I
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
    def __init__(self, I, kp, ki, kd):
        self.I = I #ADDED INERTIA VALUE AND REMOVED MASS/RADIUS
        self.bw = 0.001     #ADDED DAMPING COEFF
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = 1/(TPF*FPS)

        self.phi = 0
        self.dphi = 0
        self.ddphi = 0

        self.integral = 0
        self.maxtorque = 1.0 

    def wrap_angle(self, angle):
        return (angle + np.pi) % (2*np.pi) - np.pi
    

    def balance(self, pendulum):
        error = self.wrap_angle(pendulum.theta)
        self.integral += error * self.dt
        torque = -(self.kp * error + self.kd * pendulum.dtheta + self.ki * self.integral)
        torque = np.clip(torque, -self.maxtorque, self.maxtorque)
        self.ddphi = torque + (self.bw * self.dphi) / self.I #ADDED DAMPING TERM
        self.dphi += self.ddphi * self.dt
        self.phi += self.dphi * self.dt

        return torque


# PARAMETERS TO CHANGE
pend_height = 0.1
pend_inertia = 0.02
pend_theta_0 = 0.3
pend_initial_v = 0
wheel_inertia = 0.002
wheel_kp = 2
wheel_ki = 0.25
wheel_kd = 0.1

pend = Pendulum(pend_height, pend_inertia, pend_theta_0, pend_initial_v)
wheel = Reaction(wheel_inertia, wheel_kp, wheel_ki, wheel_kd)
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

