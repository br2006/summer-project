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
    def __init__(self, l, I, theta_0, v, m, bp):
        self.l = l
        self.I = I
        self.m = m
        self.g = 9.81
        self.bp = bp     #ADDED DAMPING COEFF   
        self.theta = theta_0
        self.dtheta = v / self.l
        self.ddtheta = 0
        self.tm = 0

    def apply_forces(self, tt, wheel):
        torque_gravity = self.m * self.g * self.l * np.sin(self.theta)
        torque_damping = self.bp * self.dtheta
        torque_control = -wheel.balance(self)   # opposite reaction on pendulum
        self.ddtheta = (torque_gravity + torque_damping + torque_control) / self.I
        dt = 1/(TPF*FPS)
        self.dtheta += self.ddtheta * dt
        self.theta += self.dtheta * dt

        return time.time()

    def draw(self, wheel):
        screen.fill((255, 255, 255))

        # Pivot point
        pivot_x, pivot_y = 400, 300

        # Bob position (end of rod)
        bob_x = int(pivot_x + 110 * np.sin(self.theta))
        bob_y = int(pivot_y - 110 * np.cos(self.theta))

        pygame.draw.line(screen, (80, 80, 80), (pivot_x, pivot_y), (bob_x, bob_y), 4)

        pygame.draw.circle(screen, (60, 60, 60), (pivot_x, pivot_y), 6)

        wheel_radius = 75
        hub_radius   = 5
        num_spokes   = 3
        phi          = wheel.phi   # accumulated wheel angle (radians)

        # Outer rim
        pygame.draw.circle(screen, (30, 30, 30),  (bob_x, bob_y), wheel_radius, 3)
        # Inner rim 
        pygame.draw.circle(screen, (80, 80, 80),  (bob_x, bob_y), wheel_radius - 6, 1)

        # Spokes are rotated by phi 
        for i in range(num_spokes):
            spoke_angle = phi + (i * 2 * np.pi / num_spokes)
            sx = int(bob_x + (wheel_radius - 2) * np.cos(spoke_angle))
            sy = int(bob_y + (wheel_radius - 2) * np.sin(spoke_angle))
            pygame.draw.line(screen, (50, 50, 50), (bob_x, bob_y), (sx, sy), 2)

        #Middle of pendulum
        pygame.draw.circle(screen, (200, 50, 50), (bob_x, bob_y), hub_radius)

        # Rotation marker dot on the rim
        marker_x = int(bob_x + (wheel_radius - 3) * np.cos(phi))
        marker_y = int(bob_y + (wheel_radius - 3) * np.sin(phi))
        pygame.draw.circle(screen, (255, 140, 0), (marker_x, marker_y), 4)


class Reaction:
    def __init__(self, I, kp, ki, kd, maxtorque, bw):
        self.I = I #ADDED INERTIA VALUE AND REMOVED MASS/RADIUS
        self.bw = bw #ADDED DAMPING COEFF
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = 1/(TPF*FPS)

        self.phi = 0
        self.dphi = 0
        self.ddphi = 0

        self.integral = 0
        self.maxtorque = maxtorque
        self.maxspeed = 60

        # Slew rate additions
        # self.actualtorque = 0
        # self.maxslew = 100.0  # N·m/s — how fast torque can ramp, tune this

    def wrap_angle(self, angle):
        return (angle + np.pi) % (2*np.pi) - np.pi

    def balance(self, pendulum):
        error = self.wrap_angle(pendulum.theta)
        self.integral += error * self.dt
        torque = (self.kp * error + self.kd * pendulum.dtheta + self.ki * self.integral)
        torque = np.clip(torque, -self.maxtorque, self.maxtorque)

        self.ddphi = (torque - (self.bw * self.dphi)) / self.I
        self.dphi += self.ddphi * self.dt
        self.dphi = np.clip(self.dphi, -self.maxspeed, self.maxspeed)
        self.phi += self.dphi * self.dt

        pendulum.tm = torque  

        return torque


# PARAMETERS TO CHANGE
pend_height = 0.10 #0.115
pend_inertia = 0.00234
pend_theta_0 = 0.3
pend_initial_v = 0
pend_mass = 0.17
pend_damping = 0
wheel_inertia = 0.00016
wheel_kp = 20
wheel_ki = 0.2
wheel_kd = 0.2
wheel_maxtorque = 0.05
wheel_damping = 0
pend = Pendulum(pend_height, pend_inertia, pend_theta_0, pend_initial_v, pend_mass, pend_damping)
wheel = Reaction(wheel_inertia, wheel_kp, wheel_ki, wheel_kd, wheel_maxtorque, wheel_damping)
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
    pend.draw(wheel)

    # Plot wrapped angle (continuous error signal), velocity, acceleration
    wrapped_theta = (pend.theta + np.pi) % (2 * np.pi) - np.pi
    plotter.push('θ (wrapped)', count, wrapped_theta)
    plotter.push('dθ/dt', count, pend.dtheta)
    plotter.push('d²θ/dt²', count, pend.ddtheta)
    plotter.push('Torque', count, pend.tm)
    plotter.push('ddphi', count, wheel.ddphi)

    if not count % TPF:
        pygame.display.update()
        plotter._update()         # manually tick the plot
        plotter._app.processEvents()  # manually tick Qt event loop

