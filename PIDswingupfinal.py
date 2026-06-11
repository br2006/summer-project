import pygame
import numpy as np
import time
import sys
from gpt import RealTimePlotAPI
from PyQt6 import QtWidgets
import pyqtgraph as pg

WIDTH, HEIGHT = 800, 600
TPF, FPS = 1, 120
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
        self.bp = bp
        self.theta = theta_0   # 0 = upright (top), pi = hanging (bottom)
        self.dtheta = v
        self.ddtheta = 0
        self.tm = 0

    def currentEnergy(self):
        """
        Total mechanical energy with theta=0 at top convention.
        PE = -mgl*cos(theta)
          theta=0  (top):    PE = -mgl        (minimum, reference)
          theta=pi (bottom): PE = +mgl        (maximum)
        E* = -mgl  (upright, stationary)
        """
        KE = 0.5 * self.I * self.dtheta ** 2
        PE = -self.m * self.g * self.l * np.cos(self.theta)
        return KE + PE

    def targetEnergy(self):
        """Target energy: sitting still at the top."""
        return -self.m * self.g * self.l

    def applyForces(self, wheel):
        torque_gravity = self.m * self.g * self.l * np.sin(self.theta)
        torque_damping = -self.bp * self.dtheta
        torque_control = -wheel.update(self)
        self.ddtheta = (torque_gravity + torque_damping + torque_control) / self.I
        dt = 1.0 / (TPF * FPS)
        self.dtheta += self.ddtheta * dt
        self.theta  += self.dtheta  * dt

    def draw(self, wheel):
        screen.fill((255, 255, 255))
        pivot_x, pivot_y = 400, 300
        bob_x = int(pivot_x + 110 * np.sin(self.theta))
        bob_y = int(pivot_y - 110 * np.cos(self.theta))
        # Upright target marker
        pygame.draw.line(screen, (220, 220, 220),
                         (pivot_x, pivot_y), (pivot_x, pivot_y - 120), 1)
        pygame.draw.line(screen, (80, 80, 80),
                         (pivot_x, pivot_y), (bob_x, bob_y), 4)
        pygame.draw.circle(screen, (60, 60, 60), (pivot_x, pivot_y), 6)
        self.drawFlywheel(bob_x, bob_y, wheel.phi)
        self.drawHUD(wheel)

    def drawFlywheel(self, cx, cy, phi):
        R, num_spokes = 75, 3
        pygame.draw.circle(screen, (30, 30, 30), (cx, cy), R, 3)
        pygame.draw.circle(screen, (80, 80, 80), (cx, cy), R - 6, 1)
        for i in range(num_spokes):
            a = phi + i * 2 * np.pi / num_spokes
            pygame.draw.line(screen, (50, 50, 50), (cx, cy),
                             (int(cx + (R-2)*np.cos(a)),
                              int(cy + (R-2)*np.sin(a))), 2)
        pygame.draw.circle(screen, (200, 50, 50), (cx, cy), 5)
        pygame.draw.circle(screen, (255, 140, 0),
                           (int(cx + (R-3)*np.cos(phi)),
                            int(cy + (R-3)*np.sin(phi))), 4)

    def drawHUD(self, wheel):
        font    = pygame.font.SysFont("monospace", 15)
        font_lg = pygame.font.SysFont("monospace", 20, bold=True)
        if wheel.mode == "BALANCE":
            mode_col, mode_text = (20, 160, 20), "MODE: BALANCE (PID)"
        else:
            mode_col, mode_text = (200, 100, 0), "MODE: SWING-UP (energy shaping)"
        screen.blit(font_lg.render(mode_text, True, mode_col), (10, 10))
        E, E_star = self.currentEnergy(), self.targetEnergy()
        for i, txt in enumerate([
            f"θ       : {self.theta:+.4f} rad  ({np.degrees(self.theta):+.1f}°)",
            f"dθ/dt   : {self.dtheta:+.4f} rad/s",
            f"E       : {E:.5f} J",
            f"E*      : {E_star:.5f} J",
            f"ΔE      : {E - E_star:+.5f} J",
            f"torque  : {self.tm:+.5f} N·m",
            f"wheel ω : {wheel.dphi:+.4f} rad/s",
        ]):
            screen.blit(font.render(txt, True, (60, 60, 60)), (10, 38 + i * 19))
        screen.blit(font.render("← → : impulse    R : reset    Q : quit",
                                True, (180, 180, 180)), (10, HEIGHT - 24))

class Reaction:
    def __init__(self, I, kp, ki, kd, maxtorque, bw, ke, switchThresholdDeg, fallbackThresholdDeg, maxSwitchVelocity):
        self.I  = I
        self.bw = bw
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = 1.0 / (TPF * FPS)

        self.phi   = 0.0
        self.dphi  = 0.0
        self.ddphi = 0.0

        self.integral  = 0.0
        self.maxtorque = maxtorque
        self.maxspeed  = 60.0

        self.ke = ke
        self.switchThreshold    = np.radians(switchThresholdDeg)
        self.fallbackThreshold  = np.radians(fallbackThresholdDeg)
        self.maxSwitchVelocity = maxSwitchVelocity
        self.mode = "SWINGUP"

    def wrapAngle(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def swingupTorque(self, pendulum):
        """
        u = ke * (E - E*) * sign(dtheta)

        delta_E = E - E*
          Hanging still at bottom: E = +mgl, E* = -mgl → delta_E = +2mgl  (way above target... wait)

        Hold on — we want to ADD energy when E < E* and REMOVE when E > E*.
        At the bottom hanging still:
          E = 0 + (-mgl*cos(pi)) = +mgl
          E* = -mgl
          delta_E = mgl - (-mgl) = +2mgl  → positive → torque opposes dtheta → BRAKES

        That's wrong — at the bottom we need to PUMP not brake.

        The fix: the sign of the control must be:
          torque = -ke * (E - E*) * sign(dtheta)
                 = ke * (E* - E) * sign(dtheta)

        At bottom: (E* - E) = -2mgl < 0 → torque opposes dtheta → wait, still wrong.

        Let's think from scratch with concrete numbers:
          Bottom, swinging right (dtheta > 0), E < E* needed... but E > E* here.

        Actually with this PE convention E is ALWAYS >= E* (since KE>=0 and
        PE >= -mgl = E*). So delta_E = E - E* >= 0 always.
        The controller should always be either pumping (when delta_E is small,
        i.e. close to E*) ... no, E* is the MINIMUM energy (top, stationary).

        The pendulum starts at the bottom with E = +mgl (high energy in this convention),
        and E* = -mgl. So delta_E = 2mgl >> 0 always. This convention doesn't work
        for swing-up from the bottom because the bottom has MORE energy than the top
        in a gravity potential sense — the pendulum is stable at the bottom.

        -----------------------------------------------------------------------
        CORRECT APPROACH for upright-zero convention:
        -----------------------------------------------------------------------
        Reframe: use energy relative to the BOTTOM (not absolute PE).
        Define:
          E_rel = KE + mgl*(1 - cos(theta))
            = 0 at top (theta=0): KE=0, PE_rel = mgl*(1-1) = 0  ← WRONG, top should be 2mgl

        Actually the standard energy shaping for inverted pendulum uses:
          E = 0.5*I*dtheta^2 + mgl*cos(theta)    [note: +cos not -cos]
          E* = mgl                                 [value at top, theta=0]
          At bottom (theta=pi): E = 0 - mgl = -mgl (lowest energy)

        Then delta_E = E - E*:
          At bottom stationary: delta_E = -mgl - mgl = -2mgl  < 0  → pump ✓
          At top stationary:    delta_E = mgl - mgl  = 0             → no torque ✓
          Spinning fast:        delta_E > 0                          → brake ✓

        Control: u = ke * delta_E * sign(dtheta)
          delta_E < 0, dtheta > 0 → u < 0 (wheel torque negative)
          Reaction on pendulum = -u > 0, aligned with dtheta → PUMPS ✓
        """
        E = 0.5 * pendulum.I * pendulum.dtheta**2 + pendulum.m * pendulum.g * pendulum.l * np.cos(pendulum.theta)
        E_star  = pendulum.m * pendulum.g * pendulum.l   # value at top
        delta_E = E - E_star
        torque  = self.ke * delta_E * np.sign(pendulum.dtheta)
        return float(np.clip(torque, -self.maxtorque, self.maxtorque))

    def balanceTorque(self, pendulum):
        error = self.wrapAngle(pendulum.theta)
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.maxtorque / max(self.ki, 1e-9), self.maxtorque / max(self.ki, 1e-9))
        torque = (self.kp * error + self.kd * pendulum.dtheta + self.ki * self.integral)
        return float(np.clip(torque, -self.maxtorque, self.maxtorque))

    def update(self, pendulum):
        angle_from_upright = abs(self.wrapAngle(pendulum.theta))
        speed = abs(pendulum.dtheta)

        if self.mode == "SWINGUP":
            if (angle_from_upright < self.switchThreshold
                    and speed < self.maxSwitchVelocity):
                self.mode = "BALANCE"
                self.integral = 0.0

        elif self.mode == "BALANCE":
            if angle_from_upright > self.fallbackThreshold:
                self.mode = "SWINGUP"
                self.integral = 0.0

        torque = (self.balanceTorque(pendulum) if self.mode == "BALANCE"
                  else self.swingupTorque(pendulum))

        self.ddphi  = (torque - self.bw * self.dphi) / self.I
        self.dphi  += self.ddphi * self.dt
        self.dphi   = np.clip(self.dphi, -self.maxspeed, self.maxspeed)
        self.phi   += self.dphi * self.dt

        pendulum.tm = torque
        return torque

    def reset(self):
        self.phi = self.dphi = self.ddphi = self.integral = 0.0
        self.mode = "SWINGUP"


# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------
pend_height    = 0.10
pend_inertia   = 0.00234
pend_theta_0   = np.pi - 0.05   # start near bottom (pi = hanging)
pend_initial_v = 0.0
pend_mass      = 0.17
pend_damping   = 0.001

wheel_inertia   = 0.00016
wheel_kp        = 20.0
wheel_ki        = 0.2
wheel_kd        = 0.2
wheel_maxtorque = 0.1
wheel_damping   = 0.0

ke_swingup             = 10.0
switchThresholdDeg   = 20.0
maxSwitchVelocity    = 2.0
fallbackThresholdDeg = 40.0

pend  = Pendulum(pend_height, pend_inertia, pend_theta_0,
                 pend_initial_v, pend_mass, pend_damping)
wheel = Reaction(wheel_inertia, wheel_kp, wheel_ki, wheel_kd, wheel_maxtorque,
                 wheel_damping, ke_swingup, switchThresholdDeg,
                 fallbackThresholdDeg, maxSwitchVelocity)
count = 0

# --- pyqtgraph setup ---
plotter = RealTimePlotAPI(title='Pendulum State', target_fps=10,
                          x_window=400, line_width=2, y_range=[-np.pi, np.pi])
plotter._running = True
plotter._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
plotter._win  = pg.GraphicsLayoutWidget(show=True, title=plotter.title)
plotter._win.resize(*plotter.window_size)
plotter._plot = plotter._win.addPlot(title=plotter.title)
plotter._plot.showGrid(x=True, y=True)
plotter._plot.addLegend()
if plotter.y_range:
    plotter._plot.setYRange(*plotter.y_range)

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
while True:
    count += 1
    clock.tick(TPF * FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()
        if event.type == pygame.KEYDOWN:
            keys = pygame.key.get_pressed()
            if   keys[pygame.K_RIGHT]: pend.dtheta += 10
            elif keys[pygame.K_LEFT]:  pend.dtheta -= 10
            elif keys[pygame.K_r]:
                pend.theta = np.pi - 0.05
                pend.dtheta = pend.ddtheta = 0.0
                wheel.reset()
            elif keys[pygame.K_q]:
                pygame.quit(); sys.exit()

    pend.applyForces(wheel)
    pend.draw(wheel)

    wrapped_theta = wheel.wrapAngle(pend.theta)
    plotter.push('θ (from upright)', count, wrapped_theta)
    plotter.push('dθ/dt',            count, pend.dtheta)
    plotter.push('ΔE',               count,
                 0.5*pend.I*pend.dtheta**2 + pend.m*pend.g*pend.l*np.cos(pend.theta)
                 - pend.m*pend.g*pend.l)
    plotter.push('Torque',           count, pend.tm)
    plotter.push('wheel ω',          count, wheel.dphi)

    if not count % TPF:
        pygame.display.update()
        plotter._update()
        plotter._app.processEvents()