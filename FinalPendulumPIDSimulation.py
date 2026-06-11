import pygame
import numpy as np
import time
import sys
from gpt import RealTimePlotAPI
from PyQt6 import QtWidgets
import pyqtgraph as pg

WIDTH, HEIGHT = 800, 600
SCALE         = 3                    # ← change to 3 for even sharper output
RENDER_W      = WIDTH  * SCALE
RENDER_H      = HEIGHT * SCALE
TPF, FPS      = 1, 120

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
canvas = pygame.Surface((RENDER_W, RENDER_H))
clock  = pygame.time.Clock()

# ---------------------------------------------------------------------------
# Warm light palette
# ---------------------------------------------------------------------------
BG           = (250, 247, 242)
PANEL_BG     = (255, 252, 247)
PANEL_BORDER = (232, 220, 205)
ROD_COL      = ( 28,  18,  12)
PIVOT_COL    = ( 22,  14,   8)
BOB_RIM      = ( 35,  22,  14)
BOB_INNER    = ( 50,  32,  20)
SPOKE_COL    = ( 40,  26,  16)
HUB_COL      = ( 30,  18,  10)
TIP_COL      = (220, 155,  60)
TARGET_COL   = (180, 210, 160)
LABEL_COL    = (160, 130, 105)
VALUE_COL    = ( 70,  55,  45)
HINT_COL     = (195, 180, 165)
MODE_BAL     = ( 80, 170, 110)
MODE_BAL_BG  = (235, 248, 238)
MODE_SWG     = (210, 120,  55)
MODE_SWG_BG  = (253, 242, 228)
DIVIDER      = (225, 212, 198)
CW_ROD_COL   = ( 90,  60,  40)   # counterweight arm colour (slightly lighter)
CW_BOB_COL   = ( 60,  40,  25)   # counterweight bob colour


def S(n):
    """Scale a coordinate or size value by SCALE."""
    return int(n * SCALE)


def rounded_rect(surface, color, rect, radius, border_color=None, border_width=1):
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    if border_color:
        pygame.draw.rect(surface, border_color, rect,
                         width=border_width, border_radius=radius)


def draw_dashed_line(surface, color, p1, p2, dash=9, gap=6, width=1):
    dx, dy  = p2[0] - p1[0], p2[1] - p1[1]
    length  = np.hypot(dx, dy)
    if length == 0:
        return
    ux, uy  = dx / length, dy / length
    pos, on = 0.0, True
    while pos < length:
        seg = min(pos + (dash if on else gap), length)
        if on:
            a = (int(p1[0] + ux * pos), int(p1[1] + uy * pos))
            b = (int(p1[0] + ux * seg), int(p1[1] + uy * seg))
            pygame.draw.line(surface, color, a, b, width)
        pos += dash if on else gap
        on   = not on


class Pendulum:
    def __init__(self, l, I, theta_0, v, m, bp,
                 cw_arm_length, cw_arm_mass, cw_weight_mass):
        self.l  = l
        self.m  = m
        self.g  = 9.81
        self.bp = bp

        # Counterweight parameters
        self.cw_l  = cw_arm_length
        self.cw_ma = cw_arm_mass
        self.cw_mw = cw_weight_mass

        # Extra inertia: arm (rod about pivot) + tip weight
        I_cw = (1/3) * cw_arm_mass * cw_arm_length**2 \
             + cw_weight_mass * cw_arm_length**2
        self.I = I + I_cw

        # Net gravitational moment arm: positive = bob-side dominant
        self.net_gravity_coeff = (m * l
                                  - cw_weight_mass * cw_arm_length
                                  - cw_arm_mass   * (cw_arm_length / 2))

        self.theta   = theta_0
        self.dtheta  = v
        self.ddtheta = 0
        self.tm      = 0

    def currentEnergy(self):
        KE      = 0.5 * self.I * self.dtheta ** 2
        PE_main = -self.m * self.g * self.l * np.cos(self.theta)
        PE_cw   = +(self.cw_mw * self.cw_l + self.cw_ma * self.cw_l / 2) \
                   * self.g * np.cos(self.theta)
        return KE + PE_main + PE_cw

    def targetEnergy(self):
        # Energy at upright (θ = 0): bob at top, counterweight at bottom
        return ( self.m    * self.g * self.l
               - self.cw_mw * self.g * self.cw_l
               - self.cw_ma * self.g * (self.cw_l / 2))

    def applyForces(self, wheel):
        torque_gravity  = self.g * self.net_gravity_coeff * np.sin(self.theta)
        torque_damping  = -self.bp * self.dtheta

        sigma_ax = 1.6282
        sigma_az = 1.6938

        base_acc_x = np.random.normal(0, sigma_ax)
        base_acc_z = np.random.normal(0, sigma_az)

        torque_disturbance = (-(base_acc_x / self.l) * np.cos(self.theta)
                            + (base_acc_z / self.l) * np.sin(self.theta)) * self.I        

        sensor_noise = np.random.normal(0, 0)
        noisy_theta  = self.theta  + sensor_noise
        noisy_dtheta = self.dtheta + sensor_noise * 0.5

        torque_control = -wheel.updateNoisy(self, noisy_theta, noisy_dtheta)

        self.ddtheta = (torque_gravity + torque_damping + torque_control + torque_disturbance) / self.I
        dt = 1.0 / (TPF * FPS)
        self.dtheta += self.ddtheta * dt
        self.theta  += self.dtheta  * dt

    def draw(self, wheel):
        canvas.fill(BG)

        pivot = (S(460), S(340))
        px, py = pivot
        bob_x  = int(px + S(110) * np.sin(self.theta))
        bob_y  = int(py - S(110) * np.cos(self.theta))

        # Counterweight arm end (opposite direction)
        cw_scale = self.cw_l / self.l   # scale CW arm relative to main arm pixels
        cw_x = int(px - S(110) * cw_scale * np.sin(self.theta))
        cw_y = int(py + S(110) * cw_scale * np.cos(self.theta))

        # upright target dashed line
        draw_dashed_line(canvas, TARGET_COL,
                         (px, py), (px, py - S(125)),
                         dash=S(7), gap=S(5), width=S(1))

        # Counterweight arm rod
        pygame.draw.line(canvas, CW_ROD_COL, pivot, (cw_x, cw_y), S(5))

        # Counterweight bob (solid circle)
        pygame.draw.circle(canvas, CW_BOB_COL, (cw_x, cw_y), S(14))
        pygame.draw.circle(canvas, PANEL_BG,   (cw_x, cw_y), S(5))

        # Main rod
        pygame.draw.line(canvas, ROD_COL, pivot, (bob_x, bob_y), S(8))

        # Pivot
        pygame.draw.circle(canvas, PANEL_BG,  pivot, S(13))
        pygame.draw.circle(canvas, PIVOT_COL, pivot, S(10))
        pygame.draw.circle(canvas, PANEL_BG,  pivot, S(3))

        self.drawFlywheel(bob_x, bob_y, wheel.phi)
        self.drawHUD(wheel)

        pygame.transform.smoothscale(canvas, (WIDTH, HEIGHT), screen)

    def drawFlywheel(self, cx, cy, phi):
        R       = S(75)
        R_inner = R - S(10)
        steps   = 48

        diam   = R * 2 + S(4)
        wsurf  = pygame.Surface((diam, diam), pygame.SRCALPHA)
        wsurf.fill((0, 0, 0, 0))
        ox, oy = diam // 2, diam // 2

        pygame.draw.circle(wsurf, (*SPOKE_COL, 255), (ox, oy), R)
        pygame.draw.circle(wsurf, (0, 0, 0, 0),      (ox, oy), R_inner)
        pygame.draw.circle(wsurf, (*BOB_RIM, 255), (ox, oy), R, S(2))

        spoke_half_angle = np.pi / 4
        spoke_offsets    = [0, np.pi]


        for offset in spoke_offsets:
            points = [(ox, oy)]
            for step in range(steps + 1):
                a = phi + offset - spoke_half_angle + step * (2 * spoke_half_angle / steps)
                points.append((int(ox + R * np.cos(a)),
                               int(oy + R * np.sin(a))))
            pygame.draw.polygon(wsurf, (*BOB_RIM, 255), points)

        pygame.draw.circle(wsurf, (*PANEL_BG, 255), (ox, oy), S(9))
        pygame.draw.circle(wsurf, (*HUB_COL,  255), (ox, oy), S(6))
        pygame.draw.circle(wsurf, (*PANEL_BG, 255), (ox, oy), S(2))

        canvas.blit(wsurf, (cx - ox, cy - oy))

        tip = (int(cx + (R - S(5)) * np.cos(phi)),
               int(cy + (R - S(5)) * np.sin(phi)))
        pygame.draw.circle(canvas, PANEL_BG, tip, S(6))
        pygame.draw.circle(canvas, TIP_COL,  tip, S(4))

    def drawHUD(self, wheel):
        font_ui   = pygame.font.SysFont("segoeui", S(14))
        font_bold = pygame.font.SysFont("segoeui", S(14), bold=True)
        font_hint = pygame.font.SysFont("segoeui", S(13))

        if wheel.mode == "BALANCE":
            m_col, m_bg, m_txt = MODE_BAL, MODE_BAL_BG, "Balance  ·  PID"
        else:
            m_col, m_bg, m_txt = MODE_SWG, MODE_SWG_BG, "Swing-up  ·  Energy shaping"

        badge_surf = font_bold.render(m_txt, True, m_col)
        bw = badge_surf.get_width() + S(24)
        bh = badge_surf.get_height() + S(10)
        rounded_rect(canvas, m_bg,
                     (S(12), S(12), bw, bh), S(6),
                     border_color=m_col, border_width=S(1))
        canvas.blit(badge_surf, (S(12) + S(12), S(12) + S(5)))

        E, E_star = self.currentEnergy(), self.targetEnergy()
        rows = [
            ("θ",       f"{self.theta:+.4f} rad   {np.degrees(self.theta):+.1f}°"),
            ("dθ/dt",   f"{self.dtheta:+.4f} rad/s"),
            ("E",       f"{E:.5f} J"),
            ("E*",      f"{E_star:.5f} J"),
            ("ΔE",      f"{E - E_star:+.5f} J"),
            ("Torque",  f"{self.tm:+.5f} N·m"),
            ("Wheel ω", f"{wheel.dphi:+.4f} rad/s"),
        ]

        card_x  = S(12)
        card_y  = S(12) + bh + S(8)
        row_h   = S(26)
        lbl_w   = S(72)
        val_x   = card_x + lbl_w + S(8)
        card_w  = S(290)
        card_h  = len(rows) * row_h + S(12)
        pad_top = S(6)

        rounded_rect(canvas, PANEL_BG,
                     (card_x, card_y, card_w, card_h), S(8),
                     border_color=PANEL_BORDER, border_width=S(1))

        for i, (label, value) in enumerate(rows):
            y = card_y + pad_top + i * row_h
            if i > 0:
                pygame.draw.line(canvas, DIVIDER,
                                 (card_x + S(8),          y - S(1)),
                                 (card_x + card_w - S(8), y - S(1)), S(1))
            canvas.blit(font_ui.render(label, True, LABEL_COL),
                        (card_x + S(12), y + S(5)))
            canvas.blit(font_ui.render(value, True, VALUE_COL),
                        (val_x,          y + S(5)))

        keys = pygame.key.get_pressed()
        font_key = pygame.font.SysFont("segoeui", S(22), bold=True)
        font_hint2 = pygame.font.SysFont("segoeui", S(13))

        y_hint = S(HEIGHT) - S(60)

        for i, (label, key) in enumerate([("←", pygame.K_LEFT), ("→", pygame.K_RIGHT)]):
            pressed = keys[key]
            bg_col  = MODE_SWG_BG  if pressed else PANEL_BG
            rim_col = MODE_SWG      if pressed else PANEL_BORDER
            txt_col = MODE_SWG      if pressed else HINT_COL
            box_x   = S(12) + i * S(48)
            box_w, box_h = S(38), S(38)
            rounded_rect(canvas, bg_col,
                        (box_x, y_hint, box_w, box_h), S(6),
                        border_color=rim_col, border_width=S(2))
            sym = font_key.render(label, True, txt_col)
            canvas.blit(sym, (box_x + (box_w - sym.get_width()) // 2,
                            y_hint + (box_h - sym.get_height()) // 2))

        rest = font_hint2.render("Impulse      R  Reset      Q  Quit", True, HINT_COL)
        canvas.blit(rest, (S(12) + S(104), y_hint + S(12)))


class Reaction:
    def __init__(self, I, kp, ki, kd, maxtorque, bw, ke,
                 switchThresholdDeg, fallbackThresholdDeg, maxSwitchVelocity):
        self.I = I
        self.bw = bw
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = 1 / (TPF * FPS)

        self.phi = 0
        self.dphi = 0
        self.ddphi = 0

        self.integral = 0
        self.maxtorque = maxtorque
        self.maxspeed = 1000

        self.ke = ke
        self.switchThreshold = np.radians(switchThresholdDeg)
        self.fallbackThreshold = np.radians(fallbackThresholdDeg)
        self.maxSwitchVelocity = maxSwitchVelocity
        self.mode = "SWINGUP"

    def wrapAngle(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def swingupTorque(self, pendulum):
        E = (0.5 * pendulum.I * pendulum.dtheta**2
             + pendulum.m * pendulum.g * pendulum.l * np.cos(pendulum.theta)
             - (pendulum.cw_mw * pendulum.cw_l + pendulum.cw_ma * pendulum.cw_l / 2)
               * pendulum.g * np.cos(pendulum.theta))
        E_star = pendulum.targetEnergy()
        delta_E = E - E_star
        torque = self.ke * delta_E * np.sign(pendulum.dtheta)
        return float(np.clip(torque, -self.maxtorque, self.maxtorque))

    def balanceTorque(self, pendulum):
        error = self.wrapAngle(pendulum.theta)
        self.integral += error * self.dt
        self.integral  = np.clip(self.integral, -self.maxtorque / max(self.ki, 1e-9), self.maxtorque / max(self.ki, 1e-9))
        torque = (self.kp * error + self.kd * pendulum.dtheta + self.ki * self.integral)
        return float(np.clip(torque, -self.maxtorque, self.maxtorque))

    def update(self, pendulum):
        angle_from_upright = abs(self.wrapAngle(pendulum.theta))
        speed = abs(pendulum.dtheta)

        if self.mode == "SWINGUP":
            if (angle_from_upright < self.switchThreshold
                    and speed < self.maxSwitchVelocity):
                self.mode     = "BALANCE"
                self.integral = 0.0
        elif self.mode == "BALANCE":
            if angle_from_upright > self.fallbackThreshold:
                self.mode     = "SWINGUP"
                self.integral = 0.0

        torque = (self.balanceTorque(pendulum) if self.mode == "BALANCE"
                  else self.swingupTorque(pendulum))

        self.ddphi = (torque - self.bw * self.dphi) / self.I
        self.dphi += self.ddphi * self.dt
        self.dphi = np.clip(self.dphi, -self.maxspeed, self.maxspeed)
        self.phi += self.dphi * self.dt

        pendulum.tm = torque
        return torque

    def updateNoisy(self, pendulum, noisy_theta, noisy_dtheta):
        angle_from_upright = abs(self.wrapAngle(noisy_theta))
        speed = abs(noisy_dtheta)

        if self.mode == "SWINGUP":
            if (angle_from_upright < self.switchThreshold
                    and speed < self.maxSwitchVelocity):
                self.mode     = "BALANCE"
                self.integral = 0.0
        elif self.mode == "BALANCE":
            if angle_from_upright > self.fallbackThreshold:
                self.mode     = "SWINGUP"
                self.integral = 0.0

        torque = (self.balanceTorqueNoisy(pendulum, noisy_theta, noisy_dtheta)
                  if self.mode == "BALANCE"
                  else self.swingupTorque(pendulum))

        self.ddphi = (torque - self.bw * self.dphi) / self.I
        dphi_new = self.dphi + self.ddphi * self.dt

        if dphi_new > self.maxspeed:
            dphi_new = self.maxspeed
            torque = self.bw * self.dphi
        elif dphi_new < -self.maxspeed:
            dphi_new = -self.maxspeed
            torque = -self.bw * self.dphi

        self.dphi = dphi_new
        self.phi += self.dphi * self.dt

        pendulum.tm = torque
        return torque

    def balanceTorqueNoisy(self, pendulum, noisy_theta, noisy_dtheta):
        error = self.wrapAngle(noisy_theta)
        self.integral += error * self.dt
        self.integral  = np.clip(self.integral, -self.maxtorque / max(self.ki, 1e-9), self.maxtorque / max(self.ki, 1e-9))
        torque = (self.kp * error + self.kd * noisy_dtheta + self.ki * self.integral)
        return float(np.clip(torque, -self.maxtorque, self.maxtorque))

    def reset(self):
        self.phi = self.dphi = self.ddphi = self.integral = 0.0
        self.mode = "SWINGUP"


# ---------------------------------------------------------------------------
# PARAMETERS
# ---------------------------------------------------------------------------
pend_height = 0.10
pend_inertia = 0.00234
pend_theta_0 = np.pi
pend_initial_v = 0.0
pend_mass = 0.17
pend_damping = 0.001

#for the counter weight
cw_arm_length  = 0.10   
cw_arm_mass    = 0.010
cw_weight_mass = 0.100 

wheel_inertia  = 0.00023717
wheel_kp = 40
wheel_ki = 1
wheel_kd = 0.7
wheel_maxtorque = 0.15
wheel_damping  = 0.005

ke_swingup = 10
switchThresholdDeg = 25
maxSwitchVelocity = 4
fallbackThresholdDeg = 60

pend  = Pendulum(pend_height, pend_inertia, pend_theta_0,
                 pend_initial_v, pend_mass, pend_damping,
                 cw_arm_length, cw_arm_mass, cw_weight_mass)
wheel = Reaction(wheel_inertia, wheel_kp, wheel_ki, wheel_kd, wheel_maxtorque,
                 wheel_damping, ke_swingup, switchThresholdDeg,
                 fallbackThresholdDeg, maxSwitchVelocity)
count = 0

# --- pyqtgraph setup ---
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
win = pg.GraphicsLayoutWidget(show=True, title='Pendulum State')
win.resize(800, 600)

plot_angle = win.addPlot(title='Angle & Velocity')
plot_angle.showGrid(x=True, y=True)
plot_angle.addLegend()

win.nextRow()

plot_velocity = win.addPlot(title='Angular Velocity')
plot_velocity.showGrid(x=True, y=True)
plot_velocity.addLegend()

win.nextRow()

plot_torque = win.addPlot(title='Torque')
plot_torque.showGrid(x=True, y=True)
plot_torque.addLegend()


curves = {
    'theta':  plot_angle.plot(pen='y', name='θ (from upright)'),
    'torque': plot_torque.plot(pen='r', name='Torque'),
    'dtheta': plot_velocity.plot(pen='c', name='dθ/dt'),
}

data = {'theta': [], 'dtheta': [], 'torque': [], 'x': []}
X_WINDOW = 400

import csv
csv_file1 = open('theta_data.csv', 'w', newline='')
csv_writer1 = csv.writer(csv_file1)
csv_writer1.writerow(['time_s', 'theta_rad'])

csv_file2 = open('motorvelocity_data.csv', 'w', newline='')
csv_writer2 = csv.writer(csv_file2)
csv_writer2.writerow(['time_s', 'motorvelocity_rads'])

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
while True:
    count += 1
    clock.tick(TPF * FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            csv_file1.close()
            csv_file2.close()
            pygame.quit(); sys.exit()
        if event.type == pygame.KEYDOWN:
            keys = pygame.key.get_pressed()
            if   keys[pygame.K_RIGHT]: pend.dtheta += 5
            elif keys[pygame.K_LEFT]:  pend.dtheta -= 5
            elif keys[pygame.K_r]:
                pend.theta   = np.pi - 0.05
                pend.dtheta  = pend.ddtheta = 0.0
                wheel.reset()
            elif keys[pygame.K_q]:
                csv_file1.close()
                csv_file2.close()
                pygame.quit(); sys.exit()

    pend.applyForces(wheel)
    csv_writer1.writerow([count / (TPF * FPS), pend.theta])
    csv_writer2.writerow([count / (TPF * FPS), wheel.dphi])

    pend.draw(wheel)

    wrapped_theta = wheel.wrapAngle(pend.theta)

    data['x'].append(count)
    data['theta'].append(wrapped_theta)
    data['dtheta'].append(wheel.dphi)
    data['torque'].append(pend.tm)

    for key in data:
        if len(data[key]) > X_WINDOW:
            data[key] = data[key][-X_WINDOW:]

    if not count % TPF:
        pygame.display.update()
        curves['theta'].setData(data['x'][-X_WINDOW:],  data['theta'])
        curves['dtheta'].setData(data['x'][-X_WINDOW:], data['dtheta'])
        curves['torque'].setData(data['x'][-X_WINDOW:], data['torque'])
        app.processEvents()