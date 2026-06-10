import pygame
import numpy as np
import time
import sys
from gpt import RealTimePlotAPI
from PyQt6 import QtWidgets
import pyqtgraph as pg

WIDTH, HEIGHT = 800, 600
SCALE         = 3                    # ← change to 3 for even sharper outputpip
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
ROD_COL      = ( 28,  18,  12)   # near-black dark brown rod
PIVOT_COL    = ( 22,  14,   8)   # near-black dark brown pivot
BOB_RIM      = ( 35,  22,  14)   # near-black dark brown wheel rim
BOB_INNER    = ( 50,  32,  20)   # slightly lighter inner ring
SPOKE_COL    = ( 40,  26,  16)   # near-black dark brown spokes
HUB_COL      = ( 30,  18,  10)   # near-black dark brown hub
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
    def __init__(self, l, I, theta_0, v, m, bp):
        self.l = l
        self.I = I
        self.m = m
        self.g = 9.81
        self.bp = bp
        self.theta = theta_0
        self.dtheta = v
        self.ddtheta = 0
        self.tm = 0

    def currentEnergy(self):
        KE = 0.5 * self.I * self.dtheta ** 2
        PE = -self.m * self.g * self.l * np.cos(self.theta)
        return KE + PE

    def targetEnergy(self):
        return -self.m * self.g * self.l

    def applyForces(self, wheel):
        torque_gravity = self.m * self.g * self.l * np.sin(self.theta)
        torque_damping = -self.bp * self.dtheta

        sensor_noise = np.random.normal(0, 0)
        disturbance = np.random.normal(0, 0)  # random disturbances

        noisy_theta = self.theta  + sensor_noise
        noisy_dtheta = self.dtheta + sensor_noise * 0.5

        torque_control  = -wheel.update_noisy(self, noisy_theta, noisy_dtheta)
        #torque_control += actuator_noise

        self.ddtheta = (torque_gravity + torque_damping + torque_control + disturbance) / self.I
        dt = 1.0 / (TPF * FPS)
        self.dtheta += self.ddtheta * dt
        self.theta  += self.dtheta  * dt

    def draw(self, wheel):
        canvas.fill(BG)

        pivot = (S(460), S(340))
        px, py = pivot
        bob_x  = int(px + S(110) * np.sin(self.theta))
        bob_y  = int(py - S(110) * np.cos(self.theta))

        # upright target dashed line
        draw_dashed_line(canvas, TARGET_COL,
                         (px, py), (px, py - S(125)),
                         dash=S(7), gap=S(5), width=S(1))

        # rod
        pygame.draw.line(canvas, ROD_COL, pivot, (bob_x, bob_y), S(8))

        # pivot
        pygame.draw.circle(canvas, PANEL_BG,  pivot, S(13))
        pygame.draw.circle(canvas, PIVOT_COL, pivot, S(10))
        pygame.draw.circle(canvas, PANEL_BG,  pivot, S(3))

        self.drawFlywheel(bob_x, bob_y, wheel.phi)
        self.drawHUD(wheel)

        # ── scale canvas → window (this is what gives the AA / sharpness) ──
        pygame.transform.smoothscale(canvas, (WIDTH, HEIGHT), screen)

    def drawFlywheel(self, cx, cy, phi):
        R       = S(75)
        R_inner = R - S(10)
        steps   = 48

        # ── create a transparent surface the size of the wheel ───────────────
        diam   = R * 2 + S(4)
        wsurf  = pygame.Surface((diam, diam), pygame.SRCALPHA)
        wsurf.fill((0, 0, 0, 0))          # fully transparent
        ox, oy = diam // 2, diam // 2     # local centre on wsurf

        # ── annular ring (between R_inner and R) ─────────────────────────────
        pygame.draw.circle(wsurf, (*SPOKE_COL, 255), (ox, oy), R)
        pygame.draw.circle(wsurf, (0, 0, 0, 0),      (ox, oy), R_inner)

        # ── outer rim ────────────────────────────────────────────────────────
        pygame.draw.circle(wsurf, (*BOB_RIM, 255), (ox, oy), R,   S(2))

        # ── thick spokes (two, each sweeping 90°) ────────────────────────────
        spoke_half_angle = np.pi / 4
        spoke_offsets    = [0, np.pi]

        for offset in spoke_offsets:
            points = [(ox, oy)]
            for step in range(steps + 1):
                a = phi + offset - spoke_half_angle + step * (2 * spoke_half_angle / steps)
                points.append((int(ox + R * np.cos(a)),
                            int(oy + R * np.sin(a))))
            pygame.draw.polygon(wsurf, (*BOB_RIM, 255), points)

        # ── hub ──────────────────────────────────────────────────────────────
        pygame.draw.circle(wsurf, (*PANEL_BG, 255), (ox, oy), S(9))
        pygame.draw.circle(wsurf, (*HUB_COL,  255), (ox, oy), S(6))
        pygame.draw.circle(wsurf, (*PANEL_BG, 255), (ox, oy), S(2))

        # ── blit wheel surface onto canvas ───────────────────────────────────
        canvas.blit(wsurf, (cx - ox, cy - oy))

        # ── tip marker (drawn directly on canvas, outside wsurf) ─────────────
        tip = (int(cx + (R - S(5)) * np.cos(phi)),
            int(cy + (R - S(5)) * np.sin(phi)))
        pygame.draw.circle(canvas, PANEL_BG, tip, S(6))
        pygame.draw.circle(canvas, TIP_COL,  tip, S(4))

    def drawHUD(self, wheel):
        # fonts are sized in final screen pixels so we scale them up too
        font_ui   = pygame.font.SysFont("segoeui", S(14))
        font_bold = pygame.font.SysFont("segoeui", S(14), bold=True)
        font_hint = pygame.font.SysFont("segoeui", S(13))

        # mode badge
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

        # telemetry card
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

        # hint bar
        hint = font_hint.render(
            "← →  Impulse      R  Reset      Q  Quit", True, HINT_COL)
        canvas.blit(hint, (S(12), S(HEIGHT) - S(22)))


class Reaction:
    def __init__(self, I, kp, ki, kd, maxtorque, bw, ke,
                 switchThresholdDeg, fallbackThresholdDeg, maxSwitchVelocity):
        self.I  = I
        self.bw = bw
        self.kp, self.ki, self.kd = kp, ki, kd
        self.dt = 1 / (TPF * FPS)

        self.phi   = 0
        self.dphi  = 0
        self.ddphi = 0

        self.integral  = 0
        self.maxtorque = maxtorque
        self.maxspeed  = 60

        self.ke = ke
        self.switchThreshold   = np.radians(switchThresholdDeg)
        self.fallbackThreshold = np.radians(fallbackThresholdDeg)
        self.maxSwitchVelocity = maxSwitchVelocity
        self.mode = "SWINGUP"

    def wrapAngle(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def swingupTorque(self, pendulum):
        E = (0.5 * pendulum.I * pendulum.dtheta**2 + pendulum.m * pendulum.g * pendulum.l * np.cos(pendulum.theta))
        E_star  = pendulum.m * pendulum.g * pendulum.l
        delta_E = E - E_star
        torque  = self.ke * delta_E * np.sign(pendulum.dtheta)
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

    #def update_noisy(self, pendulum, noisy_theta, noisy_dtheta):
        angle_from_upright = abs(self.wrapAngle(noisy_theta))
        speed              = abs(noisy_dtheta)

        if self.mode == "SWINGUP":
            if (angle_from_upright < self.switchThreshold
                    and speed < self.maxSwitchVelocity):
                self.mode     = "BALANCE"
                self.integral = 0.0
        elif self.mode == "BALANCE":
            if angle_from_upright > self.fallbackThreshold:
                self.mode     = "SWINGUP"
                self.integral = 0.0

        #torque = (self.balanceTorque_noisy(pendulum, noisy_theta, noisy_dtheta)
        #        if self.mode == "BALANCE"
        #        else self.swingupTorque(pendulum))

        #self.ddphi = (torque - self.bw * self.dphi) / self.I
        #self.dphi += self.ddphi * self.dt
        #self.dphi  = np.clip(self.dphi, -self.maxspeed, self.maxspeed)
        #self.phi  += self.dphi * self.dt

        pendulum.tm = torque
        return torque
    
    def update_noisy(self, pendulum, noisy_theta, noisy_dtheta):
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

        torque = (self.balanceTorque_noisy(pendulum, noisy_theta, noisy_dtheta)
                if self.mode == "BALANCE"
                else self.swingupTorque(pendulum))

        self.ddphi = (torque - self.bw * self.dphi) / self.I
        dphi_new = self.dphi + self.ddphi * self.dt

        # Saturation, if wheel is at maxspeed and torque pushes further it can't accelerate so no reaction torque delivered
        if dphi_new > self.maxspeed:
            dphi_new = self.maxspeed
            # Only the braking/friction component remains
            torque = self.bw * self.dphi   # reaction torque collapses to drag only
        elif dphi_new < -self.maxspeed:
            dphi_new = -self.maxspeed
            torque = -self.bw * self.dphi

        self.dphi = dphi_new
        self.phi += self.dphi * self.dt

        pendulum.tm = torque
        return torque   # reflects what the wheel actually delivered

    def balanceTorque_noisy(self, pendulum, noisy_theta, noisy_dtheta):
        error= self.wrapAngle(noisy_theta)
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

wheel_inertia = 0.00023717
wheel_kp = 70
wheel_ki = 1
wheel_kd = 0
wheel_maxtorque = 0.03
wheel_damping = 0.005

ke_swingup = 10
switchThresholdDeg = 25
maxSwitchVelocity = 4
fallbackThresholdDeg = 60

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
            if   keys[pygame.K_RIGHT]: pend.dtheta += 5
            elif keys[pygame.K_LEFT]:  pend.dtheta -= 5
            elif keys[pygame.K_r]:
                pend.theta   = np.pi - 0.05
                pend.dtheta  = pend.ddtheta = 0.0
                wheel.reset()
            elif keys[pygame.K_q]:
                pygame.quit(); sys.exit()

    pend.applyForces(wheel)
    pend.draw(wheel)

    wrapped_theta = wheel.wrapAngle(pend.theta)
    plotter.push('θ (from upright)', count, wrapped_theta)
    plotter.push('dθ/dt',            count, pend.dtheta)
    plotter.push('ΔE',               count,
                 0.5*pend.I*pend.dtheta**2
                 + pend.m*pend.g*pend.l*np.cos(pend.theta)
                 - pend.m*pend.g*pend.l)
    plotter.push('Torque',           count, pend.tm)
    plotter.push('wheel ω',          count, wheel.dphi)

    if not count % TPF:
        pygame.display.update()
        plotter._update()
        plotter._app.processEvents()