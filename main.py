import pygame
import matplotlib.pyplot as plt
import numpy as np
import time
from gpt import RealTimePlotAPI
import threading

WIDTH, HEIGHT = 800, 600
TPF, FPS = 5, 60 # Ticks per frame, frames per second
temp_t = time.time()

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()


def is_positive(n: int):
    return 1 if n >=0 else -1


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
        self.ddtheta = self.m*self.g*self.l * np.sin(self.theta) / self.I
        self.ddtheta += wheel.balance(self) / (wheel.I + wheel.m * self.l ** 2)
        # dt = time.time() - tt
        self.dtheta += self.ddtheta * (1/(TPF*FPS))
        self.theta += self.dtheta * (1/(TPF*FPS))
        # print(self.theta, self.dtheta)
        return time.time()
   
    def draw(self):
        screen.fill((255,255,255))
        pygame.draw.circle(screen, (0,0,0), (400 + 150*np.sin(self.theta), 300 - 150*np.cos(self.theta)), 10)
        pygame.draw.line(screen, (100, 100, 100), (400, 300), (400 + 150*np.sin(self.theta), 300 - 150*np.cos(self.theta)))




class Reaction:
    def __init__(self, m, r):
        self.m = m
        self.r = r
        self.I = self.m * self.r**2

        self.kp, self.ki, self.kd = 0.3, 0, 0
        self.dt = 1/(TPF*FPS)

        self.phi = 0
        self.dphi = 0
        self.ddphi = 0

        self.integral = 0
        self.torque = 0

        self.maxtorque = 15*self.I


    def balance(self, pendulum): # Actual stabilisation code goes here, can call on the ai class

        # corrected_theta = pendulum.theta % (2*np.pi) * is_positive(pendulum.theta)
        corrected_theta = pendulum.theta
        self.integral += self.dt * corrected_theta
        self.ddphi = - (corrected_theta * self.kp + pendulum.dtheta*self.kd + self.integral*self.ki)
        print((self.ddphi + pendulum.ddtheta) * self.I )
        return (self.ddphi + pendulum.ddtheta) * self.I
               
       









def main(plotter: RealTimePlotAPI):
   
    pend = Pendulum(1, 1, 0.1, 0)
    wheel = Reaction(1, 1)
    count = 0

    while True:
        count += 1
        clock.tick(TPF*FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()

           
            keys = pygame.key.get_pressed()
            if event.type == pygame.KEYDOWN:
                if keys[pygame.K_RIGHT]:
                    pend.dtheta += 10
                elif keys[pygame.K_LEFT]:
                    pend.dtheta -= 10


        pend.apply_forces(temp_t, wheel)
        pend.draw()
        
        plotter.push('s', count, pend.theta)
        plotter.push('v', count, pend.dtheta)
        plotter.push('a', count, pend.ddtheta)

        if not count % TPF:
            pygame.display.update()


plotter = RealTimePlotAPI(
    title='window',
    target_fps=10,
    x_window=400,
    line_width=2,
    y_range=[0, 6.3]
)

th = threading.Thread(target=main, args=(plotter,), daemon=True)
th.start()
plotter.start()