import pygame
import matplotlib.pyplot as plt
import numpy as np
import time
from gpt import RealTimePlotAPI
import threading

WIDTH, HEIGHT = 800, 600
TPF, FPS = 1, 120 # Ticks per frame, frames per second
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

        self.phi = 0
        self.dphi = 0
        self.ddphi = 0


    def balance(self, pendulum): # Actual stabilisation code goes here, can call on the ai class 

        corrected_theta = pendulum.theta % (2*np.pi)
        # if corrected_theta > np.pi or corrected_theta > -np.pi and corrected_theta < 0:
        #     return 10 * self.I 

        # else:
        #     return -10 * self.I             
        return 0









def main(plotter: RealTimePlotAPI):
    
    pend = Pendulum(1, 1, 0.3, 0)
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
