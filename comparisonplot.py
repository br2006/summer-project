import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import csv

#simtheta = pd.read_csv('theta_data.csv')
#åsimvel = pd.read_csv('motorvelocity_data.csv')
realtheta = pd.read_csv('stabilised_fixed2.csv')
realvel = pd.read_csv('stabilised_fixed3.csv')

plt.plot(realvel['time'], -realvel['motorvelocity'])
plt.plot(realtheta['times'], (realtheta['theta']-278.5262433))

#plt.xlabel('Time (s)')
#plt.ylabel('Theta (degrees)')
#plt.show()


# Smooth motor velocity with a moving average
window = 20  # adjust this value

smooth_vel = -realvel['motorvelocity'].rolling(
    window=window,
    center=True
).mean()

plt.plot(realvel['time'], -realvel['motorvelocity'], label=f'Motor velocity (rad/s)')
plt.plot(realtheta['times'], realtheta['theta'] - 278.5262433, label='Theta (degrees)')

plt.xlabel('Time (s)')
plt.xlim(left=0, right=60)
plt.legend()
plt.show()



#plt.plot(((simtheta['time_s']*6)-2.5), simtheta['theta_rad']-(np.pi*2))
#plt.plot(simvel['time_s']*6, (simvel['motorvelocity_rads']))

#plt.xlabel('Time (s)')
#plt.ylabel('Motor velocity (rad/s)')
#plt.show()