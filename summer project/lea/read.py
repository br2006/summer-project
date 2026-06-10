import serial
from datetime import datetime
import time
from gpt import RealTimePlotAPI
import threading



sensor = "yeah"
serial_port = 'COM5'
baud_rate = 115200
path = "%s_LOG_%s.csv" % (str(datetime.now()), sensor)
ser = serial.Serial(serial_port, baud_rate)
serMega = serial.Serial('COM6', baud_rate)

starttime = time.time()

def main_theta(plotter: RealTimePlotAPI):
    with open('path.txt', 'w+') as f:
        f.write('-------------------------------------\n')
        f.write('-------------------------------------\n')
        f.write('-------------------------------------\n')
        f.write('-------------------------------------\n')
        while True:
            try:
                theta = ser.readline().decode("utf-8")
                # f.writelines(["%s," % (time.time() - starttime)] + line.strip() + "\n")
                print(float(theta.strip()[1:-1]))
                plotter.push('theta', time.time()-starttime , float(theta.strip()[1:-1]) * 180/3.141592654)
            except:
                pass


def main_motor(plotter: RealTimePlotAPI):
    with open('path.txt', 'w+') as f:
        f.write('-------------------------------------\n')
        f.write('-------------------------------------\n')
        f.write('-------------------------------------\n')
        f.write('-------------------------------------\n')
        while True:
            try:
                plotter.push('motor velocity', time.time() - starttime, float(serMega.readline().decode("utf-8").strip()))
            except:
                pass
        
    
plotter = RealTimePlotAPI(
    title='Plot',
    target_fps=30,
    x_window=50,
    line_width=2,
    y_range=[0, 6.3]
)


th1 = threading.Thread(target=main_theta, args=(plotter,), daemon=True)
th1.start()
th2 = threading.Thread(target=main_motor, args=(plotter,), daemon=True)
th2.start()
csv_path = plotter.start_csv_logging(directory="data")
plotter.start()


