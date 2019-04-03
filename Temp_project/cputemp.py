from time import sleep, strftime, time
import matplotlib.pyplot as plt
from gpiozero import CPUTemperature



plt.ion()
x = []
y = []
cpu = CPUTemperature()
with open("/home/pi/cpu_temp.csv", "a") as log:
    while True:
        temp = cpu.temperature
	y.append(temp)
	x.append(time())
	plt.clf()
	plt.scatter(x,y)
	plt.plot(x,y)
	plt.draw()
	print temp
        log.write("{0},{1}\n".format(strftime("%Y-%m-%d %H:%M:%S"),str(temp)))
	sleep(1)
