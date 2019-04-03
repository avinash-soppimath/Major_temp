#!/home/pi/miniconda3/bin/python
from datetime import datetime
import sys
import Adafruit_DHT
from time import sleep
# print 'Temp: {0:0.1f} C  Humidity: {1:0.1f} %'.format(temperature, humidity)

def main():
    sensor = 11
    pin = 4
    record_file = '/home/pi/dht.csv'  # TODO: change to your own path

    timestr = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    humidity = None
    temperature = None
    while True:
    	try:
            humidity, temperature = Adafruit_DHT.read_retry(
                sensor, pin, retries=5, delay_seconds=1)
    	except ValueError as e:
            print 'Encounter sensor problem'

    	with open(record_file, 'a') as f:
            f.write('{0},{1},{2}\n'.format(timestr, humidity, temperature,))
	    print 'Temp: {0:0.1f} C  Humidity: {1:0.1f} %'.format(temperature, humidity)
	    sleep(3)
	#print()
	#    sleep(1)


if __name__ == '__main__':
    main()
