import sys
import urllib2
from time import sleep
import Adafruit_DHT as dht
from datetime import datetime
import firebase


firebase =firebase.FirebaseApplication('https://raspberry-pi-c77c3.firebaseio.com/RPI')

myAPI = '5PQ14ZBU1FM414EP' 

baseURL = 'https://api.thingspeak.com/update?api_key=%s' % myAPI 

record_file = '/home/pi/dht.csv'
timestr = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')


def DHT11_data():
	try:

  		humi, temp = dht.read_retry(11, 4, retries=5, delay_seconds=1) 
		result = firebase.post('https://raspberry-pi-c77c3.firebaseio.com/RPI', {'Temperature':str(temp),'humidity':str(humi)})
	  	return humi, temp
  	except ValueError as e:
    		print 'Encounter sensor problem'

with open(record_file, 'a') as f:
	while True:
	#	try:
			humi, temp = DHT11_data()

			if isinstance(humi, float) and isinstance(temp, float):

				humi = '%.2f' % humi 					   
				temp = '%.2f' % temp
#				with open(record_file, 'a') as f:
       				f.write('{0},{1},{2}\n'.format(timestr, humi, temp,))
       				print 'Temp: {} C  Humidity: {} %'.format(temp, humi)
			
				# Sending to thingspeak
				conn = urllib2.urlopen(baseURL + '&field1=0&field2=%s&field3=%s' % (temp, humi))
				print 'Connection param : {} '.format(conn.read())
				# Close connection
				conn.close()
			else:
				print 'Error'
		
			sleep(2)
	#	except:
	#		print 'Error1'
