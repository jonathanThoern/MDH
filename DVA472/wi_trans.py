import wiringpi as wp
import time

OUTPUT = 1
INPUT = 0
HIGH = 1
LOW = 0

wp.wiringPiSetupGpio()
wp.pinMode(1,0)

while True:
	value1 = wp.analogRead(25)
	time.sleep(0.1)
	value2 = wp.analogRead(25)
	speed = value2 
	
	print(value)
