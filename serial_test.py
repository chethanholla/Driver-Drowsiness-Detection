import serial
import time
data = serial.Serial(
                 'COM3',
                   baudrate = 9600,
                    parity=serial.PARITY_NONE,
                   stopbits=serial.STOPBITS_ONE,
                   bytesize=serial.EIGHTBITS,
 #                   )
                    timeout=1 # must use when using data.readline()
                    )



while True:

  data.write(str.encode('A'))
  print('sent')
  time.sleep(2)
