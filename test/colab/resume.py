from imagesearch import *
import pyautogui
import time
from datetime import datetime
import pytz

RUN_BUTTON = 'run.png'
def get_time():
	return datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')).strftime('%Y-%m-%d - %I:%M:%S %p')

print('Started at', get_time())

while True:
	pos = imagesearch(RUN_BUTTON, .9)
	if pos[0] > -1:
		print('Cell has stopped!', get_time())
		click_image(RUN_BUTTON, pos, 'left', 0.25)
	else:
		print('Waiting for cell to stop...', end='\r')
	time.sleep(5)