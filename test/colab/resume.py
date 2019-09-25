import pyautogui
import time
from datetime import datetime
import pytz
pyautogui.FAILSAFE = False

RUN_BUTTON = 'run.png'
def get_time():
	return datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')).strftime('%Y-%m-%d - %I:%M:%S %p')

print('Started at', get_time())

while True:
	pyautogui.click((1000, 150))
	pos = pyautogui.locateOnScreen(RUN_BUTTON)
	if pos:
		pos = pyautogui.center(pos)
		print('Cell has stopped!', get_time())
		pyautogui.moveTo(*pos, .25, pyautogui.easeOutQuad)
		time.sleep(.2)
		pyautogui.click()
	else:
		print('Waiting for cell to stop...', end='\r')
	time.sleep(23)