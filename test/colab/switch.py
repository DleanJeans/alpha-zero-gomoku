import pyautogui
import time

FIRST_TAB_POS = [200, 10]
TAB_SPACING = 200
TAB_COUNT = 2
WAIT_TIME = 15

i = 0
while True:
	time.sleep(WAIT_TIME)

	tab_index = i % TAB_COUNT
	pos = FIRST_TAB_POS[::]
	pos[0] += tab_index * TAB_SPACING
	pyautogui.leftClick(*pos)

	i += 1