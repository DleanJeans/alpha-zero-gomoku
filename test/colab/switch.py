import pyautogui
import time

FIRST_TAB_POS = [125, 20]
TAB_SPACING = 200
TAB_COUNT = 2
WAIT_TIME = 31

i = 0
while True:
	tab_index = i % TAB_COUNT
	time.sleep(WAIT_TIME)

	pos = FIRST_TAB_POS[::]
	pos[0] += tab_index * TAB_SPACING
	pyautogui.leftClick(*pos)

	i += 1