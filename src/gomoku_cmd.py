# -*- coding: utf-8 -*-
import os
import numpy as np
from string import ascii_uppercase
from datetime import datetime
import pytz
import sys
import time

import psutil
import humanize
import os
import GPUtil as GPU

from termcolor import colored

colors = {
    'x': 'green',
    'o': 'red',
    '_': None
}

class GomokuCMD():
    def __init__(self, n, human_color=1):
        self.n = n
        self.human_color = human_color
        self.last_move = (-1, -1)

        self.reset_status()

        self.timezone = 'Asia/Ho_Chi_Minh'
        self.show_ram = False
        self.use_gpu = True

        self.iteration = -1
        self.contest = False

    def __del__(self):
        # close window
        self.is_running = False

    def reset_status(self):
        self.top_choices = {}
        self.chosen = -1
        self.last_time = None

        has_board = hasattr(self, 'board')

        self.board = np.zeros((self.n, self.n), dtype=int)
        self.number = np.zeros((self.n, self.n), dtype=int)

        self.is_human = False
        self.human_move = -1

        self.k = 0
        if has_board:
            self.print_board()
        self.k = 1

    def set_is_human(self, value=True):
        self.is_human = value

    def get_is_human(self):
        return self.is_human

    def get_human_move(self):
        return self.human_move

    def get_human_color(self):
        return self.human_color

    def execute_move(self, color, move):
        x, y = move // self.n, move % self.n
        assert self.board[x][y] == 0

        self.board[x][y] = color
        self.number[x][y] = self.k
        
        self.last_move = (x, y)
        self.print_ram()
        self.print_board()

        self.k += 1
    
    def loop(self):
        self.is_running = True

        while self.is_running:
            if self.is_human:
                alphanum = input('Move: ')
                self.human_move = self.alphanum_to_number(alphanum)
                self.execute_move(self.human_color, self.human_move)
                self.set_is_human(False)
    
    def alphanum_to_number(self, alphanum):
        y = ascii_uppercase.find(alphanum[:1].upper())
        x = int(alphanum[1:])

        assert 0 <= x < self.n and 0 <= y < self.n, 'Move Out of Board'
        return y * self.n + x
    
    def number_to_alphanum(self, number):
        x = number // self.n
        y = number - self.n * x
        x = ascii_uppercase[x]

        return f'{x}{y}'
    
    def print_ram(self):
        if not self.show_ram: return
        
        if self.use_gpu:
            gpu = GPU.getGPUs()[0]
        ram = psutil.virtual_memory()

        ram_free = ram.total - ram.used
        ram_percent = ram.used / ram.total * 100
        
        print(f'CPU: {psutil.cpu_percent()}% | RAM Free:', humanize.naturalsize(ram_free),'| Used:', humanize.naturalsize(ram.used), '({0:.1f}%)'.format(ram_percent), '| Total:', humanize.naturalsize(ram.total))

        if self.use_gpu:
            print('GPU: {0:.1f}%'.format(gpu.load*100) ,'| RAM Free: {0:.1f} GB | Used: {1:.1f} GB ({2:.1f}%) | Total {3:.1f} GB'.format(gpu.memoryFree/1024, gpu.memoryUsed/1024, gpu.memoryUtil*100, gpu.memoryTotal/1024))
        
        print('')

    def set_top_choices(self, probs, action):
        top_actions = np.argsort(-probs)[:3]
        if action not in top_actions:
            top_actions = np.concatenate((top_actions, [action]))

        self.top_choices = { action:probs[action] for action in top_actions }
        self.chosen = np.where(top_actions==action)[0][0]

    def print_top_choices(self):
        if not self.top_choices: return
        output = 'CHOICES :: '

        for i, (action, policy) in enumerate(self.top_choices.items()):
            alphanum = self.number_to_alphanum(action)
            policy *= 100

            choice = f'{alphanum}={policy:.2f}%'
            prefix = f'#{i+1}' if i < 3 else 'Chosen'
            choice = f'{prefix} {choice}'
            if i == self.chosen:
                choice = colored(choice, 'yellow')

            output += choice
            if i+1 < len(self.top_choices):
                output += ' | '

        print(output)

    def get_time(self):
        return datetime.now(pytz.timezone(self.timezone)).strftime('%Y-%m-%d - %I:%M:%S %p')
    
    def print_board(self):
        taken = -1
        now = time.time()
        if self.last_time:
            taken = now - self.last_time
        taken = f' - TAKEN {taken:.2f} SEC' if taken >= 0 else ''
        self.last_time = now

        timestamp = self.get_time()
        contest = ' (CONTEST)' if self.contest else ''

        if self.iteration > -1:
            print(f'ITER {self.iteration} - ', end='')
        print(f'MOVE {self.k}: {timestamp}{taken}{contest}')

        self.print_top_choices()

        x_labels = '    ' + ' '.join(ascii_uppercase[0:self.n]) + '\n'
        
        board = x_labels

        for y in range(self.n):
            y_label = str(y).zfill(2) + ' '
            board += y_label
            for x in range(self.n):
                piece = self._get_piece(x, y)
                board += '|' + piece
            board += '| %s\n' % y_label
        
        board += x_labels
        print(board)
    
    def _get_piece(self, x, y):
        piece = self.board[x][y]
        piece = {
            1: 'x',
            -1: 'o',
            0: '_'
        }[piece]

        color = colors[piece]
        on_color = None

        if (x, y) == self.last_move:
            piece = piece.upper()
            color = 'yellow'
        piece = colored(piece, color, on_color)
        return piece
