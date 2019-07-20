# -*- coding: utf-8 -*-
import os
import numpy as np
from string import ascii_uppercase
from datetime import datetime
from pytz import timezone

class GomokuCMD():
    def __init__(self, n, human_color=1):
        self.n = n
        self.human_color = human_color
        self.last_move = (-1, -1)

        self.reset_status()

        self.output_board = False

    def __del__(self):
        # close window
        self.is_running = False

    def reset_status(self):
        has_board = hasattr(self, 'board')

        self.board = np.zeros((self.n, self.n), dtype=int)
        self.number = np.zeros((self.n, self.n), dtype=int)
        self.k = 1 # step number

        self.is_human = False
        self.human_move = -1

        if has_board:
            self._print_board()

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
        self.k += 1
        self.last_move = (x, y)

        self._print_board()
    
    def loop(self):
        self.is_running = True

        while self.is_running:
            if self.is_human:
                self.human_move = input('Move: ')
                self._human_move_to_number()
                self.execute_move(self.human_color, self.human_move)
                self.set_is_human(False)
    
    def _human_move_to_number(self):
        x = ascii_uppercase.find(self.human_move[:1].upper())
        y = int(self.human_move[1:])
        x, y = y, x
        self.human_move = y * self.n + x

        assert 0 <= x < self.n and 0 <= y < self.n, 'Move Out of Board'
        assert self.board[x][y] == 0, 'Position Already Occupied!'
    
    def _print_board(self):
        now = datetime.now(timezone('Asia/Ho_Chi_Minh')).strftime('%Y-%m-%d - %H:%M:%S')
        print(f'MOVE {self.k}: {now}')

        if not self.output_board: return

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
        if (x, y) == self.last_move:
            piece = piece.upper()
        return piece