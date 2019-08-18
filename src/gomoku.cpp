#include <math.h>
#include <iostream>

#include "gomoku.h"

Gomoku::Gomoku(unsigned int n, unsigned int n_in_row, int first_color)
    : n(n), n_in_row(n_in_row), cur_color(first_color), last_move(-1) {
  this->board = std::vector<std::vector<int>>(n, std::vector<int>(n, 0));
}

std::vector<int> Gomoku::get_legal_moves() {
  auto n = this->n;
  std::vector<int> legal_moves(this->get_action_size(), 0);

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      if (this->board[i][j] == 0) {
        legal_moves[i * n + j] = 1;
      }
    }
  }

  return legal_moves;
}

bool Gomoku::has_legal_moves() {
  auto n = this->n;

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      if (this->board[i][j] == 0) {
        return true;
      }
    }
  }
  return false;
}

void Gomoku::execute_move(move_type move) {
  auto i = move / this->n;
  auto j = move % this->n;

  if (this->board[i][j] != 0) {
    throw std::runtime_error("execute_move board[i][j] != 0.");
  }

  this->board[i][j] = this->cur_color;
  this->last_move = move;
  // change player
  this->cur_color = -this->cur_color;
}

std::vector<int> Gomoku::get_game_status() {
  // return (is ended, winner)
  auto n = this->n;
  auto n_in_row = this->n_in_row;

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      if (this->board[i][j] == 0)
        continue;

      if (j <= n - n_in_row)
        if (check_line(i, j, 0, 1))
          return {1, this->board[i][j]};

      if (i <= n - n_in_row)
        if (check_line(i, j, 1, 0))
          return {1, this->board[i][j]};

      if (i <= n - n_in_row && j <= n - n_in_row)
        if (check_line(i, j, 1, 1))
          return {1, this->board[i][j]};

      if (i <= n - n_in_row && j >= n_in_row - 1)
        if (check_line(i, j, 1, -1))
          return {1, this->board[i][j]};
    }
  }

  if (this->has_legal_moves()) {
    return {0, 0};
  } else {
    return {1, 0};
  }
}

bool Gomoku::check_line(int x, int y, int x_mul, int y_mul) {
  int sum = 0;
  int blocks = 0;
  int root_piece = this->board[x][y];

  int current_x = x;
  int current_y = y;
  int current_piece = root_piece;
  int direction = -1;

  do {
    current_x += x_mul * direction;
    current_y += y_mul * direction;
    if (current_x < 0 || current_x >= n || current_y < 0 || current_y >= n)
      break;
    current_piece = this->board[current_x][current_y];

    if (current_piece == -root_piece)
      blocks += 1;
  } while (current_piece == root_piece);

  direction = 1;

  do {
    current_x += x_mul * direction;
    current_y += y_mul * direction;
    if (current_x < 0 || current_x >= n || current_y < 0 || current_y >= n)
      break;
    current_piece = this->board[current_x][current_y];

    if (current_piece == root_piece)
      sum += 1;
    else if (current_piece == -root_piece)
      blocks += 1;
  } while (current_piece == root_piece && sum <= n_in_row);
  
  return abs(sum) >= n_in_row && blocks < 2;
}

void Gomoku::display() const {
  auto n = this->board.size();
  for (unsigned int i = 0; i < n; i++)
    std::cout << " _";
  std::cout << std::endl;

  for (unsigned int x = 0; x < n; x++) {
    std::cout << "|";
    for (unsigned int y = 0; y < n; y++) {
      auto s = "";
      switch (this->board[x][y]) {
        case 1: s = "x";
        break;
        case -1: s = "o";
        break;
        default: s = "_";
        break;
      }
      std::cout << s << "|";
    }
    std::cout << std::endl;
  }
}
