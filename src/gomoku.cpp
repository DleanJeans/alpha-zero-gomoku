#include <assert.h>
#include <math.h>
#include <gomoku.h>

Gomoku::Gomoku(unsigned int n, unsigned int n_in_row) {
  this->n = n;
  this->n_in_row = n_in_row;

  for (unsigned int i = 0; i < n; i++) {
    this->board.emplace_back(std::vector<int>(n, 0));
  }
};

std::vector<std::tuple<unsigned int, unsigned int>> Gomoku::get_legal_moves() {
  auto n = this->n;
  std::vector<std::tuple<unsigned int, unsigned int>> legal_moves;

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      if (this->board[i][j] == 0) {
        legal_moves.emplace_back(std::make_tuple(i, j));
      }
    }
  }

  return legal_moves;
};

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
};

void Gomoku::execute_move(int color,
                          const std::tuple<unsigned int, unsigned int> &move) {
  auto i = std::get<0>(move);
  auto j = std::get<1>(move);

  assert(this->board[i][j] == 0);

  this->board[i][j] = color;
};

const std::vector<std::vector<int>> &Gomoku::get_board() {
  return this->board;
};

std::tuple<bool, int> Gomoku::get_game_status() {
  // return (is ended, winner)
  auto n = this->n;
  auto n_in_row = this->n_in_row;

  for (unsigned int i = 0; i < n; i++) {
    for (unsigned int j = 0; j < n; j++) {
      if (this->board[i][j] == 0) {
        continue;
      }

      if (j <= n - n_in_row) {
        auto sum = 0;
        for (unsigned int k = 0; k < n_in_row; k++) {
          sum += this->board[i][j + k];
        }
        if (abs(sum) == n_in_row) {
          return std::make_tuple(true, this->board[i][j]);
        }
      }

      if (i <= n - n_in_row) {
        auto sum = 0;
        for (unsigned int k = 0; k < n_in_row; k++) {
          sum += this->board[i + k][j];
        }
        if (abs(sum) == n_in_row) {
          return std::make_tuple(true, this->board[i][j]);
        }
      }

      if (i <= n - n_in_row && j <= n - n_in_row) {
        auto sum = 0;
        for (unsigned int k = 0; k < n_in_row; k++) {
          sum += this->board[i + k][j + k];
        }
        if (abs(sum) == n_in_row) {
          return std::make_tuple(true, this->board[i][j]);
        }
      }

      if (i <= n - n_in_row && j >= n_in_row - 1) {
        auto sum = 0;
        for (unsigned int k = 0; k < n_in_row; k++) {
          sum += this->board[i + k][j - k];
        }
        if (abs(sum) == n_in_row) {
          return std::make_tuple(true, this->board[i][j]);
        }
      }
    }
  }

  if (this->has_legal_moves()) {
    return std::make_tuple(false, 0);
  } else {
    return std::make_tuple(true, 0);
  }
};