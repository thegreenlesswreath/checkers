from random import choice
import numpy as np

def jump_generator(i, j, board, player, is_king, jumped):

    d = -(-1)**(i%2)
    no_move = True

    for l in [player, -player][0:is_king+1]:
        for k in [j+d,j]:
            if -1 < i+2*l and i+2*l < 8:
                 if -1 < j+d*(-1)**(j == k) and j+d*(-1)**(j == k) < 4:
                     if board[i+l][k]*player < 0:
                         if board[i+2*l][j+d*(2*(j != k)-1)] == 0:
                             if not (i+l,k) in jumped:
                                 no_move = False
                                 for continuation in jump_generator(i+2*l,
                                                                    j+d*(-1)**(j == k),
                                                                    board,
                                                                    player,
                                                                    is_king,
                                                                    jumped+[(i+l, k)]):
                                     yield [(i+l, k)] + continuation
    if no_move:
        yield [(i, j)]

def move_generator(i, j, board, player, is_king):

        d = -(-1)**(i%2)
        no_move = True

        for l in [player, -player][0:is_king+1]:
            for k in [j+d,j]:
                if -1 < i+l and i+l < 8:
                     if -1 < k and k < 4:
                         if board[i+l][k]*player == 0:
                            yield [(i, j), (i+l, k)]
                            no_move = False
        if no_move:
            yield [(i, j)]

def rollout(state):
    if not state.score == None:
        return state.score
    new = state
    while new.score == None:
        new = new.next_state(choice(new.legal))
    return new.score

def state_to_tensor(state):
    board = state.board
    turn = board[-1]
    board = board[:-1]
    tensor = []
    for piece_type in [2,1,-1,-2][::turn]:
        tensor_slice = []
        for row in board[::turn]:
            tensor_slice.append([square == piece_type for square in row[::turn]])
        tensor_slice = np.array(tensor_slice)
        tensor.append(tensor_slice)

    return torch.tensor(np.array(tensor,dtype=np.double))

def state_to_move_tensor(state):

    move_tensor = torch.zeros(4,8,4)
    for move in state.legal:
        i, j = move[0][0], move[0][1]
        d = [move[1][0]-i, move[1][1]-j]
        e = -(-1)**(i%2)
        if state.turn == -1:
            i, j = 7-i, 3-j
            d = [7-move[1][0]-i, 3-move[1][1]-j]
            e = -e
        f_d = 2*(d[0] > 0) + (d[1]*e > 0)
        move_tensor[f_d, i, j] = 1
    return move_tensor.double()

class GameState():

    def __init__(self, board, counter=None, legal=None, score=None):

        self.board = board
        self.turn = self.board[-1]

        if legal == None:
            moves = []
            jumps = []
            for i, row in enumerate(self.board[:-1]):
                for j, square in enumerate(row):
                    if square*self.turn > 0:
                        for jump_seq in jump_generator(i, j, self.board[:-1], self.turn, square**2 > 1, []):
                            if len(jump_seq) > 1:
                                jumps.append([(i, j)]+jump_seq)
                        if not jumps:
                            for move in move_generator(i, j, self.board[:-1], self.turn, square**2 > 1):
                                if len(move) > 1:
                                    moves.append(move)
            if jumps:
                legal = jumps
            else:
                legal = moves
            legal = [tuple(a) for a in legal]
        self.legal = legal

        if counter == None:
            counter = 0
        self.counter = counter

        if score == None:
            if self.counter > 50:
                score = 0
            if not self.legal:
                score = -self.turn
        self.score = score

    def next_board(self, move):
        board = [list(row) for row in self.board[:-1]]
        is_king = board[move[0][0]][move[0][1]]**2 > 1
        if not is_king and 2*move[-1][0] == 7+7*self.turn:
            board[move[-1][0]][move[-1][1]] = self.turn*2
        else:
            board[move[-1][0]][move[-1][1]] = board[move[0][0]][move[0][1]]
        board[move[0][0]][move[0][1]] = 0
        move = move[1:-1]
        for square in move:
            board[square[0]][square[1]] = 0
        return tuple([tuple(row) for row in board])+(-self.turn,)

    def next_state(self, move):
        man_moved = self.board[move[0][0]][move[0][1]]**2 == 1
        reset_counter = man_moved or len(move) > 2
        return GameState(self.next_board(move), (not reset_counter)*(self.counter+1))


    def show(self):
        char = ['0','o',' ','x','X']
        for i, row in enumerate(self.board[:-1][::-1]):
            for square in row:
                print((char[square+2]+'_')[::-(-1)**(i%2)],end='')
            print('')
        print(self.counter)
        print(self.score)
