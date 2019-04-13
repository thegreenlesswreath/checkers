import numpy as np
int32 = np.uint32
zero = int32(0)
one = int32(1)
two = int32(2)
seven = int32(7)
fourteen = int32(14)

#classless implementation
#A board is hence a 4-tuple of np.uint32, each integer describing a piece type, together with  the information of a counter, black/white to move, and must player to move 'pass'
#The squares are given a 'torus ordering', which makes the difference between adjacent squares a constant with respect to direction
#We begin by defining some useful functions on the np.uint32 'feature planes'


def rol(b, n):
    return (b << n) | (b >> (int32(32)-n))

def ror(b, n):
    return (b >> n) | (b << (int32(32)-n))

def rot(b, n):
    n %= 32
    n = int32(n)
    return (b << n) | (b >> (int32(32)-n))

def reverse(b):
    r = zero
    for i in range(32):
        r = r << one
        if b & one:
            r = r ^ one
        b = b >> one
    return r

def swap(b):
    return reverse(rol(b, int32(20))) 

#Next we define the masks, which encode out-of-bounds moves

masks_move = (int32(2**1 + 2**5 + 2**9 + 2**11 + 2**17 + 2**25 + 2**31),
        int32(2**0 + 2**1 + 2**6 + 2**9 + 2**12 + 2**17 + 2**18 + 2**25),
        int32(2**2 + 2**5 + 2**10 + 2**11 + 2**18 + 2**25 + 2**26 + 2**31),
        int32(2**0 + 2**2 + 2**6 + 2**10 + 2**12 + 2**18 + 2**26))
masks_jump = (int32(2**0 + 2**4 + 2**8 + 2**10 + 2**16 + 2**24 + 2**30),
            int32(2**7 + 2**8 + 2**13 + 2**16 + 2**19 + 2**24),
            int32(2**3 + 2**4 + 2**19 + 2**24 + 2**27 + 2**30),
            int32(2**1 + 2**3 + 2**7 + 2**11 + 2**13 + 2**19 + 2**27))
masks = masks_move + tuple(move+jump for move, jump in zip(masks_move, masks_jump)) + (int32(2**5 + 2**11 + 2**25 + 2**31),)

# A function that takes a board and returns a tuple of resultant boards is wanted
#First a function that returns 'move boards', 

def move_boards(m1, k1, m2, k2):
    return ((k1 | m1) & ~ror( k1 | m1 | k2 | m2, one) & ~masks[0],
        k1 & ~rol( k1 | m1 | k2 | m2, one) & ~masks[1],
        (k1 | m1) & ~ror( k1 | m1 | k2 | m2, seven) & ~masks[2],
        k1 & ~rol( k1 | m1 | k2 | m2, seven) & ~masks[3])


def jump_boards(m1, k1, m2, k2):
    return ((k1 | m1) & ror(k2 | m2, one) & ~ror( k1 | m1 | k2 | m2, two) & ~masks[4],
        k1 & rol(k2 | m2, one) & ~rol( k1 | m1 | k2 | m2, two) & ~masks[5],
        (k1 | m1) & ror(k2 | m2, seven) & ~ror( k1 | m1 | k2 | m2, fourteen) & ~masks[6],
        k1 & rol(k2 | m2, seven) & ~rol( k1 | m1 | k2 | m2, fourteen) & ~masks[7])

def both_boards(m1, k1, m2, k2):
    return move_boards(m1, k1, m2, k2) + jump_boards(m1, k1, m2, k2)

# And Now the action function

def next_states(state):
    (m1, k1, m2, k2, player, counter) = state
    states = ()

    if counter > 50:
        return states

    deleted = (0, 0, 0, 0, 1, -7, 7, -1)
    destination = (1, -7, 7, -1, 2, -14, 14, -2)

    for j, board in enumerate(both_boards(m1, k1, m2, k2)):
        
        for i in range(32):
            if board & one == one:
                new_m1, new_k1, new_m2, new_k2, new_counter = m1, k1, m2, k2, counter+one
                
                new_m1 = (m1 & ~(one << int32(i)))
                new_k1 = (k1 & ~(one << int32(i)))
                #deleting where we left
                new_m2 = (new_m2 & ~(one << int32((i+deleted[j])%32)))
                new_k2 = (new_k2 & ~(one << int32((i+deleted[j])%32)))
                #deleting where we 'jumped over'
                new_m1 |= ~masks[8] & rot(m1, destination[j]) & (one << int32((i+destination[j])%32))
                new_k1 |= (rot(k1, destination[j]) | masks[8]) &  (one << int32((i+destination[j])%32))
                #adding a piece to our destination

                if j > 3:
                    
                    diff = (new_m1 | new_k1) & ~(m1 | k1)
                    can_move = zero
                    for new_board in jump_boards(new_m1, new_k1, new_m2, new_k2):
                        can_move |= new_board
                    if can_move & diff & one  == one: #If we can continue to jump the new states will maintain the orientation and player
                        new_state = (new_m1, new_k1, new_m2, new_k2, player, zero)
                        states += (new_state,)
                        continue
                    new_counter = zero

                new_state = (swap(new_m2), swap(new_k2), swap(new_m1), swap(new_k2), ~player, new_counter)
                states += (new_state,)

            board >>= one
    return states

def start_position_generator():
    board = ((1,)*4,)*3 + ((0,)*4,)*2 + ((-1,)*4,)*3
    bb1 = zero
    bb2 = zero
    x = np.int8(0)
    y = np.int8(0)
    for i in range(32):   
        v = board[y][x]
        if v == 1:
            bb1 |= one
        if v == -1:
            bb2 |= one
        if i < 31:
            bb1 *= two
            bb2 *= two
        x = (x-(y%np.int8(2)))%np.int8(4)
        y = (y+np.int8(1)+np.int8(2)*np.int8(x == 3)*np.int8(y%2))%np.int8(8)       
    return (reverse(bb1), reverse(bb2))

def show(n):
    print('{0:032b}'.format(n), n, type(n))

def display(game_state):
    char = ('x', 'X' , 'o' , '0')
    grid = []
    for i in range(8):
        grid.append([' ', ' ', ' ', ' '])
    x, y = np.int8(0), np.int8(0)
    for board, c in zip(game_state[:4], char):
        for i in range(32):
            if board & one == one:
                grid[y][x] = c
            board >>= one
            x = (x-(y%np.int8(2)))%np.int8(4)
            y = (y+np.int8(1)+np.int8(2)*np.int8(x == 3)*np.int8(y%2))%np.int8(8)
    for i, row in enumerate(grid[::-1]):
            for c in row:
                print((c+'.')[::(-1)**(i%2)],end='')
            print('')
    print(game_state[4], game_state[5])
    print('___')
    


(bb1, bb2) = start_position_generator()
display((bb1, zero, bb2, zero, zero, zero))
root = (bb1, zero, bb2, zero, zero, zero)

from random import choice

total = 0
for i in range(10**2):
    state = root
    actions = next_states(state)
    moves = 0
    while actions:
        state = choice(actions)
        actions = next_states(state)
        moves += 1
    total += moves
print(total/100)


