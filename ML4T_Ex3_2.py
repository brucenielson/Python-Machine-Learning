import numpy as np
import pandas as pd
from io import StringIO
import random
import copy

#http://quantsoftware.gatech.edu/MC3-Project-2


"""
learner = ql.QLearner(num_states = 100, \ 
    num_actions = 4, \
    alpha = 0.2, \
    gamma = 0.9, \
    rar = 0.98, \
    radr = 0.999, \
    dyna = 0, \
    verbose = False)
"""
def state_from_col(x,y):
    return y*10 + x

def col_from_state(s):
    return (s % 10, s // 10)


def readmaze():
    maze = """3, 0, 0, 0, 0, 0, 0, 0, 0, 0
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                0, 0, 1, 1, 1, 1, 1, 0, 0, 0
                0, 5, 1, 0, 0, 0, 1, 0, 0, 0
                0, 5, 1, 0, 0, 0, 1, 0, 0, 0
                0, 0, 1, 0, 0, 0, 1, 0, 0, 0
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                0, 0, 0, 0, 2, 0, 0, 0, 0, 0"""
    return maze


def loadmaze():
    maze_io = StringIO(readmaze())
    maze = pd.read_csv(maze_io, header=None)
    return maze



class QLearner(object):
    def __init__(self, num_states = 100, num_actions = 4, alpha = 0.2, gamma = 0.9, rar = 0.98, radr = 0.999, dyna = 0, verbose = False):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        self.Q = np.zeros(shape=(num_states, num_actions))

    def query(self, s, r):
        # Take a random move?
        if random.uniform(0, 1) <= self.rar:
            a = random.randint(0, 3)
        else:
            # Get actions for this state s
            actions = self.Q[s, :]
            # Get best action
            a = np.argmax(actions)

        # Decay random rate
        self.rar = self.rar * self.radr

        # Update Q table
        self.Q[s, a] = ( (1-self.alpha) * self.Q[s, a] ) + (self.alpha  * (r + (self.gamma * self.Q[s, a])) )



    def querysetstate(self, s):
        # Get actions for this state s
        actions = self.Q[s,:]
        # Get best action
        a = np.argmax(actions)
        return a


class Maze(object):
    def __init__(self):
        self.maze = loadmaze()

    def move_maze(self, s, a):
        x, y = col_from_state(s)
        # 1 in 10 chance to move random direction
        if random.randint(0, 9) == 0:
            a = random.randint(0,3)

        if a == 0:
            x -= 1
        elif a == 1:
            y += 1
        elif a == 2:
            x += 1
        elif a == 3:
            y -= 1

        # Calclate new state
        if x > 9 or x < 0:
            return s
        elif y > 9 or y < 0:
            return s
        elif self.maze.iloc[x,y] == 1:
            return s
        else:
            return state_from_col(x,y)


    def get_location(self, s):
        x, y = col_from_state(s)
        location = self.maze.iloc[x,y]
        return location

    def mark(self, s):
        x, y = col_from_state(s)
        self.maze.iloc[x,y] = 'X'



def testqlearner():
    #Instantiate the learner with the constructor QLearner()
    learner = QLearner()
    maze = Maze()
    r = 0
    #s = initial_location
    s = state_from_col(9,4)
    #a = querysetstate(s)
    a = learner.querysetstate(s)
    #s_prime = new location according to action a
    s_prime = maze.move_maze(s, a)
    r = -1.0
    #while not converged:
    i = 0
    while i < 500:
        # a = query(s_prime, r)
        a = learner.query(s_prime, r)
        # s_prime = new location according to action a
        s_prime = maze.move_maze(s_prime, a)
        # if s_prime == goal:
        if maze.get_location(s_prime) == 3:
            #  r = +1
            r = 1000
            #  s_prime = start location
            s_prime = state_from_col(9,4)
            i += 1

        # else if s_prime == quicksand:
        elif maze.get_location(s_prime) == 5:
            # r = -100
            r = -100
        # else:
        else:
            # r = -1
            r = -1

    print("Finished Training")
    return show_maze_route(learner, maze)



def show_maze_route(learner, maze):
    s = state_from_col(9, 4)
    while maze.get_location(s) != 3:
        maze.mark(s)
        a = learner.querysetstate(s)
        s = maze.move_maze(s, a)

    return maze.maze



