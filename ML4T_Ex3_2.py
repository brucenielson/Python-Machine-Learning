import numpy as np
import pandas as pd
from io import StringIO
import random
import time
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
        self.Q = np.zeros(shape=(num_states, num_actions)) #np.full((num_states, num_actions), 0)
        self.last_a = None
        self.last_s = None

    def query(self, s_prime, r):
        # Get last move
        s = self.last_s
        a = self.last_a
        # s_prime is new state, so get a_prime which is optimal move for that state according to Q
        # Get actions for this state s
        actions = self.Q[s_prime, :]
        # Get best action

        a_prime = np.argmax(actions)
        #try:
        #    a_prime = np.nanargmax(actions)
        #except:
        #    a_prime = 0

        # Decay random rate
        self.rar = self.rar * self.radr

        # Update Q table
        if (s != None and a != None):
            #self.Q[s, a] = ( (1-self.alpha) * self.Q[s, a] ) + (self.alpha  * (r + (self.gamma * self.Q[s_prime, a_prime])) )

            current_val = self.Q[s, a]
            if pd.isnull(current_val): current_val = 0
            prime = self.Q[s_prime, a_prime]
            if pd.isnull(prime): prime = 0
            new_val = ( (1-self.alpha) * current_val ) + (self.alpha  * (r + (self.gamma * prime)) )
            self.Q[s, a] = new_val
        self.last_s = s_prime

        # Are we going to take a_prime or a random move?
        if random.uniform(0, 1) <= self.rar:
            a = random.randint(0, 3)
        else:
            a = a_prime

        self.last_a = a
        return a



    def querysetstate(self, s):
        # Get actions for this state s
        actions = self.Q[s,:]
        # Get best action
        a = np.argmax(actions)
        #try:
        #    a = np.nanargmax(actions)
        #except:
        #    a = 0
        return a


class Maze(object):
    def __init__(self):
        self.maze = loadmaze()

    def move_maze(self, s, a):
        x, y = col_from_state(s)
        # 1 in 10 chance to move random direction
        if random.randint(0, 10) == 0:
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
    start = time.time()
    learner = QLearner()
    maze = Maze()
    r = 0
    #s = initial_location
    s = state_from_col(9,4)
    #a = querysetstate(s)
    a = learner.querysetstate(s)
    #s_prime = new location according to action a
    s_prime = maze.move_maze(s, a)
    r = -0 # I am not sure how to make thsi work with r = -1
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
            r = 100
            #  s_prime = start location
            s_prime = state_from_col(9,4)
            i += 1

        # else if s_prime == quicksand:
        elif maze.get_location(s_prime) == 5:
            # r = -100
            r = -1000
        # else:
        else:
            # r = -1
            r = -1 # I am not sure how to make thsi work with r = -1
            #TODO: There is something weird about this. It means that the best path will continually degrade compared to paths not used. It encourages too much wandering

    end = time.time()
    length = end - start
    print("Training Run Time: %.2f seconds" % length)
    start = time.time()
    result = show_maze_route(learner, maze)
    end = time.time()
    length = end - start
    print("Display Route Run Time: %.2f seconds" % length)
    return result



def show_maze_route(learner, maze):
    s = state_from_col(9, 4)
    while maze.get_location(s) != 3:
        maze.mark(s)
        a = learner.querysetstate(s)
        s = maze.move_maze(s, a)

    return maze.maze



