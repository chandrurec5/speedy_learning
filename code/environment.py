import numpy

class Chain(object):
    """
    state: 0,1,2,...,num_states-1
    action: -1(left), +1(right)
    """
    def __init__(self, init_state=25):
        self.chain_size = 50
        self.state = numpy.random.randint(0, self.chain_size)
        self.slip_prob = 0.1
        self.goal_reward = 1
        self.goal = [9, 40]
        
    def reset(self, init_state=25):
        self.state = numpy.random.randint(0, self.chain_size)
        
    def observeState(self):
        return self.state
        
    def isAtGoal(self):
        return self.state in self.goal
    
    # return: action using the optimal policy
    def optimalPolicy(self):
        cur_state = self.observeState()
        goal = self.goal[0]
        if abs(cur_state-self.goal[0]) > abs(cur_state-self.goal[1]):
            goal = self.goal[1]
        elif abs(cur_state-self.goal[0]) == abs(cur_state-self.goal[1]):
            goal = self.goal[numpy.random.randint(0,2)]
        return 1 if goal>cur_state else -1
    
    # return: reward
    def takeAction(self, action):
        if not action in [-1, 1]:
            print "Invalid action: ", action
            raise StandardError
        if numpy.random.random() < self.slip_prob:
            action = -action
        self.state += action
        self.state = max(0, self.state)
        self.state = min(self.chain_size-1, self.state)
        if self.isAtGoal():
            return self.goal_reward
        else:
            return 0
            
            

class RandomWalk(object):
    """
        The random-walk problems were all based on the standard
        Markov chain (Sutton 1988; Sutton & Barto 1998) with a
        linear arrangement of five states plus two absorbing termi-
        nal states at each end. 
        Episodes began in the center state of the five, then transitioned
        randomly with equal probability to a neighboring state until 
        a terminal state was reached.
        The rewards were zero everywhere except on transition into
        the right terminal state, upon which the reward was +1.
    """
    def __init__(self):
        self.num_states = 5
        self.state = int(self.num_states/2)
        self.goal_reward = 1.
        self.goal = [0, self.num_states - 1]
        
    def reset(self):
        #self.state = numpy.random.randint(1, self.num_states-1)
        self.state = int(self.num_states/2)
    
    def observeState(self):
        return self.state
    
    def isAtGoal(self):
        return self.state in self.goal
    
    # return: action following the random policy
    def randomPolicy(self):
        ava_actions = [-1,1]
        return ava_actions[numpy.random.randint(0,2)]
            
    # return: reward
    def takeAction(self, action):
        if not action in [-1, 1]:
            print "Invalid action: ", action
            raise StandardError
        self.state += action
        self.state = max(0, self.state)
        self.state = min(self.num_states-1, self.state)
        if self.state == self.num_states - 1:
            return self.goal_reward
        else:
            return 0.
            
            
class BoyanChain(object):
    """
    Justin Boyan 2002: Least Square Temporal Difference Learning
    """
    def __init__(self):
        self.num_states = 13
        self.state = self.num_states - 1
        self.goal = 0
    
    def reset(self):
        self.state = self.num_states - 1
    
    def observeState(self):
        return self.state
    
    def isAtGoal(self):
        return self.state == self.goal
    
    # return: action following the random policy
    def randomPolicy(self):
        cur_state = self.observeState()
        if cur_state == 1:
            return -1
        else:
            ava_actions = [-1,-2]
            return ava_actions[numpy.random.randint(0,2)]
    
    # return: reward
    def takeAction(self, action):
        if not action in [-1, -2]:
            print "Invalid action: ", action
            raise StandardError
        cur_state = self.observeState()
        self.state += action
        self.state = max(0, self.state)
        self.state = min(self.num_states-1, self.state)
        if cur_state == 1:
            return -2.
        else:
            return -3.
    