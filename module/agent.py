import numpy as np

def EG(costs,parms):
    '''
    Epsilon greedy action.
    '''
    return 0

def UCB(costs,parms):
    '''
    UCB action.
    '''
    return 0

def TS(costs,parms):
    '''
    Thompson sampling action.
    '''
    return 0

class AGENT:
    def __init__(self, num_action, policy, parms):
        '''
        Agent could only take action based on their strategy/cost
        '''
        self.costs = []
        self.policy = policy
        self.parms = parms
        self.cur_action = 0
        self.t = 1

    def move(self, env):
        '''
        Agent chooses an action.
        alg: EG, UCB, TS

        Inputs:
        - env: environment
        - action: element of action space
        - alg: action strategy
        - parms: alg parameters
        '''
        action = None

        if self.policy == 'EG':
            action = EG(self.costs, self.parms)
        elif self.policy == 'UCB':
            action = UCB(self.costs, self.t)
        elif self.policy == 'TS':
            action = TS(self.costs, self.parms)
        else:
            print("Unrecoginized strategy!")
            return 

        self.cur_action = action
        for idx in env.paths[action]:
            env.flows[idx] += 1

        self.t += 1

    def receive_cost(self, env):
        cost = 0
        for idx in env.paths[self.cur_action]:
            cost += env.edge_cost(idx)

        self.costs.append((self.cur_action,cost))
        return cost

