import numpy as np


def EG(costs, epsilon):
    # If sampled value < epsilon, then randomly explore.
    if np.random.random_sample() < epsilon:
        return np.random.randint(len(costs))

    # Compute the average cost for each action and return the argmin.
    averages = []
    for action in costs:
        # If an action hasn't been taken at least once, explore it.
        if len(costs[action]) == 0:
            return action

        averages.append(np.mean(costs[action]))

    # Random tie-breaking.
    averages = np.array(averages)
    return np.random.choice(np.where(averages == averages.min())[0])


def UCB(costs, t):
    averages = []
    for action in costs:
        # If an action hasn't been taken at least once, explore it.
        if len(costs[action]) == 0:
            return action

        averages.append(np.mean(costs[action]))

    # Lower-confidence bounds for costs.
    bounds = [average - np.sqrt(2 * np.log(t) / len(costs[action]))
              for average, action in zip(averages, costs)]

    # Random tie-breaking.
    bounds = np.array(bounds)
    return np.random.choice(np.where(bounds == bounds.min())[0])


def TS(costs, mus, vs, alphas, betas):
    # If an action hasn't been taken at least once, explore it.
    for action in costs:
        if len(costs[action]) == 0:
            return action

    # Sample from the posterior predictive t distributions.
    samples = [np.random.standard_t(alpha * 2) * np.sqrt(beta * (v + 1) / v / alpha) + mu
               for mu, v, alpha, beta in zip(mus, vs, alphas, betas)]
    # print(samples, mus, alphas)

    # Random tie-breaking.
    samples = np.array(samples)
    return np.random.choice(np.where(samples == samples.min())[0])


class AGENT:
    def __init__(self, num_action, policy, parms):
        '''
        Agent could only take action based on their strategy/cost
        '''
        self.costs = {i: [] for i in range(num_action)}
        self.policy = policy
        if policy == 'TS':
            self.mus = [0 for _ in range(num_action)]
            self.vs = [0 for _ in range(num_action)]
            self.alphas = [0 for _ in range(num_action)]
            self.betas = [0 for _ in range(num_action)]

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
        if self.policy == 'EG':
            action = EG(self.costs, self.parms)
        elif self.policy == 'UCB':
            action = UCB(self.costs, self.t)
        elif self.policy == 'TS':
            action = TS(self.costs, self.mus, self.vs, self.alphas, self.betas)
        else:
            print("Unrecognized strategy!")
            return 

        self.cur_action = action
        for idx in env.paths[action]:
            env.flows[idx] += 1

        self.t += 1

    def receive_cost(self, env):
        cost = 0
        for idx in env.paths[self.cur_action]:
            cost += env.edge_cost(idx)

        # self.costs.append((self.cur_action, cost))
        self.costs[self.cur_action].append(cost)
        # Update the running Thompson sampling probability.
        if self.policy == 'TS':
            mu = self.mus[self.cur_action]
            v = self.vs[self.cur_action]

            self.mus[self.cur_action] = (v * mu + cost) / (v + 1)
            self.vs[self.cur_action] += 1
            self.alphas[self.cur_action] += 1 / 2
            self.betas[self.cur_action] += v * ((cost - mu) ** 2) / (v + 1) / 2

        return cost
