import matplotlib.pyplot as plt
import numpy as np
from module.agent import AGENT
from module.environment import ENVIRONMENT
from module.cost import GetLinearCost

C1 = GetLinearCost(0,25)
C2 = GetLinearCost(1/100,0)
C3 = GetLinearCost(0,0)
EDGES = [C2, C1, C3, C1, C2]

PATHS = [[0,1], [3,4], [0,2,4]]
nA = len(PATHS)
env = ENVIRONMENT(EDGES, PATHS)

def example():
    N = 2000
    agents = [AGENT(nA, 'EG', 0) for i in range(N)]
    iters = 100
    costs = np.zeros((iters, N))
    for it in range(iters):
        for agent in agents:
            agent.move(env)
        
        for i,agent in enumerate(agents):
            costs[it, i] = agent.receive_cost(env)
        
        env.reset()

    plt.plot(range(iters), np.average(costs, axis=1))
    print(np.average(costs, axis=1))
    plt.show()

def main():
    example()

if __name__ == "__main__":
    main()