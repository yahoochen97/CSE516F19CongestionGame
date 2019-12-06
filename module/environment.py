import numpy as np

class ENVIRONMENT:
    def __init__(self, edges, paths):
        '''
        Environment for congestion games.
        Number of agents that go through each edge are maintained in self.flows.
        Paths that agents could take are stored in self.paths.

        Inputs:
        - edges: a list of functions that specify cost functions of agent
        - paths: a list of path which includes indexes of edges   
        '''
        self.edges = edges
        self.flows = np.zeros((len(edges),1))
        self.paths = []
        for path in paths:
            self.add_path(path)

    def add_path(self, edges):
        '''
        Add a possile path to the environment

        Inputs:
        - edges: index of edges that this path includes
        '''
        for edge in edges:
            if edge > len(self.edges):
                print("Index exceeds # of edges!")
                return
        self.paths.append(edges)

    def edge_cost(self, idx):
        '''
        Compute one edge cost
        '''
        return self.edges[idx](self.flows[idx])

    def reset(self):
        self.flows = np.zeros((len(self.edges),1))
