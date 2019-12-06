def GetLinearCost(k,b):
    '''
    Inputs:
    - k: slope
    - b: intercept

    Outputs:
    - f: linear cost function obejct
    '''
    def f(x): 
        return k*x+b
    return f 