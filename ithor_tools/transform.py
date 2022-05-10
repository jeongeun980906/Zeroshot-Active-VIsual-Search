import numpy as np

def cornerpoint_projection(cornerpoints):
    res = []
    for e,c in enumerate(cornerpoints):
        if e%4==0 or e%4==1:
            res.append([c[0],c[2]])
    return res

def ndarray(list):
    array = np.array(list)
    return array