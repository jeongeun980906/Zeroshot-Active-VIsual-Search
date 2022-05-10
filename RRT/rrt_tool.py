import copy
import numpy as np

def get_rstateidx(rstate,state):
    rstate = copy.deepcopy(rstate)
    array_rstate = []
    
    for state in rstate:
        temp =  [state['x'],state['z']]
        array_rstate.append(temp)
        
    array_rstate = np.array(array_rstate)
    
    array_state = np.array([state['x'],state['z']])
    final = np.linalg.norm(array_rstate - array_state,axis=1)

    idx= np.argmin(final)
    
    return idx