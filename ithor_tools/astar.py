import math

class Node:
    def __init__(self,parent,position):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

def heuristic(node,goal):
    dx = abs(node.position[0] - goal.position[0])
    dy = abs(node.position[1] - goal.position[1])
    return math.sqrt(dx**2 + dy**2) # + (2**0.5 - 2 * 1) * min(dx, dy)

def astar(map,goal,cpos):
    map = 1-map[:,:,0]
    start_node = Node(None,cpos)
    end_node = Node(None,goal)

    openList = []
    closedList = []
    openList.append(start_node)

    while openList:
        currentNode = openList[0]
        currentidx = 0
        for e,o in enumerate(openList):
            # print(o.position,o.f)
            if o.f < currentNode.f:
                currentNode = o
                currentidx = e
        openList.pop(currentidx)
        closedList.append(currentNode)

        if currentNode.position == end_node.position:
            print('Done')
            path = []
            cur = currentNode
            while cur is not None:
                path.append(cur.position)
                cur = cur.parent
            return path[::-1]

        children = []
        for newpos in [[0,-1],[0,1],[1,0],[-1,0]]:#,[1,-1],[1,1],[-1,-1],[-1,1]]:
            newnodepos = [
                currentNode.position[0]+newpos[0], 
                currentNode.position[1]+newpos[1]
            ]
            range_cri = [
                newnodepos[0]>(map.shape[0]-1), newnodepos[0]<0,
                newnodepos[1]>(map.shape[1]-1), newnodepos[1]<0
            ]
            range_cri = sum(range_cri)
            colli_cri = map[newnodepos[0],newnodepos[1]]
            cri = range_cri+colli_cri
            if cri >0:
                continue
            new_node = Node(currentNode,newnodepos)
            children.append(new_node)
        
        for child in children:
            if child in closedList:
                continue
            child.g = currentNode.g + 1
            child.h = heuristic(child,end_node)
            child.f = child.g+child.h
            flag = True
            for openNode in openList:
                # print(openNode.position)
                if child.position == openNode.position and child.g > openNode.g:
                    flag = False
                    break
            if flag:
                openList.append(child)