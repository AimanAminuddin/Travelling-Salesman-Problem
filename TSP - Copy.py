import matplotlib.pyplot as plt
import time
import numpy as np
import random
import heapq
import math 

# displays a MST
def plot_MST(pts,MST):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot([x[0] for x in pts], [x[1] for x in pts], "ko")
    for i in range(len(MST)): 
        for j in range(len(MST)): 
            if MST[i][j]!= np.infty: ax.plot([pts[i][0],pts[j][0]], [pts[i][1],pts[j][1]], "bo-")
    ax.title.set_text('Minimum Spanning Tree')
            
# computes the Euclidean distance between two points p1 and p2
def euclidean_distance(p1, p2):
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

# computes the length of a TSP solution
def compute_sol_length(graph,solution):
    length = 0
    for i in range(len(solution)-1): length = length + graph[solution[i]][solution[i+1]]
    return length

# computes with random method the TSP solution
def TSP_random(graph):
    return list(np.random.permutation(len(graph))) 
    
# computes with closest neighbor method the TSP solution

def TSP_closest_neighbor(graph):
    length = len(graph)
    visited = [False] * length
    # Generate a random start_node #
    # We have nodes: 0,1,2,3,...,n-1 
    start_node = random.randint(0,length - 1)
    visited[start_node] = True
    ## create a copy of start_node #
    vertex = start_node
    TSP_soln = [vertex]  
    while len(TSP_soln) < length:
        min_dist = float('inf')
        idx = None 
        for num in range(length):
            if num != vertex and not visited[num]:
                distance = graph[vertex][num]
                if min_dist > distance:
                    min_dist = distance
                    idx = num


        ## check if min_dist = inf = > implies that there no path
        ## between vertex and all nodes
        if min_dist == float('inf'):
            break
        else:
            visited[idx] = True
            TSP_soln.append(idx)
            # repeat the loop # 
            vertex = idx

    TSP_soln.append(start_node)

    return TSP_soln 


# computes the Minimum Spanning Tree
def compute_MST(graph): 
    length = len(graph)
    edge_lst = []
    MST = []
    # create a list of sets that represent
    # connected subgraphs used to build up
    # MST
    lst = [{i} for i in range(length)]

    for i in range(length):
        for j in range(length):
            if i >= j:
                continue
            else:
                edge = [i,j,graph[i][j]]
                edge_lst.append(edge)

    # sort edge_lst in terms of weight in ascending order #
    edge_lst = sorted(edge_lst,key = lambda x: x[2])

    # Iteratively build up MST #
    while len(MST) < length - 1:
        current_edge = edge_lst.pop(0)
        first_node = current_edge[0]
        second_node = current_edge[1]
        is_cycle = False
        first_idx = None
        second_idx = None 

        for num in range(len(lst)):
            subset = lst[num]
            is_first = first_node in subset
            is_second = second_node in subset 
            if is_first and is_second:
                is_cycle = True 
                break 
            
            else:
                if is_first:
                    first_idx = num

                elif is_second:
                    second_idx = num

                else:
                    continue

        if is_cycle:
            continue

        else:
            
            ## combine 2 sets together ##
            first_subset = lst[first_idx]
            second_subset = lst[second_idx]
            first_subset.update(second_subset)
            lst[first_idx] = first_subset
            lst.pop(second_idx)
            MST.append((first_node,second_node))

    return MST 
                    
    
    
#graph = [[0, 2.83, 5],[2.83, 0, 2.24],[5, 2.24, 0]]
#print(compute_MST(graph)) 


# computes the preorder walk in the tree corresponding to DFS
def convert_to_adj_lst(graph):
    length = len(graph)
    my_dict = {num:None for num in range(length)}
    for x in range(length):
        adjacent = [] 
        for y in range(length):
            if x != y and graph[x][y] != float('inf'):
                adjacent.append(y)

        my_dict[x] = adjacent

    return my_dict 
            
def DFS_preorder(graph,start_node):
    victory = convert_to_adj_lst(graph)
    length = len(graph)
    stack = [start_node]
    explored = [False]*length
    explored[start_node] = True
    walk = []

    while stack != []:
        current_node = stack.pop(-1)
        adjacent = victory[current_node]
        walk.append(current_node) 

        for node in adjacent:
            if not explored[node]:
                stack.append(node)
                explored[node] = True

    return walk 
       
        
        

# computes with Minimum Spanning Tree the TSP solution
def TSP_min_spanning_tree(graph): 
    MST = compute_MST(graph)
    
    graph_MST = [[]]*len(graph)
    for i in range(len(graph)): graph_MST[i] = [np.infty for j in range(len(graph))] 
    for i in range(len(MST)): 
        graph_MST[MST[i][0]][MST[i][1]] = graph[MST[i][0]][MST[i][1]]
        graph_MST[MST[i][1]][MST[i][0]] = graph[MST[i][1]][MST[i][0]]
    
    plot_MST(pts,graph_MST)        
    return DFS_preorder(graph_MST,0)

    
    
NUMBER_OF_POINTS = 20 

# generates random points and sort them accoridng to x coordinate
pts = []
for i in range(NUMBER_OF_POINTS): pts.append([random.randint(0,1000),random.randint(0,1000)])
pts = sorted(pts, key=lambda x: x[0])

graph = [[]]*NUMBER_OF_POINTS
for i in range(NUMBER_OF_POINTS): graph[i] = [euclidean_distance(pts[i],pts[j]) for j in range(NUMBER_OF_POINTS)]

# computes the TSP solutions
print("Computing TSP solution using random technique ... ",end="")
t = time.time()
TSP_sol_random = TSP_random(graph)
print("done ! \n It took %.2f seconds - " %(time.time() - t),end="")
print("length found: %.2f" % (compute_sol_length(graph,TSP_sol_random)))

print("Computing TSP solution using closest neighbor technique ... ",end="")
t = time.time()
TSP_sol_closest_neighbor = TSP_closest_neighbor(graph)
print("done ! \n It took %.2f seconds - " %(time.time() - t),end="")
print("length found: %.2f" % (compute_sol_length(graph,TSP_sol_closest_neighbor)))

print("Computing TSP solution using Minimum Spanning Tree technique ... ",end="")
t = time.time()
TSP_sol_min_spanning_tree = TSP_min_spanning_tree(graph)
print("done ! \n It took %.2f seconds - " %(time.time() - t),end="")
print("length found: %.2f" % (compute_sol_length(graph,TSP_sol_min_spanning_tree)))


# closes the TSP solution for display if needed
if TSP_sol_random[0] != TSP_sol_random[-1]: TSP_sol_random.append(TSP_sol_random[0])
if TSP_sol_closest_neighbor[0] != TSP_sol_closest_neighbor[-1]: TSP_sol_closest_neighbor.append(TSP_sol_closest_neighbor[0])
if TSP_sol_min_spanning_tree[0] != TSP_sol_min_spanning_tree[-1]: TSP_sol_min_spanning_tree.append(TSP_sol_min_spanning_tree[0])


# displays the TSP solution
if NUMBER_OF_POINTS<100:
    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.plot([x[0] for x in pts], [x[1] for x in pts], "ko")
    ax.title.set_text('Points')  
    ax = fig.add_subplot(222)
    ax.plot([x[0] for x in pts], [x[1] for x in pts], "ko")
    ax.plot([pts[x][0] for x in TSP_sol_random], [pts[x][1] for x in TSP_sol_random], "ro--")
    ax.title.set_text('TSP Random')
    ax = fig.add_subplot(223)
    ax.plot([x[0] for x in pts], [x[1] for x in pts], "ko")
    ax.plot([pts[x][0] for x in TSP_sol_closest_neighbor], [pts[x][1] for x in TSP_sol_closest_neighbor], "ro--")
    ax.title.set_text('TSP Closest Neighbor')
    ax = fig.add_subplot(224)
    ax.plot([x[0] for x in pts], [x[1] for x in pts], "ko")
    ax.plot([pts[x][0] for x in TSP_sol_min_spanning_tree], [pts[x][1] for x in TSP_sol_min_spanning_tree], "ro--")
    ax.title.set_text('TSP Minimum Spanning Tree')
    plt.show(block=False)
    
    
    


    
