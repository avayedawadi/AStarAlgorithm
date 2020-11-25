import heapq
from warnings import warn
import pygame
import cv2
import numpy as np
import math
img = cv2.imread("AStarProject/maze2.jpeg")
image = cv2.imread("AStarProject/maze2.jpeg")
window_name = 'image'


scale_percent = 15
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

(thresh, img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

grid = []
width = img.shape[1]

count = 0

# print(len(img[0]))
# print(width)

grid = img.copy()
#grid = grid.tolist()


for y in range(img.shape[1]):
    for x in range(img.shape[0]):
        if grid[x][y] == 255:
            grid[x][y] = 0
        elif grid[x][y] == 0:
            grid[x][y] = 1



path = []

"""
from warnings import warn
import heapq

class Node:
    
    #A node class for A* Pathfinding
    

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
    def __repr__(self):
      return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
      return self.f < other.f
    
    # defining greater than for purposes of heap queue
    def __gt__(self, other):
      return self.f > other.f

def return_path(current_node):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    return path[::-1]  # Return reversed path


def astar(maze, start, end, allow_diagonal_movement = True):
    

    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Heapify the open_list and Add the start node
    heapq.heapify(open_list) 
    heapq.heappush(open_list, start_node)

    # Adding a stop condition
    outer_iterations = 0
    max_iterations = (len(maze[0]) * len(maze) // 2)

    # what squares do we search
    adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)
    if allow_diagonal_movement:
        adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1),)

    # Loop until you find the end
    while len(open_list) > 0:

        #print(open_list)

        outer_iterations += 1

        if outer_iterations > max_iterations:
          # if we hit this point return the path such as it is
          # it will not contain the destination
          warn("giving up on pathfinding too many iterations")
          return return_path(current_node)       
        
        # Get the current node
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            return return_path(current_node)

        # Generate children
        children = []
        
        for new_position in adjacent_squares: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + (((child.position[0] - child.parent.position[0]) ** 2) + ((child.position[1] - child.parent.position[1]) ** 2))**0.5
            child.h = (((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2))**0.5
            child.f = child.g + child.h
            
            # Child is already in the open list
            index = None
            for i in range(0, len(open_list)):
                if child.position == open_list[i].position:
                    index = i
                    break
        
            if index:
                if child.g >= open_list[index].g:
                    continue
                else:
                    open_list[index] = open_list[-1]
                    open_list.pop()
                    if index < len(open_list):
                        heapq._siftup(open_list, index)
                        heapq._siftdown(open_list, 0, index)

            # Add the child to the open list
            heapq.heappush(open_list, child)

    warn("Couldn't get a path to destination")
    return None

def example(print_maze = False):

    maze = grid

    
    start = (200, 200)
    #end = (len(maze)-1, len(maze[0])-1)
    end = (250,250)

    path = astar(maze, start, end)

    if print_maze:
      for step in path:
        maze[step[0]][step[1]] = 2
      
      for row in maze:
        line = []
        for col in row:
          if col == 1:
            line.append("\u2588")
          elif col == 0:
            line.append(" ")
          elif col == 2:
            line.append(".")
        print("".join(line))

    #print(path)
    return path

#example()
"""

def manhattan(pixel1, pixel2):
    distance = abs(pixel1[0]-pixel2[0]) + abs(pixel1[1]-pixel2[1])
    return distance

def euclidian(pixel1, pixel2):
    distance = math.sqrt((pixel1[0]-pixel2[0])**2 + (pixel1[1]-pixel2[1])**2)
    return distance

def blocked(pixel):
    xVal = pixel[0]
    yVal = pixel[1]
    actualPixel = grid[xVal][yVal]
    if actualPixel == 1:
        return True
def vonNeumannNeighborhood(pixel):
    xVal = pixel[0]
    yVal = pixel[1]
    neighborList = []
    neighbor[0] = xVal-1,y
    


path = example()
path = np.array(path)
print(path)

for y in range(img.shape[1]):
    for x in range(img.shape[0]):
        for i in path:
            if y == i[0] and x == i[1]:
                img[i[0], i[1]] = 0


cv2.imshow(window_name, img)
cv2.waitKey(0)
cv2.destroyAllWindows()
