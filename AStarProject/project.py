import heapq
from warnings import warn
import cv2
import numpy as np
import math
import time
def main(image,startingX,startingY,endingX,endingY):
    #img = cv2.imread("AStarProject/maze2.jpeg")
    #image = cv2.imread("AStarProject/maze2.jpeg")
    img = cv2.imread(image)
    image = cv2.imread(image)
    window_name = 'image'

    start_time = time.time()

    """scale_percent = 15
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)"""

    dim = None
    if(image.shape[0] > 700):
        r = 700 / float(image.shape[0])
        dim = (int(image.shape[1] * r), 700)

        img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    #Morphological Transformation
    #kernel = np.ones((2,2),np.uint8)
    #img = cv2.dilate(img,kernel,iterations = 1)



    img = cv2.copyMakeBorder(
        img,
        top=1,
        bottom=1,
        left=1,
        right=1,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    (thresh, img) = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)


    grid = []
    width = img.shape[1]

    count = 0

    grid = img.copy()

    for y in range(img.shape[1]):
        for x in range(img.shape[0]):
            if grid[x][y] == 255:
                grid[x][y] = 0
            elif grid[x][y] == 0:
                grid[x][y] = 1



    path = []

    print("Image Manipulation Time: " + str(time.time()-start_time))

    algorithmStart_time = time.time()
    def manhattan(pixel1, pixel2):
        distance = abs(pixel1[0]-pixel2[0]) + abs(pixel1[1]-pixel2[1])
        return distance

    def euclidian(pixel1, pixel2):
        distance = math.sqrt((pixel1[0]-pixel2[0])**2 + (pixel1[1]-pixel2[1])**2)
        return distance

    def vonNeumannNeighborhood(pixel):
        xVal = pixel[0]
        yVal = pixel[1]
        neighborList = [(xVal-1,yVal),(xVal,yVal-1),(xVal+1,yVal),(xVal,yVal+1)]
        openNeighbors=[]
        for point in neighborList:
            x = point[0]
            y = point[1]
            actualPixel = grid[x][y]
            if actualPixel != 1:
                openNeighbors.append(point)
        
        return openNeighbors

    def AStarAlgorithm(begin,end,neighbors,cost):
        g = {
            start:0
        }
        f = {
            start: g[start] + cost(begin,end)
        }
        openset = []
        heapq.heappush(openset,(f[start],begin))
        closedset = set()
        cameFrom = {
            start:None
        }
        
        while openset:
            #print(openset)
            currentNode = heapq.heappop(openset)[1]
            if currentNode == end:
                path = []
                while currentNode != None:
                    path.insert(0,currentNode)
                    currentNode = cameFrom[currentNode]
                return path
            #openset.remove(currentNode)
            closedset.add(currentNode)
            for neighborNode in neighbors(currentNode):
                if neighborNode in closedset:
                    continue
                tentativeGScore = g[currentNode] + cost(currentNode,neighborNode)
                if tentativeGScore < g.get(neighborNode,float('inf')):
                    cameFrom[neighborNode] = currentNode
                    g[neighborNode] = tentativeGScore
                    f[neighborNode] = g[neighborNode] + cost(neighborNode,end)
                    if neighborNode not in openset:
                        heapq.heappush(openset, (f[neighborNode],neighborNode))
        return ["Failure"]
        
    start = (startingY,startingX)
    goal = (endingY,endingX)

    path = AStarAlgorithm(start,goal,vonNeumannNeighborhood,euclidian)
    print("Algorithm run time: " + str(time.time()-algorithmStart_time))      
    path = np.array(path)
    print(path)
    print(len(path))

    for i in path:
        image[i[0],i[1]] = [255,0,0]

    print(time.time()-start_time)

    #cv2.imshow(window_name, image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    
    #return tuple(map(tuple, path))
    return path


    #Without Priority Queue
    """
    def AStarAlgorithm(begin,end,neighbors,cost):
        g = {
            start:0
        }
        f = {
            start: g[start] + cost(begin,end)
        }
        openset = {start}
        closedset = set()
        cameFrom = {
            start:None
        }
        
        while openset:
            currentNode = min(openset, key=lambda x: f[x])
            
            if currentNode == end:
                path = []
                while currentNode != None:
                    path.insert(0,currentNode)
                    currentNode = cameFrom[currentNode]
                return path
            openset.remove(currentNode)
            closedset.add(currentNode)
            for neighborNode in neighbors(currentNode):
                if neighborNode in closedset:
                    continue
                tentativeGScore = g[currentNode] + cost(currentNode,neighborNode)
                if tentativeGScore < g.get(neighborNode,float('inf')):
                    cameFrom[neighborNode] = currentNode
                    g[neighborNode] = tentativeGScore
                    f[neighborNode] = g[neighborNode] + cost(neighborNode,end)
                    if neighborNode not in openset:
                        openset.add(neighborNode)
        return []
        """

    #With Priority Queue
    """
    def AStarAlgorithm(begin,end,neighbors,cost):
        g = {
            start:0
        }
        f = {
            start: g[start] + cost(begin,end)
        }
        openset = []
        heapq.heappush(openset,(f[start],begin))
        closedset = set()
        cameFrom = {
            start:None
        }
        
        while openset:
            #print(openset)
            currentNode = heapq.heappop(openset)[1]
            if currentNode == end:
                path = []
                while currentNode != None:
                    path.insert(0,currentNode)
                    currentNode = cameFrom[currentNode]
                return path
            #openset.remove(currentNode)
            closedset.add(currentNode)
            for neighborNode in neighbors(currentNode):
                if neighborNode in closedset:
                    continue
                tentativeGScore = g[currentNode] + cost(currentNode,neighborNode)
                if tentativeGScore < g.get(neighborNode,float('inf')):
                    cameFrom[neighborNode] = currentNode
                    g[neighborNode] = tentativeGScore
                    f[neighborNode] = g[neighborNode] + cost(neighborNode,end)
                    if neighborNode not in openset:
                        heapq.heappush(openset, (f[neighborNode],neighborNode))
        return ["Failure"]
        """