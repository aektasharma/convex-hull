import numpy as np
import matplotlib.pyplot as plt
from random_geometry_points.circle2d import Circle2D
import time as tm

def square(n):

    k=n/4
    k=round(k)
    x=np.array([(x,k) for x in range(0,k)])
    x=np.concatenate((x,([(0,x) for x in range(0,k)])),axis=0)
    x=np.concatenate((x,([(x,0) for x in range(0,k)])),axis=0)
    x=np.concatenate((x,([(k,x) for x in range(0,k+1)])),axis=0)
    #x=np.concatenate([k,k],axis=0)
    #print(x)
    plt.scatter(x[:,0],x[:,1],c='g')
    #P = np.random.uniform(low=0,high=10,size=(n,2))
    return x

def straightline(n):
    
    P1 = np.random.uniform(low=0,high=10,size=(n,2))
    P1[:,1]=(P1[:,0]*2) + 3
    plt.scatter(P1[:,0],P1[:,1], c='g')
    
    return P1
    

def alltogether(n):
    P1 = np.random.uniform(low=0,high=10,size=(n,2))
    P1[:,1]=(P1[:,0]*2) + 3
    
    j=n/5
    j=round(j)
    x=np.array([(x,j) for x in range(0,j+1)])
    x=np.concatenate((x,([(0,x) for x in range(0,j+1)])),axis=0)
    x=np.concatenate((x,([(x,0) for x in range(0,j+1)])),axis=0)
    x=np.concatenate((x,([(j,x) for x in range(0,j+1)])),axis=0)
    
    P1=np.concatenate((P1,x))
    
    
    circle = Circle2D(150, 100, 100)
    #circle = Circle2D(750, 750, 700)
    circlepoints = circle.create_random_points(n)

    y=np.asarray(circlepoints)
    
    P1=np.concatenate((P1,y))
    print("Length of pointset is ",len(P1))
    plt.scatter(P1[:,0],P1[:,1],c='g')
    return P1
    
    
    
    
def circle(n):
    circle = Circle2D(10, 10, 10)
    circlepoints = circle.create_random_points(n)
#print(random_circle_points)
    x=np.asarray(circlepoints)
    plt.scatter(x[:,0],x[:,1],c='g')
    #plt.scatter(x[:,0],x[:,1])
    return x
#p=pointCloud1(21)

#plt.scatter(p[:,0],p[:,1])

def randompoints(n):

   
    P1 = np.random.uniform(low=0,high=1000,size=(n,2))
    plt.scatter(P1[:,0],P1[:,1],c='g')
    
   
    return P1

def main():
    start=tm.time()
    n = 50
    P = square(n)
    #P=straightline(n)
    #P=circle(n)
   # P=alltogether(n)
    #P=randompoints(n)
    m = 3
    
    guess=False
   # t=0
    t=1
    while guess==False:
        #t=t+1
        #m = 2**(2**t)
        m=m**t if m**t <= n else n
        t=2
        print("m value is ",m)
        k = np.floor(1 + n/m)
        print("Value for k")
        print(k)
        subConv = subsetConv(P,k)
        #print("subConv")
        #print(subConv)
        plt.figure()
        
        plt.plot(P[:,0],P[:,1],'g o',label='Point Cloud',markersize=10)
      #  #plt.plot(P[:,0],P[:,1],'o',label='Point Cloud',markersize=10)
        #plt.plot(P[:,0],P[:,1],'o',label='Point Cloud')
        #plt.grid()
        plt.tick_params(axis='both', labelsize=10)
        #plt.title('Convex Hull using Chan\'s Algorithm',fontsize=20,color='black')
        plt.title('Sub Hulls using Graham Scan',fontsize=20,color='black')
        nSubHullPoints = np.zeros(len(subConv))
        SubHullPoints = np.zeros(shape=(1,2))
        
        
        for key, value in subConv.items():
            
            L = value
            SubHullPoints = np.concatenate((SubHullPoints,L),axis=0)
            nSubHullPoints[key] = np.shape(L)[0]
            
            plt.plot(L[:,0],L[:,1], 'b-', picker=5)
            plt.plot(L[:,0],L[:,1], 'b o',markersize=10, picker=5)
            plt.plot([L[-1,0],L[0,0]],[L[-1,1],L[0,1]], 'b-', picker=5)
        plt.show()
            
            
    
        nSubHullPoints = np.sum(nSubHullPoints)
        print('Points On Sub-Hulls = {}'.format(nSubHullPoints))
        SubHullPoints = np.delete(SubHullPoints,(0),axis=0)
        
       
        print("gift wrapping on total points-",len(SubHullPoints))
        H = GiftWrapping(SubHullPoints,m)
        if isinstance(H, np.ndarray) == True:
            guess= True
            print('Points On the Total Hull = {}'.format(len(H)))
            plt.plot(P[:,0],P[:,1],'g o',label='Point Cloud',markersize=10)
            
            plott(H)
        
        
    
    plt.show()
    end=tm.time()
    print("Time Taken: ",end-start,"seconds")
    
    
def plott(H):
    plt.title('Convex Hull using Chan\'s Algorithm',fontsize=20,color='black')
    plt.plot(H[:,0],H[:,1],'r-',linewidth=3,label='Total Convex Hull')
    plt.plot(H[:,0],H[:,1],'r .',markersize=20,label='Total Convex Hull')
    plt.plot([H[-1,0],H[0,0]],[H[-1,1],H[0,1]], 'r-',linewidth=3)



def subsetConv(P,k):
    # split the index into k parts.
    subset_indicies = splitPoints(range(len(P)),k)
    #print("PRINTING VALUE",subset_indicies)
    
    subHulls = dict.fromkeys(range(len(subset_indicies)), [])
    
   # print("grahm scan is called -",len(subset_indicies))
    for k in range(len(subset_indicies)):
      
        Pi = P[subset_indicies[k]]
       # print("grahm scan points is",len(Pi))
        
        
       
        subHulls[k] = myGrahmScan(Pi)
    #print("PRINT P",Pi)
       
        
    
    return subHulls

def splitPoints(seq, num):
   # print("seqseq=",seq)
    avgerage = len(seq) / float(num)
    out = []
    last = 0.0
  
    while last < len(seq):
       
        out.append(seq[int(last):int(last + avgerage)])
        last += avgerage
    
    return out

def myGrahmScan(P):
    
    P = np.array(P)
    #print("print p",P)
    # Sorts the points by the y coordinate.
    P = sorted(P,key=lambda x:x[1])
    count1=0
    if len(P) <2:
        count1=count1+1
        #print("this many times grahm scan dint work-",count1)
        return np.array(P)
    
    L_upper = [P[0],P[1]]
    
    for i in range(2,len(P)):
        
        L_upper.append(P[i])
        while len(L_upper) > 2 and not Right_Turn(L_upper[-1],L_upper[-2],L_upper[-3]):
            
            del L_upper[-2]

    L_lower = [P[-1], P[-2]]
    for i in range(len(P)-3,-1,-1):
        
        L_lower.append(P[i])
        while len(L_lower) > 2 and not Right_Turn(L_lower[-1],L_lower[-2],L_lower[-3]):
            
            del L_lower[-2]
    del L_lower[0]
    del L_lower[-1]
    
    try:
        L = np.concatenate((L_upper,L_lower),axis=0)
    except ValueError:
       
        L = L_upper
    
    

    return np.array(L)


def Right_Turn(p1, p2, p3):
        if (p3[1]-p1[1])*(p2[0]-p1[0]) >= (p2[1]-p1[1])*(p3[0]-p1[0]):
                return False
        return True

def orient(p1, p2, p3):
        if (p3[1]-p1[1])*(p2[0]-p1[0]) >= (p2[1]-p1[1])*(p3[0]-p1[0]):
                return True
        return False


def GiftWrapping(G,m):
    
    n = len(G)
    
    P = [[None for x in range(2)] for y in range(n)]
    
    G = sorted(G,key=lambda x:x[0])
    
    pointsOnfinalHull = G[0]
    i = 0
    
    t=0
    while t <=m:  
       
        P[i] = pointsOnfinalHull
        extremepoint = G[0]
        for j in range(1,n):
            
            if (extremepoint[0] == pointsOnfinalHull[0] and extremepoint[1] == pointsOnfinalHull[1]) or not orient(G[j],P[i],extremepoint): # or not means -> if CCW(S[j],P[i],endpoint) false only then go inside 
                extremepoint = G[j]
        i = i + 1
        
        pointsOnfinalHull = extremepoint
        
        if extremepoint[0] == P[0][0] and extremepoint[1] == P[0][1]:
            #print("HElllooo")
            P = [p for p in P if p[0] is not None]
            return np.array(P)
        t=t+1
    

    print("Points on convex hull more than assumed m, hence m is incremented!")
    return False

if __name__ == '__main__':
    main()
