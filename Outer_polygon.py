# import the libraries
import cv2
import numpy as np
import math

# array for saving the selected coordinates
array = np.array([[0,0]])

counter = 0

# angle finder that calculates the external corner based on three inner points
def angle(x1,y1,x2,y2,x3,y3):
    # distance of the outer polygon from the inner polygon
    distance = 50
    # dot and cross products of vector1  with  [1,0]  vector
    dot_value1 = (x1-x2) 
    cross_value1 = (y1-y2)
    # norm of the vector1 and [1,0] vector.
    norm11 = ((x1-x2)**2 + (y1-y2)**2)**(0.5)
    norm21 = 1
    # finding the quadrant based on the cross and dot products of vector 1 with [1,0] vector
    if cross_value1 >  0 :
        theta = math.acos(dot_value1/(norm11*norm21))
    elif cross_value1 < 0 :
        theta = - math.acos(dot_value1/(norm11*norm21))
    elif cross_value1 == 0:
      if (x1 - x2)  > 0:
        theta = 0
      if (x1 - x2) < 0:
        theta = math.pi
    # cross and dot products of vector1 and vector2
    dot_value = (x1-x2)*(x3-x2) + (y1-y2)*(y3-y2)
    cross_value = (x1-x2)*(y3-y2) - (y1-y2)*(x3-x2)
    # norm of vector 1 and vector 2
    norm1 = ((x1-x2)**2 + (y1-y2)**2)**(0.5)
    norm2 = ((x3-x2)**2 + (y3-y2)**2)**(0.5)


    # finding the external angle based on cross and dot products
    if cross_value >  0 :
        inner_theta = math.acos(dot_value/(norm1*norm2))
    elif cross_value < 0 :
        inner_theta = 2*math.pi - math.acos(dot_value/(norm1*norm2))

    # finding the bisection of the angle between vector1 and vector 2 
    if inner_theta < math.pi : 
      bi_theta = inner_theta/2.0
      x_inner = distance/math.tan(bi_theta)
      y_inner = distance

    elif inner_theta > math.pi : 
      bi_theta = (inner_theta - math.pi)/2.0
      x_inner = - distance*math.tan(bi_theta)
      y_inner = distance

    # coordinate transform to get the x,y positons relative to point 2 based on [1,0] and [0,1] basis
    x_inner_trans = math.cos(theta)*x_inner - math.sin(theta)*y_inner
    y_inner_trans = math.sin(theta)*x_inner + math.cos(theta)*y_inner
    # shifting with respect to point 2
    x_outer = x_inner_trans + x2
    y_outer = y_inner_trans + y2

    return round(x_outer,0),round(y_outer,0)

# mouse click routine to find the coordinates of clicked position

def draw_circle(event,x,y,flags,param):
    global counter
    counter = 1
    global mouseX,mouseY
    global array
    if event == cv2.EVENT_LBUTTONDBLCLK:

        mouseX,mouseY = x,y
        array = np.vstack((array,[mouseX,mouseY]))


# window to display the mouse clicks
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

switcher = False
cap = cv2.VideoCapture(0)


# loop for video input which would initiate feed
while(True):
    ret, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if counter == 1:   
        for xx in array[1:]:
            x = xx[0]
            y = xx[1]
            cv2.circle(img,(x,y),2,(255,0,0),-1)
            cv2.putText(img,"{},{}".format(x,y),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    
    k = cv2.waitKey(20) & 0xFF

    # escape sequence : breaks on Esc key and draws on "a" key
    if k == 27:
        print(array)
        break
    elif k == ord('a'):
        print(array[1:])
        switcher =  not switcher
    
    # connecting the lines to get the polygons
    pts = np.array(array[1:], np.int32)
    pts = pts.reshape((-1,1,2))
    points = array[1:]
    if len(points) >= 3:
        points = np.vstack((array[1:],array[1],array[2]))
        pts1 = []
        for i in range(len(array[1:])):
            x1,y1 = points[i]
            x2,y2 = points[i+1]
            x3,y3 = points[i+2]
            nn = angle(x1,y1,x2,y2,x3,y3)
            pts1.append(nn)
    # Switch to activate the connection of points to form polygons
    if switcher:
        cv2.polylines(img,[np.array(pts1,dtype= np.int32)],True,(0,255,0),3)
        cv2.polylines(img,[pts],True,(0,0,255),3)
    # showiung the image
    cv2.imshow('image',img)

# release the resources and destroy all windows.
cap.release()
cv2.destroyAllWindows()


    

