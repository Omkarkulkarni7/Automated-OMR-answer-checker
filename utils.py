import cv2
import numpy as np

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
 


def rectContours(contours):
    
    rectCon = []
    
    for i in contours:
        #print("Contour", i)
        area = cv2.contourArea(i)
        #print("Area",area)
        if area>20:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*peri, True) #approximate a shape with a polygon with 0.02*peri accuracy. returns the corner points
            
            #print("Corner Points", len(approx))
            
            if len(approx) == 4:
                rectCon.append(i)
    
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    
    return rectCon

def getCornerPoints(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02*peri, True) #approximate a shape with a polygon with 0.02*peri accuracy. returns the corner points
    return approx

def reorder(myPoints):
    myPoints = myPoints.reshape(4,2) # reshape the img to 4x2
    myPointsNew = np.zeros((4,1,2), np.int32) # create a new array with 4x1x2
    add = myPoints.sum(1) # sum of x and y
    
    #print(myPoints)
    #print(add)
    
    myPointsNew[0] = myPoints[np.argmin(add)] # 0,0
    myPointsNew[3] = myPoints[np.argmax(add)] # w,h
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)] # w,0
    myPointsNew[2] = myPoints[np.argmax(diff)] # 0,h
    
    #print(myPointsNew)
    
    return myPointsNew

def splitBoxes(img):
    rows = np.vsplit(img, 6)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 5)
        for box in cols:
            boxes.append(box)
    return boxes

