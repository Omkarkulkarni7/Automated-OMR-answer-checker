import cv2
import numpy as np
import utils
#################### Varaibles ####################

path = 'omr4-marked.jpg'
widthImg = 600
heightImg = 600
questions = 6
choices = 5
answer = [0,1,1,3,2,4]

###################################################

# Load the image
img = cv2.imread(path)
img = cv2.resize(img, (widthImg, heightImg))
imgContour = img.copy()
imgBiggestContour = img.copy()

#convert the image to gray
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#blur the image
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) 

#detect edges
imgCanny = cv2.Canny(imgBlur, 10, 50) 

#blank image
imgBlank = np.zeros_like(img)

#find the contours
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgContour, contours, -1, (0, 255, 0), 1)


#fing the area of the contours
rectCon = utils.rectContours(contours)

biggestContour = utils.getCornerPoints(rectCon[1])

#print(biggestContour.shape)

if biggestContour.size != 0:
    cv2.drawContours(imgBiggestContour, biggestContour, -1, (255, 0, 0), 20)
    biggestContour = utils.reorder(biggestContour)
    
    pt1 = np.float32(utils.reorder(biggestContour))
    pt2 = np.float32([[0,0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    
    matrix = cv2.getPerspectiveTransform(pt1, pt2) #get the perspective transform matrix
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) #warp the image
    imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY) #convert the warped image to gray
    imgThresh = cv2.threshold(imgWarpGray, 170, 255, cv2.THRESH_BINARY_INV)[1] #threshold the image
    
    boxes = utils.splitBoxes(imgThresh)
    #cv2.imshow('Split Image', boxes[6])
    #print(cv2.countNonZero(boxes[1]),cv2.countNonZero(boxes[2]))
    
#find the pixel value of each box
    myPixelVal = np.zeros((questions, choices))
    countC = 0
    countR = 0
    
    for image in boxes:
        totalPixels = cv2.countNonZero(image)
        myPixelVal[countR][countC] = totalPixels
        countC += 1
        if countC == choices:
            countR += 1
            countC = 0
            
    #print(myPixelVal)
   
#storing the index of the boxes with the highest pixel value 
    myIndex = []
    
    for x in range(0, questions):
        arr = myPixelVal[x]
        myIndexVal = np.where(arr == np.amax(arr))
        #print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
    print(myIndex)
    
#compare the pixel value with the answer key
    grading = []
    
    for x in range(0, questions):
        if myIndex[x] == answer[x]:
            grading.append(1)
        else:
            grading.append(0)
    print(grading)
    
#find the percentage
    score = (sum(grading)/questions)*100
    print(score)
    


#stakcing all images inside an array
stackImages = utils.stackImages(0.6, ([imgCanny, imgContour, imgBiggestContour], [imgWarpColored, imgWarpGray, imgThresh]))

cv2.imshow('Original Image', stackImages)
cv2.waitKey(0)

