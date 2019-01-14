import cv2
import numpy as np


def make_coordinates(image,line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0])
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters [1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
        
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

#canny, "cannies" an image, meaning that it turns the image to grayscale, smooths out jaged edges,
# then uses .Canny to pull all edges out and represent them with white lines.
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # convert the image to grayscale
    blur = cv2.GaussianBlur(gray, (5,5), 0) #smooth out the image to detect edges better
    canny = cv2.Canny(blur, 50, 150) #creates black and white image. white being the sharp edges or grades of the image
    return canny

#display_lines creates lines that follow along the lanes of a road
def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
            
    return line_image

#region_of_interest creates an area on our image/video that we wish to focus on. 
#will be used to help narrow the area we wish to view to find lane lines
def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200,height),(1100,height), (550,250)]]) #left right cent of polygon
    mask = np.zeros_like(image) #zeros_like is making a black image in the exact pixel height and length as the image passed in
    cv2.fillPoly(mask,polygons,255)#pass' in blank slate, the shape we are projecting, and the black and white gradiant we want it to have
    masked_image = cv2.bitwise_and(image,mask) #overlays the mask on the image
    return masked_image


cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
        _, frame = cap.read()
        canny_image = canny(frame)
        cropped_image = region_of_interest(canny_image)
        lines = cv2.HoughLinesP(cropped_image,2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        averaged_lines = average_slope_intercept(frame,lines)
        line_image = display_lines(frame, averaged_lines)
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        cv2.imshow("result", combo_image) #shows image that was created
        cv2.waitKey(1) # used to display the imshow on our screen