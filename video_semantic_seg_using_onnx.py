# Utility functions
from array import *
# Returns edges detected in an image
def canny_edge_detector(frame):
    
    # Convert to grayscale as only image intensity needed for gradients
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # 5x5 gaussian blur to reduce noise 
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detector with minVal of 50 and maxVal of 150
    canny = cv2.Canny(blur, 50, 150)
    
    return canny  

# Returns a masked image

def ROI_mask(image):
    
	height = image.shape[0]
	width = image.shape[1]

    
    # A triangular polygon to segment the lane area and discarded other irrelevant parts in the image
    # Defined by three (x, y) coordinates    
	polygons = np.array([ 
        [132,720],[192,264],[811,268],[919,720] 
        ]) 
    
	mask = np.zeros_like(image) 
	cv2.fillConvexPoly(mask, polygons, 255)  ## 255 is the mask color
    
    # Bitwise AND between canny image and mask image
	masked_image = cv2.bitwise_and(image, mask)
	#cv2.imwrite("masked_roi.jpg", masked_image)
    
	return masked_image

def get_coordinates (image, params):
     
    slope, intercept = params 
    y1 = image.shape[0]     
    y2 = int(y1 * (3/5)) # Setting y2 at 3/5th from y1
    x1 = int((y1 - intercept) / slope) # Deriving from y = mx + c
    x2 = int((y2 - intercept) / slope) #contours
    
    return np.array([x1, y1, x2, y2])

#Returns averaged lines on left and right sides of the image
def avg_lines(image, lines): 
    
	left = [] 
	right = [] 
    
	for line in lines: 
		x1, y1, x2, y2 = line.reshape(4)
          
		# Fit polynomial, find intercept and slope 
		params = np.polyfit((x1, x2), (y1, y2), 1)  
		slope = params[0] 
		y_intercept = params[1] 
        
		if slope < 0: 
			left.append((slope, y_intercept)) #Negative slope = left lane
			#print("left",left)
		else: 
			right.append((slope, y_intercept)) #Positive slope = right lane
			#print("right",right)
    
    # Avg over all values for a single slope and y-intercept value for each line
    
	left_avg = np.average(left, axis = 0) 
	right_avg = np.average(right, axis = 0) 
    
    # Find x1, y1, x2, y	2 coordinates for left & right lines
	left_line = get_coordinates(image, left_avg) 
	
	right_line = get_coordinates(image, right_avg)
	
	return np.array([left_line, right_line])

# Draws lines of given thickness over an image
def draw_lines(image, lines, thickness): 
   
	# print("lines",lines)	
	
	line_image = np.zeros_like(image)
	color=[0, 0, 255]
    
	# if lines is not None: 
	# 	for x1, y1, x2, y2 in lines:
	
	# 		cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
	


	left_line =[]
	right_line =[]
	left_line.append(lines[0])
	right_line.append(lines[1])
	# print("left_line",left_line)
	# print("right_line",right_line)
	x1 =left_line[0][0]
	y1 = left_line[0][1]
	x2 = left_line[0][2]
	y2 = left_line[0][3]
	x3 = right_line[0][2]
	y3 = right_line[0][3]
	x4 = right_line[0][0]
	y4 = right_line[0][1]

	# roi_drawing_red = np.array([[x1 - 60, y1], [x2-60, y2-140], [x3+60, y3-140], [x4+60, y4]])
	# #return roi_drawing_red
	# print(roi_drawing_red)
	#blue = cv2.polylines(combined_img,[roi_drawing_blue],True,(255,0,0))
	frame_width = 1280
	frame_height = 720






	#left_line = left_line.tolist()

	# print("left_line",left_line[0][0])
	# print(right_line)
	

	# for l in lines :
	# 	left_coor.append(l[0])
	# 	right_coor.append(l[1])
	# print("left_coor",left_coor)
	# print("right_coor",right_coor)

	
	# dynamic_roi_left = []
	# dynamic_roi_right = []
	# for i in left_coor:
	# 	print(i[0])


            
    # Merge the image with drawn lines onto the original.
	combined_image = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)
	# cv2.polylines(combined_image,[roi_drawing_red],True,(0,0,255))
	return combined_image


# from audioop import add
# from ctypes.wintypes import RGB
# from dis import dis
# from types import coroutine
import cv2
import copy
import numpy as np

import time
from topformer import TopFormer
font = cv2.FONT_HERSHEY_TRIPLEX

import matplotlib.pyplot as plt


# Initialize video
# cap = cv2.VideoCapture("input.mp4")


videoUrl = "new.mp4"
#videoPafy = pafy.new(videoUrl)
#print(videoPafy.streams)
cap = cv2.VideoCapture(videoUrl)
start_time = 80 # skip first {start_time} seconds
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time*30)
frame_width = 1280
frame_height = 720

x=0.5  # start point/total width
y=0.8  # start point/total width
threshold = 60  # BINARY threshold
blurValue = 7  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0 
''''''
size = (frame_width, frame_height)
# Initialize semantic segmentator

model_path = "bisenetv2.onnx"
segmentator = TopFormer(model_path)

cv2.namedWindow("Semantic Segmentation", cv2.WINDOW_NORMAL)	
#writer= cv2.VideoWriter('dynamic_roi.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, size)
# out = cv2.VideoWriter('bbb.avi', -1, 20.0, (1280,720))

while cap.isOpened():

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

	try:
		# Read frame from the video
		ret, frame = cap.read()
		frame = cv2.resize(frame, (1280,720))
		
	except:
		continue
	
	# Update semantic segmentator
	seg_map = segmentator(frame)
	#print(type(seg_map))
	# frame1 = np.zeros((1280,720,3), np.uint8)
	# frame1.fill(255)
	combined_img = segmentator.draw_segmentation(seg_map, alpha=0.5)
	
	img = cv2.resize(combined_img, (1280,720))

	canny_edges = canny_edge_detector(img)
	
	cropped_image = ROI_mask(canny_edges)
	
	lines = cv2.HoughLinesP(
    cropped_image,
    rho=2,              #Distance resolution in pixels
    theta=np.pi / 180,  #Angle resolution in radians
    threshold=100,      #Min. number of intersecting points to detect a line  
    lines=np.array([]), #Vector to return start and end points of the lines indicated by [x1, y1, x2, y2] 
    minLineLength=40,   #Line segments shorter than this are rejected
    maxLineGap=25       #Max gap allowed between points on the same line
 )

# Visualisations
	averaged_lines = avg_lines (img, lines)              #Average the Hough lines as left or right lanes
	#print("avg_lines", averaged_lines)
	combined_image = draw_lines(img, averaged_lines, 5)  #Combine the averaged lines on the real frame
	gray = cv2.cvtColor(combined_image, cv2.COLOR_BGR2GRAY)
	ret, thresh = cv2.threshold(cropped_image,threshold, 255, cv2.THRESH_BINARY)
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	c = max(contours, key = cv2.contourArea)
	cv2.drawContours(combined_image, contours, -1, (0,255,0), 3)
	sorted_contours =  sorted(contours, key=cv2.contourArea, reverse=True)
	for i, cont in enumerate(sorted_contours[:-1],2):
		print(i)
		#print(cont)
		cv2.drawContours(combined_image, cont, -1, (0,255,0),3)
	
	result = cv2.VideoWriter('dynamic_roi_hough.mp4', 
                         cv2.VideoWriter_fourcc(*'MP4V'),
                         10, size)
	result = cv2.VideoWriter('track_detection_and_dynamic_roi.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
	out.write(combined_image)

	cv2.imshow("Semantic Sementation", cropped_image)

cap.release()
# out.release()
cv2.destroyAllWindows()



	

	# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	# kernel_size = 5
	# blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)

	# low_threshold = 50
	# high_threshold = 150
	# edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

	# rho = 1  # distance resolution in pixels of the Hough grid
	# theta = np.pi / 180  # angular resolution in radians of the Hough grid
	# threshold = 15  # minimum number of votes (intersections in Hough grid cell)
	# min_line_length = 50  # minimum number of pixels making up a line
	# max_line_gap = 20  # maximum gap in pixels between connectable line segments
	# line_image = np.copy(img) * 0  # creating a blank to draw lines on

	# # Run Hough on edge detected image
	# # Output "lines" is an array containing endpoints of detected line segments
	# lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
	# 					min_line_length, max_line_gap)

	# for line in lines:
	# 	for x1,y1,x2,y2 in line:
	# 		cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

	# lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)

	# ####
	# black = np.zeros((lines_edges.shape[0], lines_edges.shape[1], 3), np.uint8)
	# #blank = np.zeros(img.shape[:2], dtype='uint8')
	# black1 = cv2.rectangle(black,(250,230),
	# 				(850,720),(255, 255, 255), -1)   #---the dimension of the ROI
	# gray = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)               #---converting to gray
	# ret,b_mask = cv2.threshold(gray,127,255, 0)       
	# fin = cv2.bitwise_and(lines_edges,lines_edges,mask = b_mask)
	# fin = cv2.cvtColor(fin, cv2.COLOR_BGR2GRAY)
	# ret, thresh = cv2.threshold(fin,60, 255, cv2.THRESH_BINARY)
	# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# #cv2.drawContours(fin, contours, -1, (0,255,0), 20)
	# sorted_contours =  sorted(contours, key=cv2.contourArea, reverse=True)
	# for i, cont in enumerate(sorted_contours[:1],2):
	# 	cv2.drawContours(fin, cont, 1, (0,255,0),20)
	# black = np.zeros(combined_img.shape[0], combined_img.shape[1], 3), np.uint8
	# black1 = cv2.rectangle(black,(250,230),
    #              (850,720),(255, 255, 255), -1)
	
	# gray = cv2.cvtColor(combined_img, cv2.COLOR_BGR2GRAY)
	# blur = cv2.blur(gray, (10,10),0)
	# ret,b_mask = cv2.threshold(gray,127,255, 0)         
	# #roi =cv2.rectangle(combined_img, (250,230),
    #             #   (850,720), (255, 0, 0), 2)
	# fin = cv2.bitwise_and(combined_img,black,mask = black1)
	# ret, thresh = cv2.threshold(blur,threshold, 255, cv2.THRESH_BINARY)
	# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# cv2.drawContours(combined_img, contours, -1, (0,255,0), 20)
	# mask = np.zeros(roi.shape,np.uint8)
	
				   #---the dimension of the ROI
		
	# gray = cv2.cvtColor(black,cv2.COLOR_BGR2GRAY)               #---converting to gray
	# ret,b_mask = cv2.threshold(gray,127,255, 0)            
	# roi =cv2.rectangle(combined_img, (250,230),
    #              (850,720), (255, 0, 0), 2) #drawing ROI
	# gray = cv2.cvtColor(combined_img, cv2.COLOR_BGR2GRAY)
	# #blur image to reduce the noise in the image while thresholding. #This smoothens the sharp edges in the image.
	# blur = cv2.blur(gray, (10,10),0)
	# #Apply thresholding to the image
	# mask = np.zeros(roi.shape,np.uint8)
	# ret, thresh = cv2.threshold(blur,threshold, 255, cv2.THRESH_BINARY)
	# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# cv2.drawContours(mask, contours, -1, (0,255,0), 3)
	# thresh1 = copy.deepcopy(thresh)
	# #find the contours in the image
	# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	# length = len(contours)
	# #print(length)
	# maxArea = -1
	# if length > 0:
	# 	for i in range(length):  # find the biggest contour (according to area)
	# 		temp = contours[i]
	# 		area = cv2.contourArea(temp)
	# 		if area > maxArea:
	# 			maxArea = area
	# 			ci = i

	# 	res = contours[ci]
	# 	hull = cv2.convexHull(res) #applying convex hull technique
	# 	drawing = np.zeros(combined_img.shape, np.uint8)
	# 	cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2) #drawing contours 
	# 	cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3) #drawing convex hull
	#draw the obtained contour lines(or the set of coordinates forming a line) on the original image
	#cv2.drawContours(combined_img, contours, -1, (0,255,0), 20)
	#print(combined_img.shape)
	# sorted_contours =  sorted(contours, key=cv2.contourArea, reverse=True)
	# for i, cont in enumerate(sorted_contours[:1],2):
	# 	cv2.drawContours(combined_img, cont, 1, (0,255,0),20)
		#print(cont)
		#print(cont.shape)
	# list_1_3d=[]
	# for i in cont:
	# 	j=i.tolist()
	# 	list_1_3d.append(j)
	# #print(k)
	# print("contours:", list_1_3d)
	# print("len:",len(list_1_3d))
	# print("first coor", list_1_3d[0])
	# list_2_2d=[]
	# for t in list_1_3d:
	# 	for f in t:
	# 		list_2_2d.append(f)
	# print("each coordinate in contour:", list_2_2d)
	# list_3_x=[]
	# list_3_y=[]
	# for c in list_2_2d:
	# 	list_3_x.append(c[0])
	# 	list_3_y.append(c[1])
	# print("x-coordinate:", list_3_x)
	# print("y-coordinate:",list_3_y)
	# sorted_x=sorted(list_3_x)
	# sorted_y=sorted(list_3_y)
	# print("sorted x coor:",sorted_x)
	# print("sorted y coor:", sorted_y)
	# #print(x)
	# #print(y)
	# #print(x)
	# #print(y)
	# # xst=sum(sorted_x[0:50])/100
	# # yst=sum(sorted_y[0:50])/100
	# # print("to restrict x start points other than around 100 points:",xst)
	# # print("to restrict y start points other than around 100 points:",yst)
	# # xend=sum(sorted_y[-50:-1])/100
	# # yend=sum(sorted_y[-50:-1])/100
	# # print("to restrict x end points other than around 100 points:",xend)
	# # print("to restrict y end points other than around 100 points:",yend)
	# # print("first coor", list_1_3d[0])
	# x1 = 454
	# x2 = 683
	# x3 = 1000
	# x4 = 1186
	# y1 = 1073
	# y2 = 599
	# y3 = 596
	# y4 = 1067
	# ##########
	# x5 = 683
	# x6 = 757
	# x7 = 952
	# x8 = 1000
	# y5 = 599
	# y6 = 452
	# y7 = 456
	# y8 = 596

	
	# #roi_drawing_red = np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]])
	# #roi_drawing_blue = np.array([[x5,y5], [x6,y6], [x7,y7],[x8,y8]])
	# # roi_drawing_red = np.array([[454,1073], [683,596], [1000,596], [1186,1073]])
	# # roi_drawing_blue = np.array([[683,599], [757,456], [952,456],[1000,599]])
	# color =RGB(255,255,0)
	# # #pts = pts.reshape((-1,1,2))
	# #red =cv2.polylines(combined_img,[roi_drawing_red],True,(0,0,255))
	# #blue = cv2.polylines(combined_img,[roi_drawing_blue],True,color)
	# x1n=[]
	# for m,n in zip(list_3_x,list_3_y):
	# 	if n==1073:
	# 		x1n.append(m)
	# print("x1n",x1n)

	# x2n=[]
	# for m,n in zip(list_3_x,list_3_y):
	# 	if n==596:
	# 		x2n.append(m)
	# print("x2n",x2n)
	# x3n=[]
	# for m,n in zip(list_3_x,list_3_y):
	# 	if n==599:	
	# 		x3n.append(m)
	# print("x3n", x3n)
	# x4n=[]
	# for m,n in zip(list_3_x,list_3_y):
	# 	if n==1067:
	# 		x4n.append(m)
	# print("x4n", x4n)
	# newa=[]
	# for c in x2n:
	# 	if x2-30<=c<=x2+30:
	# 		newa.append(c)

	# print("newa:", newa)
	# newb = []
	# for c in x3n:
	# 	if x3-50<=c<=x3+50:
	# 		newb.append(c)
	# print("newb:", newb)
	# newcx1 = []
	# for c in x1n:
	# 	if x1-50<=c<=x1+50:
	# 		newcx1.append(c)
	# print("newcx1:", newcx1)
	# newdx4= []
	# for c in x4n:
	# 	if x4-50<=c<=x4+50:
	# 		newdx4.append(c)
	# print("newdx4:", newdx4)
	# # newot1 = []
	# # for c in x2n:
	# # 	if  x3 - 50 <= c <= x3 + 50:
	# # 		newot1.append(c)
	# # newb = []
	# # for c in x3n:
	# #     if xvsd - 100 <= c <= xvsd + 100 or xvsf - 100 <= c <= xvsf + 100:
	# #         newb.append(c)
	# #print(newa)
	# # while True:
	
	# # 	if x2 - 50 <= min(newa) <= x2 + 50:
	# # 			x2 = min(newa)
	# # 	if x3 - 50 <= min(newb) <= x3 + 50:
	# # 			x3 = min(newb)
	# try:
	# 	if x1 + 30<= max(newcx1) :

	# 			x1 = max(newcx1)
	# 	else:
	# 		x1 = min(newcx1)

				
	# except:
	# 		pass
	# try:
	# 	if x2  - 30>= min(newa) :

	# 			x2 = min(newa)
	# 	else:
	# 		x2 = max(newa)

				
	# except:
	# 		pass
	
	# try:
	# 	if x4  - 30>= min(newdx4) :

	# 			x4 = min(newdx4)
	# 	else:
	# 		x4 = max(newdx4)

				
	# except:
	# 		pass


	# try:
	# 	if x3  + 30 <=max(newb) :
	# 		x3 = max(newb)
	# 	else:
	# 		x3 =min(newb)

	# except:
	# 	pass
		
	# x6n=[]
	# for m,n in zip(list_3_x,list_3_y):
	# 	if n==452:
	# 		x6n.append(m)
	# print("x6n",x6n)
	# x7n=[]
	# for m,n in zip(list_3_x,list_3_y):
	# 	if n==456:
	# 		x7n.append(m)
	# print("x7n",x7n)
	# x5n=[]
	# for m,n in zip(list_3_x,list_3_y):
	# 	if n==599:
	# 		x5n.append(m)
	# print("x5n",x5n)
	# x8n=[]
	# for m,n in zip(list_3_x,list_3_y):
	# 	if n==596:
	# 		x8n.append(m)
	# print("x8n",x8n)
	# newbx6=[]
	# for k in x6n:
	# 	if x6-50<=k<=x6+50:
	# 		newbx6.append(k)

	# print(newbx6)
	
	# newbx7 = []
	# for k in x7n:
	# 	if x7 - 50 <= k <= x7 + 50 :
	# 		newbx7.append(k)
	
	# newbx5=[]
	# for k in x5n:
	# 	if x5-50<=k<=x5+50:
	# 		newbx5.append(k)

	# print(newbx5)
	
	# newbx8 = []
	# for k in x8n:
	# 	if x8 - 50 <= k <= x8 + 50 :
	# 		newbx8.append(k)
	# # newb = []
	# # for c in x3n:
	# #     if xvsd - 100 <= c <= xvsd + 100 or xvsf - 100 <= c <= xvsf + 100:
	# #         newb.append(c)
	# #print(newa)
	# try:
	# 	if x6 +30<= max(newbx6) :

	# 			x6 = max(newbx6)
	# 	else:
	# 		x6 = min(newbx6)

	# except:
	# 	pass
	# try:
	# 	if x7 - 30 >= max(newbx7) :
	# 		x7 = max(newbx7)
	# 	else:
	# 		x7 =min(newbx7)

	# except:
	# 	pass
	# try:
	# 	if x8 -30<= max(newbx8) :

	# 			x8 = max(newbx8)
	# 	else:
	# 		x8 = min(newbx8)

	# except:
	# 	pass
	# try:
	# 	if x5 + 30 >= max(newbx5) :
	# 		x5 = max(newbx5)
	# 	else:
	# 		x5 =min(newbx5)

	# except:
	# 	pass

	


	
		
	# #red_coor = [x1,x2,x3,x4]
	# #blue_coor = [x5,x6,x7,x8]
	# roi_drawing_red = np.array([[x1, 1073], [x2, 596], [x3, 599], [x4, 1067]])
	# #roi_drawing_blue = np.array([[683,599 ], [x6, 452], [x7, 456], [1000, 596]])  
	# #roi_drawing_red = np.array([[454,1073], [683,596], [1000,596], [1186,1073]])
	# #print("roi_drawing_red:", roi_drawing_red) 
	# red =cv2.polylines(combined_img,[roi_drawing_red],True,(0,0,255))
	# #blue = cv2.polylines(combined_img,[roi_drawing_blue],True,(255,0,0))
	# frame_width = 1280
	# frame_height = 720
   
	# size = (frame_width, frame_height)
	# result = cv2.VideoWriter('filename.mp4', 
    #                      cv2.VideoWriter_fourcc(*'MJPG'),
    #                      10,10,size)
	#cv2.imshow("Semantic Sementation", combined_img)
	#cv2.polylines(overlay, [roi_drawing_red], True, (0, 0, 255))
	# #cv2.fillPoly(combined_img, pts = [roi_drawing], color =(255,))
	# imgray = cv2.cvtColor(combined_img, cv2.COLOR_BGR2GRAY) 
	# edged = cv2.Canny(imgray, 30, 200)
	# #ret, thresh = cv2.threshold(imgray, 127,255,0)
	# #print(combined_img.shape)
	# #red_contours = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	# #blue_contours = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
	# contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	
	#print(len(sorted_contours))
	
		#cont.transpose(2,0,1).reshape(3,-1)
		# import numpy
		# arr = numpy.array(
		# 	cont
		# 	)
		# newarr = arr.reshape(arr.shape[0],2)
		# #newarr =newarr + [[40 40]]
		# print(newarr)
		# print(newarr.shape)
		# roi_drawing_dummy = np.array(newarr)
		# dummy = cv2.polylines(combined_img,[roi_drawing_dummy],True,(255,255,255))

		#(2251, 1, 2)

	
	#(1080, 1920, 3)

	
	# INPUT_FILE=combined_img
	# #OUTPUT_FILE='rail_marking/segmentation/deploy/darknet/predicted.jpg'
	# LABELS_FILE='darknet/data/coco.names'
	# CONFIG_FILE='darknet/cfg/yolov3.cfg'
	# WEIGHTS_FILE='darknet/yolov3.weights'
	# CONFIDENCE_THRESHOLD=0.75

	# LABELS = open(LABELS_FILE).read().strip().split("\n")

	# np.random.seed(4)
	# COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	# 	dtype="uint8")


	# net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)

	# image = combined_img
	# (H, W) = combined_img.shape[:2]

	# # determine only the *output* layer names that we need from YOLO
	# ln = net.getLayerNames()
	# ln = [ln[i-1] for i in net.getUnconnectedOutLayers()]


	# blob = cv2.dnn.blobFromImage(combined_img, 1 / 255.0, (416, 416),
	# 	swapRB=True, crop=False)
	# net.setInput(blob)
	# start = time.time()
	# #print(ln)
	# layerOutputs = net.forward(ln)
	# end = time.time()


	# print("[INFO] YOLO took {:.6f} seconds".format(end - start))


	# # initialize our lists of detected bounding boxes, confidences, and
	# # class IDs, respectively
	# boxes = []
	# confidences = []
	# classIDs = []

	# # loop over each of the layer outputs
	# for output in layerOutputs:
	# 	# loop over each of the detections
	# 	for detection in output:
	# 		# extract the class ID and confidence (i.e., probability) of
	# 		# the current object detection
	# 		scores = detection[5:]
	# 		classID = np.argmax(scores)
	# 		confidence = scores[classID]

	# 		# filter out weak predictions by ensuring the detected
	# 		# probability is greater than the minimum probability
	# 		if confidence > CONFIDENCE_THRESHOLD:
	# 			# scale the bounding box coordinates back relative to the
	# 			# size of the image, keeping in mind that YOLO actually
	# 			# returns the center (x, y)-coordinates of the bounding
	# 			# box followed by the boxes' width and height
	# 			box = detection[0:4] * np.array([W, H, W, H])
	# 			(centerX, centerY, width, height) = box.astype("int")

	# 			# use the center (x, y)-coordinates to derive the top and
	# 			# and left corner of the bounding box
	# 			x = int(centerX - (width / 2))
	# 			y = int(centerY - (height / 2))

	# 			# update our list of bounding box coordinates, confidences,
	# 			# and class IDs
	# 			boxes.append([x, y, int(width), int(height)])
	# 			confidences.append(float(confidence))
	# 			classIDs.append(classID)

	# # apply non-maxima suppression to suppress weak, overlapping bounding
	# # boxes
	# idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD,
	# 	CONFIDENCE_THRESHOLD)

	# # ensure at least one detection exists
	# if len(idxs) > 0:
	# 	# loop over the indexes we are keeping
	# 	for i in idxs.flatten():
	# 		# extract the bounding box coordinates
	# 		(x, y) = (boxes[i][0], boxes[i][1])
	# 		(w, h) = (boxes[i][2], boxes[i][3])

	# 		color = [int(c) for c in COLORS[classIDs[i]]]

	# 		cv2.rectangle(combined_img, (x, y), (x + w, y + h), color, 2)
	# 		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
	# 		cv2.putText(combined_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
	# 			0.5, (255,255,255), 2)
	# 		dist1 = cv2.pointPolygonTest(roi_drawing_red,(x,y),False)

			#dist2 = cv2.pointPolygonTest(roi_drawing_blue,(x,y),False)

			#src = (np.zeros((image.height, width), dtype=np.uint8), np.zeros((height, width), dtype=np.uint8))
			#Create a sequence of points to make a contour
			#verts = [[(280, 720), (500, 280), (750, 280), (950, 720)],
			#  				[(500, 280), (550, 180), (700, 180), (750, 280)]]

			# # # Draw it in src
			# for vert, src in zip(verts, src):
			# 	for index, i in enumerate(vert):
			# 		cv2.line(src, vert[index], vert[(index + 1) % len(vert)], (255), 3)


			
			# if dist1 == +1 : cv2.putText(combined_img,'Brake',(830, 86), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
			#if dist2 == +1: cv2.putText(combined_img,'Horn',(830, 86), font, 4, (255, 255, 255), 2, cv2.LINE_AA)
			# if dist1 + dist2 == -2: cv2.putText(image,'Outside',(830, 86), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

	

            		
	
#		cv2.putText(combined_img, str(i), (cont[0,0,0], cont[0,0,1]-10), cv2.FONT_HERSHEY_SIMPLEX,1.4,(0,255,0),4)
	# 	#result.write(combined_img)
	# 	#cv2.imshow("image",combined_img)

	# # dist1 = cv2.pointPolygonTest(cont, (1110, 476), True)
	# # if dist1 > 3:
	# # 	print("True")
	# # else:
	# # 	print("False")

    #cv2.imshow("Semantic Sementation", combined_img)
