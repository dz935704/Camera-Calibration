

# import the necessary packages
from imutils import paths
import numpy as np
import imutils
import cv2
import glob
def find_marker(image):
	# convert the image to grayscale, blur it, and detect edges
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 35, 125)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key = cv2.contourArea)
	# compute the bounding box of the of the paper region and return it
	return cv2.minAreaRect(c)



def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth



# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 24.0
# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 11.0
# load the furst image that contains an object that is KNOWN TO BE 2 feet
# from our camera, then find the paper marker in the image, and initialize
# the focal length
image = cv2.imread("images/2ft.png")
marker = find_marker(image)
focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH


images = glob.glob('images/*.png')

for image in images:
# loop over the images
#for imagePath in sorted(paths.list_images("images")):
	# load the image, find the marker in the image, then compute the
	# distance to the marker from the camera
	image = cv2.imread(image)
	marker = find_marker(image)
	inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
	# draw a bounding box around the image and display it
	box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
	box = np.int0(box)
	cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
	cv2.putText(image, "%.2fft" % (inches / 12),
		(image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
		2.0, (0, 255, 0), 3)
	cv2.imshow("image", image)
	cv2.waitKey(0)
	




# # install opencv "pip install opencv-python"
# import cv2
  
# # distance from camera to object(face) measured
# # centimeter
# Known_distance = 76.2
  
# # width of face in the real world or Object Plane
# # centimeter
# Known_width = 14.3
  
# # Colors
# GREEN = (0, 255, 0)
# RED = (0, 0, 255)
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
  
# # defining the fonts
# fonts = cv2.FONT_HERSHEY_COMPLEX
  
# # face detector object
# face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
  
# # focal length finder function
# def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
  
#     # finding the focal length
#     focal_length = (width_in_rf_image * measured_distance) / real_width
#     return focal_length
  
# # distance estimation function
# def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
  
#     distance = (real_face_width * Focal_Length)/face_width_in_frame
  
#     # return the distance
#     return distance
  
  
# def face_data(image):
  
#     face_width = 0  # making face width to zero
  
#     # converting color image to gray scale image
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
#     # detecting face in the image
#     faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
  
#     # looping through the faces detect in the image
#     # getting coordinates x, y , width and height
#     for (x, y, h, w) in faces:
  
#         # draw the rectangle on the face
#         cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)
  
#         # getting face width in the pixels
#         face_width = w
  
#     # return the face width in pixel
#     return face_width
  
  
# # reading reference_image from directory
# ref_image = cv2.imread("Ref_image.png")
  
# # find the face width(pixels) in the reference_image
# ref_image_face_width = face_data(ref_image)
  
# # get the focal by calling "Focal_Length_Finder"
# # face width in reference(pixels),
# # Known_distance(centimeters),
# # known_width(centimeters)
# Focal_length_found = Focal_Length_Finder(
#     Known_distance, Known_width, ref_image_face_width)
  
# print(Focal_length_found)
  
# # show the reference image
# cv2.imshow("ref_image", ref_image)
  
# # initialize the camera object so that we
# # can get frame from it
# cap = cv2.VideoCapture(0)
  
# # looping through frame, incoming from 
# # camera/video
# while True:
  
#     # reading the frame from camera
#     _, frame = cap.read()
  
#     # calling face_data function to find
#     # the width of face(pixels) in the frame
#     face_width_in_frame = face_data(frame)
  
#     # check if the face is zero then not 
#     # find the distance
#     if face_width_in_frame != 0:
        
#         # finding the distance by calling function 
#         # Distance finder function need 
#         # these arguments the Focal_Length,
#         # Known_width(centimeters),
#         # and Known_distance(centimeters)
#         Distance = Distance_finder(
#             Focal_length_found, Known_width, face_width_in_frame)
  
#         # draw line as background of text
#         cv2.line(frame, (30, 30), (230, 30), RED, 32)
#         cv2.line(frame, (30, 30), (230, 30), BLACK, 28)
  
#         # Drawing Text on the screen
#         cv2.putText(
#             frame, f"Distance: {round(Distance,2)} CM", (30, 35), 
#           fonts, 0.6, GREEN, 2)
  
#     # show the frame on the screen
#     cv2.imshow("frame", frame)
  
#     # quit the program if you press 'q' on keyboard
#     if cv2.waitKey(1) == ord("q"):
#         break
  
# # closing the camera
# cap.release()
  
# # closing the windows that are opened
# cv2.destroyAllWindows()