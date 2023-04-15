import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

with np.load('B.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    print(mtx)


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((7*10,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, (int(corner[0]), int(corner[1])), (int(tuple(imgpts[0].ravel())[0]), int(tuple(imgpts[0].ravel())[1])), (255,0,0), 5)
    img = cv.line(img, (int(corner[0]), int(corner[1])), (int(tuple(imgpts[1].ravel())[0]), int(tuple(imgpts[1].ravel())[1])), (0,255,0), 5)
    img = cv.line(img, (int(corner[0]), int(corner[1])), (int(tuple(imgpts[2].ravel())[0]), int(tuple(imgpts[2].ravel())[1])), (0,0,255), 5)
    return img


def randrange(n, vmin, vmax):
            return (vmax - vmin)*np.random.rand(n) + vmin

import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

np.random.seed(19680801)

        
            
fig = plt.figure()
ax = fig.add_subplot(projection='3d')


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    #cv.imshow('frame', gray)


    success, corners = cv.findChessboardCorners(gray, (10,7),None) #new line
    while success == True: #new conditional
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
    
        ax.scatter(tvecs[0], tvecs[1], tvecs[2], c='red', marker='.', s=10)
        plt.show()



        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        success, corners = cv.findChessboardCorners(gray, (10,7),None)
        
    # else:
    #     cv.imshow('img',frame)
    #     cv.waitKey(1)

        if cv.waitKey(1) == ord('q'):
            break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()