#####################################################################
# Import Libriries
#####################################################################
import cv2 as cv
import numpy as np
import time
import imutils
import datetime
from imutils.video import FPS

#####################################################################
print("\n[INFO] Read frames from Video Stream\n")
#####################################################################
video = cv.VideoCapture(0)

if video.isOpened() == False:
    print("[INFO] Unable to read the video stream")


disp_param = dict(fontFace=cv.FONT_HERSHEY_SIMPLEX)

print("[INFO] Processing...(Press q to stop)")
frameIdx = 0
fps = FPS().start()

while(1):
    # Return Value and the current frame
    ret, frame = video.read()

    #  Check if a current frame actually exist
    if not ret:
        break

    frameIdx += 1
    (H, W) = frame.shape[:2]
    print('[INFO] Resolution: {} x {} pixels'.format(H, W))
    print('\n[INFO] Frame Number: %d' % (frameIdx))

    # frame = imutils.resize(frame, width=400)

    timeStamp = datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p")

    cv.putText(frame, timeStamp, (20, H-20),
               disp_param['fontFace'], 0.45, (0, 255, 0), 1)
    cv.imshow('CCTV ({} x {})'.format(H, W), frame)

    # if the `q` key was pressed, break from the loop
    #####################################################################
    key = cv.waitKey(1) & 0xFF
    if key == ord("q") or key == ord("Q"):
        break

    fps.update()

# Stop the timer and display FPS information
fps.stop()
print("\n[INFO] Elasped time: {:.2f} seconds".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f} ".format(fps.fps()))

print("[INFO] Cleanup")
video.release()
cv.destroyAllWindows()

print("\n[INFO] Done !\n")
