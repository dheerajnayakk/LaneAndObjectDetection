from ultralytics import YOLO
import cv2
import numpy as np
from Finalutils import *

# YOLOv8 model initialization
model = YOLO("yolov8n.pt")  

# Video capture (0 for webcam or video file path)
cap = cv2.VideoCapture("yellow_lane_video.mp4")

# Video frame properties
frameWidth = 640
frameHeight = 480

# Trackbar initialization for lane detection
initialTracbarVals = [42, 63, 14, 87]
initializeTrackbars(initialTracbarVals)

# Array for storing curve values
arrayCurve = np.zeros([10])
arrayCounter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video stream")
        break

    img = cv2.resize(frame, (frameWidth,frameHeight))
    imgWarpPoints = img.copy()
    imgFinal = img.copy()
    imgCanny = img.copy()

    imgUndis = undistort(img)
    imgThres, imgCanny, imgColor = thresholding(img)
    src = valTrackbars()
    imgWarp = perspective_warp(imgThres, dst_size=(frameWidth, frameHeight), src=src)
    imgWarpPoints = drawPoints(imgWarpPoints, src)
    imgSliding, curves, lanes, ploty = sliding_window(imgWarp, draw_windows=True)
    
    lane_curve=0
    try:
        curverad = get_curve(imgFinal, curves[0], curves[1])
        lane_curve = np.mean([curverad[0], curverad[1]])
        imgFinal = draw_lanes(img, curves[0], curves[1], frameWidth, frameHeight, src=src)

        # Average Curve Calculation
        currentCurve = lane_curve // 50
        if int(np.sum(arrayCurve)) == 0:
            averageCurve = currentCurve
        else:
            averageCurve = np.sum(arrayCurve) // arrayCurve.shape[0]
        if abs(averageCurve - currentCurve) > 200:
            arrayCurve[arrayCounter] = averageCurve
        else:
            arrayCurve[arrayCounter] = currentCurve
        arrayCounter += 1
        if arrayCounter >= 10:
            arrayCounter = 0
    except Exception as e:
        print(f"Lane detection error: {e}")
        pass


    imgFinal = drawLines(imgFinal, lane_curve)
    

    # Object Detection with YOLOv8
    results = model(frame)
    annotated_frame = results[0].plot()  # Draw YOLO results on frame
    object_frame = cv2.resize(annotated_frame, (640,500))
    
    # Overlaying the outputs
    imgStacked = stackImages(0.7, ([imgColor, imgCanny],
                                   [imgWarp, imgSliding]))
    
    
    #resizing the Frames
    laneFrame = cv2.resize(imgFinal,(640,480))
    pipelineFrame = cv2.resize(imgStacked,(640,640))

    # Display the outputs
    cv2.imshow("YOLOv8 Object Detection", object_frame)
    cv2.imshow("Lane Detection", laneFrame)
    #cv2.imshow("Combined Pipeline", pipelineFrame)

    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()