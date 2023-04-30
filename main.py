import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

'''
Here's what's happening in the code:

1. We import the necessary libraries, including OpenCV.
2. We create a CascadeClassifier object and load the XML file that contains the face detection algorithm.
3. We create a VideoCapture object to access the default camera (or a specified camera).
4. We start a while loop that continuously captures frames from the camera and processes them for face detection.
5. Inside the loop, we use the detectMultiScale() method to detect faces in the captured frames. 
6. The method takes in the grayscale version of the image, a scale factor, 
and a minimum number of neighbors required for a positive detection.
7. If a face is detected, we draw a rectangle around it using the rectangle() method.
8. We display the processed image using the imshow() method, and wait for a key press to exit the program.
9. Once the program is exited, we release the camera and destroy all OpenCV windows.

Note that in order for the face detection to work properly, you'll need to download the 
haarcascade_frontalface_default.xml file from OpenCV's GitHub repository and place it in the same directory as your Python script.
'''