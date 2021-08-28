import cv2
from random import randrange

#load some pre_trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier("C:\haarcascade_frontalface_default.xml")

# choose an image to detect faces in
#img = cv2.imread("photo.png")
# To capture video from webcam
webcam = cv2.VideoCapture(0) # 0 is your default webcam, also you can put a video path instead of webcam
key = cv2.waitKey(1)

# Iterate forever over frames
while True:
  ### Read the current frame
  successful_frame_read, frame = webcam.read()

  # Must convert to grayscale
  grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #in opencv RGB is BGR

  # Detect Faces
  #Detect objects of different sizes in the input image (in our case face image and objects are like noise, lips & ...).
  #The detected objects are returned as list of rectangles.
  face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

  # Draw rectangles around the faces
  for (x, y, w, h) in face_coordinates:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

  cv2.imshow('Razie Face Detector',  frame)
  cv2.waitKey(1)

  ### Stop if Q key is pressed
  if key == 81 or key == 113:
    break

  # release the VideoCapture object
  #webcam.release()
