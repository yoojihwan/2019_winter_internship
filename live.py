# This script will detect faces via your webcam.
# Tested with OpenCV3

import cv2
from datetime import datetime

cap = cv2.VideoCapture(0)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect faces in the image
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)

	print("Found {0} faces!".format(len(faces)))

	if len(faces) > 0:
		for (x, y, w, h) in faces:
			cropped = frame[y - int(h / 10):y + h + int(h/10), x - int(w / 10):x + w + int(w / 10)]
			cv2.imwrite("./face_img/" + str(datetime.now()) + ".png", cropped)
		break

	# Display the resulting frame
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cv2.imshow('cropped', cropped)
cv2.waitKey(0)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
