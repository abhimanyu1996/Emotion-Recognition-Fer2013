
import sys

if len(sys.argv) < 2:
    print("No Input File given")
    sys.exit()
elif len(sys.argv) == 2:
    input_file = sys.argv[1]
    out_file = "outVideo.mp4"
else:
    input_file = sys.argv[1]
    out_file = sys.argv[2] + ".mp4"
    
import os

if not os.path.exists(input_file):
    print('Input file does not exist')
    sys.exit()
    
from tensorflow.keras.models import load_model
import cv2
import numpy as np

img_size = 48
emotion_to_str = {0:"ANGRY", 1:"DISGUST", 2:"FEAR",
                  3:"HAPPY", 4:"SAD", 5:"SURPRISE", 6:"NEUTRAL"}

model = load_model('content/model.h5')
model.load_weights("content/weights.best.hdf5")

face_cascade = cv2.CascadeClassifier('content/haar.xml')

def detect_image_emotion(img):
  grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(grayImg,1.3,5)

  imgcopy = img.copy()

  for (x,y,w,h) in faces: 

    facearray_gray = grayImg[y:y+h, x:x+w]

    width_original = facearray_gray.shape[1]   
    height_original = facearray_gray.shape[0]  

    faceimg_gray = cv2.resize(facearray_gray, (img_size, img_size))   
    faceimg_gray = faceimg_gray/255.

    faceimg_model = np.reshape(faceimg_gray, (1,img_size,img_size,1)) 
    keypoints = model.predict(faceimg_model)[0]

    rectangle_bgr = (0, 0, 255)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (x,y)
    fontScale              = 0.7
    fontColor              = (0,0,0)
    thickness               = 2

    text = emotion_to_str[np.argmax(keypoints)]
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=fontScale, thickness=thickness)[0]
    text_offset_x = x
    text_offset_y = y
    box_coords = ((text_offset_x, text_offset_y), 
                  (text_offset_x + text_width, text_offset_y - text_height ))

    cv2.rectangle(imgcopy,(x,y),(x+h,y+w),rectangle_bgr,5)
    cv2.rectangle(imgcopy, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(imgcopy, text, (text_offset_x, text_offset_y), font, 
                fontScale=fontScale, color=fontColor, thickness=thickness)

  return imgcopy
  
  
cap = cv2.VideoCapture(input_file)

ret, frame = cap.read()
video_shape = (int(cap.get(3)), int(cap.get(4)))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(out_file,fourcc, 20.0, video_shape, True)

while ret:
  predict_image = detect_image_emotion(frame)
  out.write(predict_image)
  ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()

print(out_file + " created")

