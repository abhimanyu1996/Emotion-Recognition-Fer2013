# Emotion-Recognition-Fer2013

### Description
This project uses images/videos as input and outputs the emotion of people in the image/video. There are 7 different emotions "Angry", "Disgust", "Scared", "Happy", "Sad", "Surprised", "Neutral".

<br>
![Demo Image](content/Demo2.png)

### Dataset
I have used [Fer2013 dataset](https://www.kaggle.com/c/3364/download-all) https://www.kaggle.com/c/3364/download-all
<br>It has over 30000 images

### Requirements
1. Python
2. Tensorflow and Keras
3. OpenCV (cv2)
4. Numpy and Pandas

### Usage
You can check the Jupyter-notebook for the training and testing of the model.<br>
To use the code in command use the following commands.
<br> For *Images*:

> python detectImage.py (input_file_path) (optional output_file_path)

Example: python detectImage.py content/TestImage.jpg out.jpg

<br> For *Videos*:
> python detectVideo.py (input_file_path) (optional output_file_path)

Example: python detectVideo.py content/TestVideo.mp4 outVideo

<br>
If you like this work please help me by following and giving me some stars.


