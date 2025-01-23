



#### **Introduction to Deep Learning**
Deep learning has revolutionized many fields, including computer vision, by enabling systems to understand and process images and videos with remarkable accuracy. This project uses deep learning to build a system capable of detecting human faces in images or videos and predicting their approximate age and gender.



#### **Deep Learning Models Used**
This system leverages the following pre-trained deep learning models for its functionality:

1. **Face Detection Model**: Uses OpenCV's DNN module with the `opencv_face_detector.pb` and `opencv_face_detector.pbtxt` files to locate faces in images and videos.

2. **Age Prediction Model**: A pre-trained Caffe model (`age_net.caffemodel` with `age_deploy.prototxt`) predicts the age group of the detected face from a list of age ranges (e.g., `(0-4)`, `(25-29)`, etc.).

3. **Gender Prediction Model**: Another pre-trained Caffe model (`gender_net.caffemodel` with `gender_deploy.prototxt`) predicts the gender (Male/Female) of the detected face.



#### **System Methodology**
The methodology behind this system involves the following steps:

1. **Input Processing**: Accepts input as an image file or live webcam feed.

2. **Face Detection**: Detects faces in the input using a deep learning-based face detection model.

3. **Preprocessing**: Aligns and resizes the detected face region to meet model requirements.

4. **Feature Prediction**: Processes the face through the gender and age prediction models to estimate both attributes.

5. **Output Rendering**: Displays the original image or video with bounding boxes around detected faces and labels indicating the predicted age and gender.



#### **System Execution**
The system supports two modes of operation:

1. **Photo Upload Mode**: Users can upload an image, and the system will process the photo to detect faces and predict age and gender.

2. **Webcam Mode**: The system uses a live webcam feed to detect faces in real-time, providing predictions for each frame.

**Steps to Run the System:**

1. Clone the repository and navigate to the project directory.
2. Ensure all dependencies are installed:
   ```bash
   pip install opencv-python opencv-python-headless
   pip install numpy
   pip install argparse
   ```
3. Run the Python script:
   ```bash
   python age.py
   ```
4. Choose an operation mode (photo upload or webcam).


