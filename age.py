import cv2
import argparse
import os
import sys

# Function to detect and highlight faces
def highlightface(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    blob = cv2.dnn.blobFromImage(
        frameOpencvDnn,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104, 117, 123),
        swapRB=True,
        crop=False
    )
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)

            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(
                frameOpencvDnn,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                thickness=int(round(frameHeight / 150)),
                lineType=8
            )

    return frameOpencvDnn, faceBoxes

# Main script
if __name__ == "__main__":
    # Argument parsing
    if "ipykernel" in sys.modules:  # Running in Jupyter Notebook
        args = argparse.Namespace(image=None)  # Set to None as we will prompt the user
    else:  # Running from the command line
        parser = argparse.ArgumentParser()
        parser.add_argument('--image', required=False, default=None, help="Path to the input image")
        args = parser.parse_args()

    # Prompt user for input method
    print("Choose an option:")
    print("1. Upload a photo")
    print("2. Use webcam")
    option = input("Enter option number : ")

    # Model files
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"

    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"

    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    # Mean values for preprocessing
    Model_Mean_Values = (78.4263377603, 87.7689143744, 114.895847746)

    # Age and gender lists
    ageList = [
        '(0-4)', '(5-9)', '(10-14)', '(15-19)', '(20-24)', '(25-29)',
        '(30-34)', '(35-39)', '(40-44)', '(45-49)', '(50-54)', '(55-59)',
        '(60-64)', '(65-69)', '(70-74)', '(75-79)', '(80-84)', '(85-90)',
        '(90-94)', '(95-100)'
    ]
    genderList = ['Male', 'Female']

    # Load the models
    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    padding = 20

    # Option 1: Upload photo
    if option == "1":
        image_path = input("Enter the path to the image file: ")
        if os.path.isfile(image_path):
            frame = cv2.imread(image_path)
            resultImg, faceBoxes = highlightface(faceNet, frame)
            if not faceBoxes:
                print("No face detected")
            else:
                for faceBox in faceBoxes:
                    face = frame[
                        max(0, faceBox[1] - padding): min(faceBox[3] + padding, frame.shape[0] - 1),
                        max(0, faceBox[0] - padding): min(faceBox[2] + padding, frame.shape[1] - 1)
                    ]
                    blob = cv2.dnn.blobFromImage(
                        face,
                        scalefactor=1.0,
                        size=(227, 227),
                        mean=Model_Mean_Values,
                        swapRB=False
                    )

                    # Predict gender
                    genderNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]
                    print(f"Gender Predicted: {gender}")  # Debugging print statement

                    # Predict age
                    ageNet.setInput(blob)
                    agePreds = ageNet.forward()
                    age = ageList[agePreds[0].argmax()]

                    label = f"{gender}, {age}"
                    cv2.putText(resultImg, label, (faceBox[0], faceBox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # Display the result for images
            cv2.imshow("Detecting Age and Gender", resultImg)
            cv2.waitKey(0)  # Wait indefinitely until a key is pressed
            cv2.destroyAllWindows()

        else:
            print("Invalid image path or file not found.")

    # Option 2: Use webcam
    elif option == "2":
        video = cv2.VideoCapture(0)  # Initialize video capture for the webcam

        while True:
            hasFrame, frame = video.read()  # Capture a frame
            if not hasFrame:
                print("Video stream ended or no frame captured")
                break

            resultImg, faceBoxes = highlightface(faceNet, frame)
            if not faceBoxes:
                print("No Face is Detected")
            else:
                for faceBox in faceBoxes:
                    face = frame[
                        max(0, faceBox[1] - padding): min(faceBox[3] + padding, frame.shape[0] - 1),
                        max(0, faceBox[0] - padding): min(faceBox[2] + padding, frame.shape[1] - 1)
                    ]
                    blob = cv2.dnn.blobFromImage(
                        face,
                        scalefactor=1.0,
                        size=(227, 227),
                        mean=Model_Mean_Values,
                        swapRB=False
                    )

                    # Predict gender
                    genderNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    gender = genderList[genderPreds[0].argmax()]
                    print(f"Gender Predicted: {gender}")  # Debugging print statement

                    # Predict age
                    ageNet.setInput(blob)
                    agePreds = ageNet.forward()
                    age = ageList[agePreds[0].argmax()]

                    label = f"{gender}, {age}"
                    cv2.putText(resultImg, label, (faceBox[0], faceBox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

            # Display the result for video frames
            cv2.imshow("Detecting Age and Gender", resultImg)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()  # Release the webcam resource
        cv2.destroyAllWindows()

    else:
        print("Invalid option. Please choose either 1 or 2.")
