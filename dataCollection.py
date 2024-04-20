import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2
from time import time

classID = 1  # 0 is fake and 1 is real
output_folder_path = "Dataset/Real"
confidence = 0.8
save = True
blurThreshold = 35  # Large value means less blur

debug = False
offset_percentage_W = 10
offset_percentage_H = 20
cam_width, cam_height = 640, 480
floatingPoint = 6  # Floating point precision

cap = cv2.VideoCapture(0)
cap.set(3, cam_width)
cap.set(4, cam_height)

detector = FaceDetector()

# Run the loop to continually get frames from the webcam
while True:
    # Read the current frame from the webcam
    # success: Boolean, whether the frame was successfully grabbed
    # img: the captured frame
    success, img = cap.read()

    img_out = img.copy()
    # Detect faces in the image
    # img: Updated image
    # bboxs: List of bounding boxes around detected faces
    img, bboxs = detector.findFaces(img, draw=False)

    listBlur = []  # True False values indicating if the face is blurry or not
    listInfo = []  # The normalized values and class name for the label txt file

    # Check if any face is detected
    if bboxs:
        # Loop through each bounding box
        for bbox in bboxs:
            # bbox contains 'id', 'bbox', 'score', 'center'
            # ---- Get Data  ---- #
            center = bbox["center"]
            x, y, w, h = bbox["bbox"]
            score = int(bbox["score"][0] * 100)

            if score > confidence:

                offsetW = int((offset_percentage_W / 100) * w)
                offsetH = int((offset_percentage_H / 100) * h)

                x, y, w, h = (
                    x - offsetW,
                    y - offsetH * 3,
                    w + 2 * offsetW,
                    h + 3 * offsetH,
                )

                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                if w < 0:
                    w = 0
                if h < 0:
                    h = 0

                # ------- Find Blurriness --------
                imgFace = img[y : y + h, x : x + w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())

                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)

                # ----- Normlize Blurriness -------
                ih, iw, _ = img.shape
                xc, yc = int(x + w / 2), int(y + h / 2)
                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)

                if xcn > 1:
                    xcn = 1
                if ycn > 1:
                    ycn = 1
                if wn > 1:
                    wn = 1
                if hn > 1:
                    hn = 1

                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")

                # ---- Draw Data  ---- #
                # cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
                # cvzone.putTextRect(img, f'{score}%', (x, y - 10))
                cv2.rectangle(img_out, (x, y, w, h), (0, 255, 0), 2)
                cvzone.putTextRect(
                    img_out,
                    f"Score: {int(score)}% Blur:  {blurValue}",
                    (x, y - 20),
                    scale=1,
                    thickness=2,
                )

                if debug:
                    cv2.rectangle(img, (x, y, w, h), (0, 255, 0), 2)
                    cvzone.putTextRect(
                        img,
                        f"Score: {int(score)}% Blur: {blurValue}",
                        (x, y - 20),
                        scale=1,
                        thickness=2,
                    )

        # ------ To save --------
        if save:
            if all(listBlur) and listBlur != []:
                timeNow = time()
                timeNow = str(timeNow).split(".")
                timeNow = timeNow[0] + timeNow[1]
                cv2.imwrite(f"{output_folder_path}/{timeNow}.jpg", img)

                for info in listInfo:
                    f = open(f"{output_folder_path}/{timeNow}.txt", "a")
                    f.write(info)
                    f.close()

    # Display the image in a window named 'Image'
    cv2.imshow("Image", img_out)
    # Wait for 1 millisecond, and keep the window open
    cv2.waitKey(1)
