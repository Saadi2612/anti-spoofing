import cvzone
from cvzone.FaceDetectionModule import FaceDetector
import cv2

cap = cv2.VideoCapture(0)

detector = FaceDetector()

offset_percentage_W = 10
offset_percentage_H = 20
confidence = 0.8

# Run the loop to continually get frames from the webcam
while True:
    # Read the current frame from the webcam
    # success: Boolean, whether the frame was successfully grabbed
    # img: the captured frame
    success, img = cap.read()
    # Detect faces in the image
    # img: Updated image
    # bboxs: List of bounding boxes around detected faces
    img, bboxs = detector.findFaces(img, draw=False)
    # Check if any face is detected
    if bboxs:
        # Loop through each bounding box
        for bbox in bboxs:
            # bbox contains 'id', 'bbox', 'score', 'center'
            # ---- Get Data  ---- #
            center = bbox["center"]
            x, y, w, h = bbox['bbox']
            score = bbox['score'][0]

            if score > confidence:

                offsetW = int((offset_percentage_W / 100) * w)
                offsetH = int((offset_percentage_H / 100) * h)

                x, y, w, h = x - offsetW, y - offsetH*3, w + 2 * offsetW, h + 3 * offsetH

                if x<0:
                    x = 0
                if y<0:
                    y = 0
                if w<0:
                    w = 0
                if h<0:
                    h = 0

                # ------- Find Blurriness --------
                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())


                cv2.rectangle(img, (x, y, w, h), (0, 255, 0), 2)
                score = int(bbox['score'][0] * 100)
                
                # ---- Draw Data  ---- #
                cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
                # cvzone.putTextRect(img, f'{score}%', (x, y - 10))
                cvzone.putTextRect(img, f'Blur:  {blurValue}', (x, y-20))
            
    # Display the image in a window named 'Image'
    cv2.imshow("Image", img)
    # Wait for 1 millisecond, and keep the window open
    cv2.waitKey(1)
    