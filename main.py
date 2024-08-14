# Import Libraries and Initialize Video Feed
import cv2
import pickle
import cvzone
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase with Firestore
cred = credentials.Certificate('parkir/key.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Mapping ID to Firestore Document Paths
id_to_slot = {
    1: 'slot_1',
    2: 'slot_2',
    3: 'slot_3',
    4: 'slot_4',
    5: 'slot_5',
    6: 'slot_6'
    # Add more slots as needed
}

cap = cv2.VideoCapture(1)

# Load Parking Slot Positions from File 
with open('parkir/mobil_pos', 'rb') as f:
    posList = pickle.load(f)

# Define and Implement checkParkingSpace Function
width, height = 40, 68

def checkParkingSpace(imgPro):
    spaceCounter = 0
    
    for id, x, y in posList:
        imgCrop = imgPro[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        if count < 300:
            color = (0, 255, 0)
            thickness = 5
            spaceCounter += 1
            status = False
            
        else:
            color = (0, 0, 255)
            thickness = 2
            status = True

        # Map ID to Firestore Document and Update/Delete Status
        slot_name = id_to_slot.get(id)
        if slot_name:
            doc_ref = db.collection('slot_parking').document(slot_name)
            if status:
                # If the slot is occupied, add/update the document
                doc_ref.set({'status': status})
            else:
                # If the slot is free, delete the document
                doc_ref.delete()

        # Display ID and Pixel Count
        cv2.rectangle(img, (x, y), (x + width, y + height), color, thickness)
        cvzone.putTextRect(img, str(id), (x + 5, y + 15), scale=1, thickness=1, offset=0, colorR=color)
        cvzone.putTextRect(img, str(count), (x + 5, y + height - 10), scale=1, thickness=1, offset=0, colorR=color)

    # Display Occupied/Total Parking Slots
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3,
                       thickness=5, offset=20, colorR=(0, 200, 0))


# Set window size when running
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", 800, 600)

# Main loop
while True:
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    success, img = cap.read()
    if not success:
        break
    
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 25, 16)
    imgMedian = cv2.medianBlur(imgThreshold, 5)
    kernel = np.ones((3, 3), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

    checkParkingSpace(imgDilate)
    cv2.imshow("Image", img)
    
    if cv2.waitKey(10) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Close video capture
cap.release()
cv2.destroyAllWindows()
