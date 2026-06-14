# OpenCV library ko import karte hain, jo computer vision ke kaam ke liye use hoti hai
import cv2 
 # Threshold value set karte hain; confidence score ke niche ke objects ko ignore karenge 
thres = 0.5 
# Webcam se video capture karne ke liye object banate hain
cap = cv2.VideoCapture(0)  # "0" ka matlab hai default webcam ka use karna
# Webcam ke properties set karte hain
cap.set(3, 648)  # Frame width ko 648 pixels set karte hain
cap.set(4, 448)  # Frame height ko 448 pixels set karte hain
cap.set(10, 70)  # Brightness ko 70 set karte hain

# Object detection ke liye class names (categories) ka list banate hain
className = []  # Empty list banayi jisme classes store hongi
classFile = 'coco.names'  # File jisme object categories ki list hai
with open(classFile, 'rt') as f:  # File ko read mode me open karte hain
    className = f.read().rstrip('\n').split('\n')  # File ke content ko list me convert karte hain

# Configuration aur weights file ke path define karte hain
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'  # Model ka configuration file
weightsPath = 'frozen_inference_graph - Copy.pb'  # Pre-trained model ka weight file

# DNN (Deep Neural Network) detection model banate hain
net = cv2.dnn_DetectionModel(weightsPath, configPath)  # Model ko weights aur config ke saath load karte hain
net.setInputSize(320, 320)  # Input image size 320x320 pixels set karte hain
net.setInputScale(1.0 / 127.5)  # Scale factor normalize karte hain (1/127.5)
net.setInputMean((127.5, 127.5, 127.5))  # Mean subtraction ke liye RGB mean values set karte hain
net.setInputSwapRB(True)  # RGB se BGR channel order swap karte hain (OpenCV ke liye zaruri)

# Output window ko full-screen banane ke liye properties set karte hain
cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)  # Output window ka naam define karte hain
cv2.setWindowProperty("Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Window ko full-screen banate hain

# Main loop jo frames ko continuously capture aur process karega
while True:
    success, img = cap.read()  # Webcam se ek frame capture karte hain
    classIds, confs, bbox = net.detect(img, confThreshold=thres)  # Object detection perform karte hain
    # `classIds`: Detected object ke IDs
    # `confs`: Confidence scores
    # `bbox`: Bounding box coordinates

    print(classIds, bbox)  # Debugging ke liye detected objects aur unke boxes ko print karte hain

    # Agar koi object detect hota hai to processing karein
    if len(classIds) != 0:
        # Har detected object ke liye loop chalayenge
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)  # Bounding box draw karte hain (green color)
            cv2.putText(img, className[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Object name display karte hain
            cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)  # Confidence score display karte hain

    cv2.imshow("Output", img)  # Processed frame ko "Output" window me display karte hain

    # Agar user 'q' dabaye to loop break karein
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Resource cleanup karte hain
cap.release()  # Webcam ko release karte hain
cv2.destroyAllWindows()  # Sabhi OpenCV windows ko close karte hain





