import cv2
import numpy as np
import time

# Start time
start_time = time.time()

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load image
img = cv2.imread('soccer_game.jpg')
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Showing information on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            print('Object detected \nConfidence: ' , confidence * 100 , '%')
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.3)
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        try:
            label = str(classes[class_ids[i]])
            if label == 'player': 
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        except:
            print("An exception occurred")
        
            
# Total execution time
end_time = time.time()
print("Total execution time: {:.2f} seconds".format(end_time - start_time))

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()




