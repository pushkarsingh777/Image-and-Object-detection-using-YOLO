import cv2
import numpy as np


weights_path = "yolov3.weights"
cfg_path = "yolov3.cfg"
names_path = "coco.names"


with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]


colors = np.random.uniform(0, 255, size=(len(classes), 3))


net = cv2.dnn.readNet(weights_path, cfg_path)


image_path = "image.jpg"  
image = cv2.imread(image_path)




new_width = 800 
aspect_ratio = new_width / image.shape[1]
new_height = int(image.shape[0] * aspect_ratio)
image = cv2.resize(image, (new_width, new_height))

height, width, _ = image.shape


blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
net.setInput(blob)


layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


detections = net.forward(output_layers)

boxes = []
confidences = []
class_ids = []

for output in detections:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5: 
            center_x, center_y, w, h = (
                int(detection[0] * width),
                int(detection[1] * height),
                int(detection[2] * width),
                int(detection[3] * height),
            )

            x = center_x - w // 2
            y = center_y - h // 2

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)


indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

if len(indices) > 0:
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        color = colors[class_ids[i]]

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


output_width = 800  
output_height = 600
resized_output = cv2.resize(image, (output_width, output_height))


cv2.imshow("YOLO Object Detection", resized_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
