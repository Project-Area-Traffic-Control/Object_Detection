import cv2
import numpy as np

###=======================================================================================Images

# net =  cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
# classes = []

# with open('coco.names','r') as f:
#     classes = f.read().splitlines()

# img = cv2.imread('images/horses.jpg')

# height, width, _ = img.shape

# blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)
# net.setInput(blob)
# output_layers_names = net.getUnconnectedOutLayersNames()
# layersOutputs = net.forward(output_layers_names)

# boxes = []
# confidences = []
# class_ids = []

# for output in layersOutputs:
#     for detection in output:
#         scores = detection[5:]
#         class_id = np.argmax(scores)
#         confidence = scores[class_id]
#         if confidence > 0.5:
#             center_x = int(detection[0]*width)
#             center_y = int(detection[1]*height)
#             w = int(detection[2]*width)
#             h = int(detection[3]*height)


#             x = int(center_x - w/2)
#             y = int(center_y - h/2)

#             boxes.append([x,y,w,h])
#             confidences.append((float(confidence)))
#             class_ids.append(class_id)


# indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

# font = cv2.FONT_HERSHEY_PLAIN
# colors = np.random.uniform(0,255,size=(len(boxes), 3))

# for i in indexes.flatten():
#     x,y,w,h = boxes[i]
#     label = str(classes[class_ids[i]])
#     confidence = str(round(confidences[i],2))
#     color = colors[i]
#     cv2.rectangle(img, (x,y), (x+w,y+h), color, 2)
#     cv2.putText(img,label+ " "+ confidence, (x,y+20), font, 2,(255,255,255), 2)



# cv2.imshow('Image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

###====================================================================================== VDO

net =  cv2.dnn.readNet('yolov4-tiny.weights','yolov4-tiny.cfg')
classes = []

with open('coco.names','r') as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture('test_1.mp4')  # edit = 0 open camera notebook

def rescale_frame(frame, percent):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while True:
    _, img =cap.read()
    frame75 = rescale_frame(img, percent=50)
    height, width, _ = frame75.shape
    blob = cv2.dnn.blobFromImage(frame75, 1/255, (416,416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layersOutputs = net.forward(output_layers_names)
    # cv2.line(frame75, (20,200), (50,200), (0,0,255) ,2)
    boxes = []
    confidences = []
    class_ids = []
    for output in layersOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)


                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x,y,w,h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
    print(indexes.flatten())
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0,255,size=(len(boxes), 3))

    for i in indexes.flatten():
        x,y,w,h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i],2))
        color = colors[i]
        cv2.rectangle(frame75, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame75,label+ " "+ confidence, (x,y+20), font, 2,(255,255,255), 2)

    cv2.imshow('VDO',frame75)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


