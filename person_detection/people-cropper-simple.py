import cv2
import os
import numpy as np

net_file= 'deploy.prototxt'  
caffe_model='mobilenet_iter_73000.caffemodel'  
test_dir = "images"
output_dir = "cropped_people"

if not os.path.exists(caffe_model):
    print(caffe_model + " does not exist")
    exit()
if not os.path.exists(net_file):
    print(net_file + " does not exist")
    exit()

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Use OpenCV DNN instead of Caffe
net = cv2.dnn.readNetFromCaffe(net_file, caffe_model)

CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(imgfile):
    origimg = cv2.imread(imgfile)
    
    # Let OpenCV handle preprocessing
    blob = cv2.dnn.blobFromImage(origimg, 0.007843, (300, 300), 127.5)
    
    net.setInput(blob)
    out = net.forward()
    
    # Reshape output to match expected format
    out = {'detection_out': out.reshape(1, 1, -1, 7)}
    
    box, conf, cls = postprocess(origimg, out)

    # Get all detected people
    people_crops = []
    for i in range(len(box)):
        classification = CLASSES[int(cls[i])]
        if classification == 'person' and conf[i] > 0.1:
            x1, y1, x2, y2 = box[i]
            cropped_img = origimg[y1:y2, x1:x2]
            people_crops.append((cropped_img, conf[i]))
    
    return people_crops
    # for i in range(len(box)):
    #    classification = CLASSES[int(cls[i])]
    #    if classification == 'person':
    #     p1 = (box[i][0], box[i][1])
    #     p2 = (box[i][2], box[i][3])
    #     cv2.rectangle(origimg, p1, p2, (0,255,0))
    #     p3 = (max(p1[0], 15), max(p1[1], 15))
    #     title = "%s:%.2f" % (classification, conf[i])
    #     cv2.putText(origimg, title, p3, cv2.FONT_HERSHEY_PLAIN, 5, (34, 0, 145), 5)
    # cv2.imshow("SSD", origimg)
 
    # k = cv2.waitKey(0) & 0xff
    #     #Exit if ESC pressed
    # if k == 27 : return False
    # return True

for f in os.listdir(test_dir):
    # if detect(test_dir + "/" + f) == False:
    #    break
    people_crops = detect(test_dir + "/" + f)
    base_name = os.path.splitext(f)[0]
    
    # Create output subdirectory for this image
    image_output_dir = os.path.join(output_dir, base_name)
    os.makedirs(image_output_dir, exist_ok=True)
    
    if people_crops:
        for i, (cropped, confidence) in enumerate(people_crops):
            output_filename = f"{image_output_dir}/person_{i}_conf_{confidence:.2f}.jpg"
            cv2.imwrite(output_filename, cropped)
            print(f"Saved: {output_filename}")
    else:
        print(f"No people detected in {f}")

# # Loop over detections
# for i in range(detections.shape[2]):
#     confidence = detections[0, 0, i, 2]

#     if confidence > 0.5:
#         box = detections[0, 0, i, 3:7] * [w, h, w, h]
#         (startX, startY, endX, endY) = box.astype("int")

#         # Ensure bounding box is within image dimensions
#         startX = max(0, startX)
#         startY = max(0, startY)
#         endX = min(w - 1, endX)
#         endY = min(h - 1, endY)

#         person_crop = image[startY:endY, startX:endX]
#         crop_path = os.path.join(output_dir, f"person_{person_count}.jpg")
#         cv2.imwrite(crop_path, person_crop)
#         person_count += 1

# print(f"Saved {person_count} cropped person images to '{output_dir}/'")