import cv2
import os
import numpy as np

default_net_file= 'person_detection/deploy.prototxt'  
default_caffe_model='person_detection/mobilenet_iter_73000.caffemodel'  
default_output_dir = "person_detection/cropped_people"

CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

class PeopleCropper:
    def __init__(self, net_file = default_net_file, caffe_model = default_caffe_model, output_dir = default_output_dir):
        self.net_file = net_file
        self.caffe_model = caffe_model
        self.output_dir = output_dir

        if not os.path.exists(caffe_model):
            print(caffe_model + " does not exist")
            exit()
        if not os.path.exists(net_file):
            print(net_file + " does not exist")
            exit()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Use OpenCV DNN instead of Caffe
        self.net = cv2.dnn.readNetFromCaffe(net_file, caffe_model)

    def _preprocess(self, src):
        img = cv2.resize(src, (300,300))
        img = img - 127.5
        img = img * 0.007843
        return img

    def _postprocess(self, img, out):   
        h = img.shape[0]
        w = img.shape[1]
        box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

        cls = out['detection_out'][0,0,:,1]
        conf = out['detection_out'][0,0,:,2]
        return (box.astype(np.int32), conf, cls)

    def _detect(self, imgfile):
        origimg = cv2.imread(imgfile)
        
        # Let OpenCV handle preprocessing
        blob = cv2.dnn.blobFromImage(origimg, 0.007843, (300, 300), 127.5)
        
        self.net.setInput(blob)
        out = self.net.forward()
        
        # Reshape output to match expected format
        out = {'detection_out': out.reshape(1, 1, -1, 7)}
        
        box, conf, cls = self._postprocess(origimg, out)

        # Get all detected people
        people_crops = []
        for i in range(len(box)):
            classification = CLASSES[int(cls[i])]
            if classification == 'person' and conf[i] > 0.1:
                x1, y1, x2, y2 = box[i]
                cropped_img = origimg[y1:y2, x1:x2]
                people_crops.append((cropped_img, conf[i]))
        
        return people_crops

    def detect_dir(self, directory):
        for f in os.listdir(directory):
            self.detect(directory + "/" + f)
            
    def detect(self, filepath):
        people_crops = self._detect(filepath)
        
        if people_crops:
            # Create output subdirectory for this image
            filename = filepath.split('/')[-1]
            print(filename)
            image_output_dir = os.path.join(self.output_dir, filename)
            os.makedirs(image_output_dir, exist_ok=True)

            output_filenames = []
            for i, (cropped, confidence) in enumerate(people_crops):
                if cropped.shape[0] < 10 or cropped.shape[1] < 10:
                    print(f"Skipping person {i} - too small")
                    continue
                output_filename = f"{image_output_dir}/person_{i}_conf_{confidence:.2f}.jpg"
                cv2.imwrite(output_filename, cropped, [cv2.IMWRITE_JPEG_QUALITY, 20])
                print(f"Saved: {output_filename}")
                output_filenames.append(output_filename)
            return output_filenames 