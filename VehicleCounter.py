import cv2
import numpy as np
import time

from yolov3.yolov3 import Create_Yolov3
from yolov3.utils import load_yolo_weights, image_preprocess, postprocess_boxes, nms, nms_noMask, draw_bbox, read_class_names
import time
from yolov3.configs import *

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.preprocessing import non_max_suppression
from deep_sort import generate_detections as gdet

import tensorflow as tf

class VehicleCounter:
    def __init__(self,
                 yoloModelName = "yolov3",
                 yoloSize = 832,
                 trackerModel_filename = "mars-small128",
                 tracker_max_cosine_distance = 0.2,
                 tracker_maxAge = 7,
                 exit_lines = np.array([[0,1,-1000], [1, 0, -1850]])):
        print("Initializing detector and tracker")

        #Initialize Object Detection Model
        yoloWeights_filename = "model_data/" + yoloModelName + ".weights"
        self.input_size = yoloSize
        self.detectionModel = Create_Yolov3(input_size=yoloSize)
        load_yolo_weights(self.detectionModel, yoloWeights_filename)

        #Initialize Tracker Model
        trackerModel_filename = "model_data/" + trackerModel_filename + '.pb'
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", tracker_max_cosine_distance, None)
        self.encoder = gdet.create_box_encoder(trackerModel_filename, batch_size=1)
        self.tracker = Tracker(metric , max_age = tracker_maxAge)

        #Counting Variables
        self.initialBoxes = {} #Boxes coordinates of first frame
        self.frame_counter = 0 
        self.allowPassFrames = 20 #If frame_counter < allowpassFrames, detection below exit lines will be counted
        self.exit_lines = exit_lines #Coefficient of exit line equations [a b c], for ax + by + c = 0
        self.exit_records = [] #Tracking ids of objects that exited
        self.unsure_records = [] #Tracking ids of objects detected beyond exit lines

        print("Done")
        print()

    def model_prediction(self, model, img, confThreshold=0.7, nmsThreshold=0.2):
        image_data = image_preprocess(np.copy(img), [self.input_size, self.input_size])
        image_data = tf.expand_dims(image_data, 0)
        pred_bbox = model.predict(image_data)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        bboxes = postprocess_boxes(pred_bbox, img, self.input_size, confThreshold)
        bboxes = nms(bboxes, nmsThreshold, method='nms')

        boxes = []
        scores = []
        classIds = []
        for bbox in bboxes:            
            boxes.append([bbox[0].astype(int), bbox[1].astype(int), bbox[2].astype(int)-bbox[0].astype(int), bbox[3].astype(int)-bbox[1].astype(int)])
            scores.append(int(bbox[4]))
            classIds.append(int(bbox[5]))

        return np.array(classIds), np.array(scores), np.array(boxes)

    def vehicleDetection(self, img, confThreshold=0.2, nmsThreshold=0.2, filterClass = [2, 3, 5, 7]):
        #input: img
        #output: detections of filtered classes
        
        classIds, scores, boxes = self.model_prediction(self.detectionModel, img, confThreshold=0.7, nmsThreshold=nmsThreshold)

        #Filter only classes that are vehicles
        filterKey = np.isin(classIds, np.array(filterClass))
        if len(filterKey) == 0:
            return [], [], []
        classIds = classIds[filterKey]
        scores = scores[filterKey]
        boxes = boxes[filterKey]

        return classIds, scores, boxes

    def determineExit(self, bbox):
        #input: Bounding box (x,y,w,h)
        #ouput: True/False depending if object is above or below line (left or right for line along y-axis)
        
        #Get bottom right
        br_x = bbox[0] + bbox[2]
        br_y = bbox[1] + bbox[3]

        #form column vector
        coordinate_vector = np.array([[br_x],[br_y],[1]])
        
        #check if coordinate passed either lines
        return np.amax(np.dot(self.exit_lines, coordinate_vector)) > 0

    def trackVehicle(self, img, classIds, scores, boxes):
        #input: img and detetions
        #outputs:
            # Detections (classes, boxes, ids)
            # Exit records (tracking ids that has exited)
            # Unsure records (tracking ids of detections that were detection beyond exit lines and did not pass through from enter region to exit egion)
        features = np.array(self.encoder(img, boxes))
        
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, classIds, features)]
        
        self.tracker.predict()
        self.tracker.update(detections)
        
        tracked_boxes = []
        tracked_ids = []
        tracked_classes = []
        exit_records = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue 
            bbox = track.to_tlwh().astype(int) # Get the corrected/predicted bounding box
            class_name = track.get_class() #Get the class name of particular object
            tracking_id = track.track_id # Get the ID for the particular track

            tracked_boxes.append(bbox)
            tracked_classes.append(class_name)
            tracked_ids.append(tracking_id)
        

        for (bbox, tracking_id)  in zip(tracked_boxes, tracked_ids):
            if tracking_id not in self.initialBoxes:
                self.initialBoxes[tracking_id] = bbox
                
            if self.determineExit(bbox) and (tracking_id not in self.exit_records):
                if self.determineExit(self.initialBoxes[tracking_id])==False:
                    self.exit_records.append(tracking_id)
                elif self.determineExit(self.initialBoxes[tracking_id]) and (self.frame_counter < self.allowPassFrames):
                    self.exit_records.append(tracking_id)
                elif self.determineExit(self.initialBoxes[tracking_id]) and (self.frame_counter >= self.allowPassFrames):
                    self.unsure_records.append(tracking_id)

        if self.frame_counter < (self.allowPassFrames + 5):
            self.frame_counter+=1

        return tracked_classes, tracked_boxes, tracked_ids, self.exit_records, self.unsure_records

    
    def trackSouthwardVehicles(self, img, confThreshold=0.7, nmsThreshold=0.05):
        classIds, scores, boxes = self.vehicleDetection(img, confThreshold=0.7, nmsThreshold=0.05)
        tracked_classes, tracked_boxes, tracked_ids, exit_records, unsure_records = self.trackVehicle(img, classIds, scores, boxes)

        return tracked_classes, tracked_boxes, tracked_ids, exit_records, unsure_records


class VideoPreprocessor:
    def __init__(self,
                 height,
                 width,
                 percentage_crop,
                 roiPolyPoints = np.array([[0,827],
                                           [1290, 110],
                                           [1919, 533],
                                           [1919, 1079],
                                           [0, 1079]])):
        self.count = 0
        self.mask = self.createRoiMask((int(height), int(width)), roiPolyPoints) #Create mask to filter southbound lanes
        self.percentage_crop = percentage_crop # Crop percentage to conduct feature point mathcing
        

    def createRoiMask(self, imgShape, polypoints):
        #Input: Polygon points
        #Output: Southbound Lane mask
        mask = np.zeros(imgShape).astype(np.uint8)
        cv2.fillPoly(mask, pts = [polypoints], color = (255, 255, 255))
        return mask
    
    def registerOriginal(self, originalFrame):
        # To register point features for first frame
        
        self.originalFrame = originalFrame
        # Crop Only Top part of Frame
        originalFrame = self.cropStabilizeRoi(originalFrame, originalFrame.shape[0])

        #Identify point features for tracking
        source_pts = cv2.goodFeaturesToTrack(originalFrame,
                                             maxCorners=200,
                                             qualityLevel=0.01,
                                             minDistance=30,
                                             blockSize=3)
        self.source_pts = source_pts

    def stabilize_frame(self, frame, width, height):
        # If first frame, register original frame and point features as reference points
        if self.count == 0:
            self.registerOriginal(frame)
            self.count+=1
            return np.identity(3), np.identity(3), frame, frame

        # Else, Initialize original frames
        originalFrame = self.originalFrame
        originalFrame = self.cropStabilizeRoi(originalFrame, height)
        source_pts = self.source_pts

        # Crop top portion of current frame and perform tracking with optical flow
        currentFrame = self.cropStabilizeRoi(frame, height)
        target_pts, status, err = cv2.calcOpticalFlowPyrLK(originalFrame, currentFrame, source_pts, None)

        # Get rigid transformation of reference image to current image: C = HR
        transform, _ = cv2.estimateAffinePartial2D(source_pts, target_pts)
        transformation_mat = np.identity(3)
        transformation_mat[0:2, 0:3] = transform

        # Get inverse transformation, that is transformation of current image to reference point. To obtain alignment to reference, H'HR = H'C = C'
        inv_transformation = np.linalg.inv(transformation_mat)
        self.count+=1
        new_frame = cv2.warpAffine(frame, inv_transformation[0:2, 0:3], (width,height))

        return inv_transformation, transformation_mat, new_frame, frame
        
    def cropStabilizeRoi(self, img, height):
        percentage_crop= self.percentage_crop
        crop_height = height * percentage_crop // 100
        img = img[0:crop_height,:]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def preprocess_img(self, frame, width, height):
        inv_transformation, transformation_mat, new_frame, frame = self.stabilize_frame(frame, width, height)
        masked_frame = cv2.bitwise_and(new_frame, new_frame, mask = self.mask)
        return inv_transformation, transformation_mat, masked_frame, frame

    def transform2OriginalCoord(box, transformation_mat):
        boxes_arr = np.array([[box[0]], [box[1]], [1]])
        boxes_transformed = np.matmul(transformation_mat[0:2,0:3], boxes_arr).astype(np.int32)
        return [boxes_transformed[0][0], boxes_transformed[1][0]]





if __name__ == '__main__':
    vidcap = cv2.VideoCapture("VehicleTest.mp4")
    count_frame = 0
    
    width  = int(vidcap.get(3))  # float `width`
    height = int(vidcap.get(4))
    vid_stabilizer = VideoPreprocessor(height, width, 20)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter("test.mp4", fourcc, 30, (width,height))
    
    while True:
        t1 = time.time()
        ret, frame = vidcap.read()
        
        #Break loop when video ends
        if ret != True:
            print("BREAK")
            cv2.destroyAllWindows()
            break


        ## Stabilize frame
        inv_transformation, transformation_mat, new_frame, frame = vid_stabilizer.preprocess_img(frame, width, height)
        #cv2.imwrite("stable.jpg", new_frame)
        #cv2.imwrite("unstable.jpg", frame)

        writer.write(new_frame)

        frame_show = cv2.resize(new_frame, (960,540))
        cv2.imshow("output", frame_show)
        key = cv2.waitKey(1)
        
        if ret != True or key==27:
            print("BREAK")
            cv2.destroyAllWindows()
            break

        #previous_frame = frame
        print(1/(time.time()-t1))
        count_frame+=1
    writer.release()
