import cv2
import numpy as np
import time
from VehicleCounter import VideoPreprocessor, VehicleCounter
import yaml




def read_configs(filename):
    with open(filename, 'r') as stream:
        configs = yaml.load(stream, Loader = yaml.FullLoader)
    return configs
    
    

if __name__ == '__main__':
    configs = read_configs('configs.yaml')
    vidcap = cv2.VideoCapture(configs['video_input'])

    #Get video frame data
    width  = int(vidcap.get(3))   
    height = int(vidcap.get(4))
    original_fps = vidcap.get(cv2.CAP_PROP_FPS)

    output_filename = "outputs/" + configs['video_input'].split('.')[0] + "_" + str(int(time.time())) + ".mp4" 
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, original_fps, (width,height))

    count_frame = 0
    vid_proprocessor = VideoPreprocessor(height, width, configs["stabilizerPercentageCropFromTop"], np.array(configs["roiPolyPoints"]))
    vehicle_counter = VehicleCounter(configs['yoloModelName'],
                                       configs['yoloSize'],
                                       configs['trackerModel_filename'],
                                       configs['tracker_max_cosine_distance'],
                                       configs['tracker_maxAge'],
                                       np.array(configs['exit_lines']))
    
    with open('model_data/names.txt', 'r') as f:
        classes = f.read().splitlines()

    while True:
        t1 = time.time()

        ret, frame = vidcap.read()
        #Break loop when video ends
        if ret != True:
            print("BREAK")
            cv2.destroyAllWindows()
            break

        ## Stabilize frame
        inv_transformation, transformation_mat, processed_frame, original_frame = vid_proprocessor.preprocess_img(frame, width, height)
        classIds, boxes, trackIds, exit_records, unsure_records = vehicle_counter.trackSouthwardVehicles(processed_frame,
                                                                                                          confThreshold = configs["confThreshold"],
                                                                                                          nmsThreshold = configs["nmsThreshold"])


        for (classId, box, trackId) in zip(classIds, boxes, trackIds):
            #Transform to original
            top_left = (box[0], box[1])
            bottom_right = (box[0] + box[2], box[1] + box[3])
            top_left = VideoStabilizer.transform2OriginalCoord(top_left, transformation_mat)
            bottom_right = VideoStabilizer.transform2OriginalCoord(bottom_right, transformation_mat)

            if trackId in exit_records:
                color = (0, 255 , 0)
            elif trackId in unsure_records:
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)
            
            cv2.rectangle(original_frame, (bottom_right[0], bottom_right[1]),(top_left[0], top_left[1]),
                  color=color, thickness=2)
            text =  str(trackId) + " " + classes[classId]
            cv2.putText(original_frame, text, (top_left[0], top_left[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                color=color, thickness=2)

        vehicles_exited_text = "Vehicles Exited (South): " + str(len(exit_records))
        cv2.rectangle(original_frame, (93, 45),(970, 117),
                  color=(0,0,0), thickness=-1)
        cv2.putText(original_frame, vehicles_exited_text, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
            color=(0, 0, 255), thickness=3)

        writer.write(frame)
        original_frame = cv2.resize(original_frame, (1920//2, 1080//2))
        
        cv2.imshow("output", original_frame)
        key = cv2.waitKey(1)
        if ret != True or key==27:
            print("BREAK")
            cv2.destroyAllWindows()
            break

        #previous_frame = frame
        print(1/(time.time()-t1))
        count_frame+=1
        
    writer.release()
