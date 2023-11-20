import pandas as pd
import numpy as np
import cv2
import json

def prepare_keypoints_image(keypoints,tag):
    # this vaariable is feeded in draw points process and used in the draw joint lines proceess
    part_line = {}

    # DRAW POINTS
    img = np.zeros((256, 256, 3), np.uint8)

    for n, coords in enumerate(keypoints):

        cor_x = int(coords[0] * 256)
        cor_y = int(coords[1] * 256)

        cv2.circle(img, (cor_x, cor_y), 1, (0, 0, 255), -1)
        part_line[n] = (cor_x, cor_y)

    #cv2.imwrite(f'foo_{tag}.jpg', img)

    return img

def prepare_keypoints_image(keypoints, connections=[], pos_rel='', addText=None):
    # this vaariable is feeded in draw points process and used in the draw joint lines proceess
    part_line = {}
    part_type = {}

    # DRAW POINTS
    img= np.zeros((256, 256, 3), np.uint8)
    #imgP = np.zeros((256, 256, 3), np.uint8)
    #imgLH = np.zeros((256, 256, 3), np.uint8)
    #imgRH = np.zeros((256, 256, 3), np.uint8)

    # To print numbers
    fontScale = 0.5
    color = (0, 255, 0)
    thickness = 2

    org = (220, 20)
    img = cv2.putText(img, str(pos_rel), org, cv2.FONT_HERSHEY_SIMPLEX, 
                          fontScale, color, thickness, cv2.LINE_AA)

    # To print the text
    if addText:
        org = (20, 20)
        img = cv2.putText(img, addText, org, cv2.FONT_HERSHEY_SIMPLEX, 
                          fontScale, color, thickness, cv2.LINE_AA)

    #pose, face, leftHand, rightHand = body_parts_class.body_part_points()

    for n, coords in enumerate(keypoints):

        cor_x = int(coords[0] * 256)
        cor_y = int(coords[1] * 256)
        #cv2.circle(img, (cor_x, cor_y), 2, (0, 0, 255), -1)
        part_line[n] = (cor_x, cor_y)
        '''
        if n in pose:
            cv2.circle(imgP, (cor_x, cor_y), 2, (0, 0, 255), -1)
            #part_line[n] = (cor_x, cor_y)
            #part_type[n] = 'pose'
        elif n in leftHand:
            cv2.circle(imgLH, (cor_x, cor_y), 2, (0, 0, 255), -1)
            #part_line[n] = (cor_x, cor_y)
            #part_type[n] = 'left_hand'
        elif n in rightHand:
            cv2.circle(imgRH, (cor_x, cor_y), 2, (0, 0, 255), -1)
            #part_line[n] = (cor_x, cor_y)
            #part_type[n] = 'right_hand'
        #else:
            #part_line[n] = (cor_x, cor_y)
            #part_type[n] = 'blank'
        '''

    # DRAW JOINT LINES
    for start_p, end_p in connections:
        if start_p in part_line and end_p in part_line:
            #s_type, e_type = part_type[start_p], part_type[end_p]
            start_p = part_line[start_p]
            end_p = part_line[end_p]
            cv2.line(img, start_p, end_p, (0,255,0), 2)
            cv2.circle(img, start_p, 2, (0, 0, 255), -1)
            cv2.circle(img, end_p, 2, (0, 0, 255), -1)
            '''
            if s_type == e_type:
                if s_type == 'pose':
                    cv2.line(imgP, start_p, end_p, (0,255,0), 2)
                if s_type == 'left_hand':
                    cv2.line(imgLH, start_p, end_p, (0,255,0), 2)
                if s_type == 'right_hand':
                    cv2.line(imgRH, start_p, end_p, (0,255,0), 2)
            '''


    #final_img = np.concatenate((imgP, imgLH, imgRH), axis=1)
    return img#final_img

def get_edges_index(keypoints_number=71):
    
    points_joints_info = pd.read_csv(f'./points_{keypoints_number}.csv')
    # we subtract one because the list is one start (we wanted it to start in zero)
    ori = points_joints_info.origin-1
    tar = points_joints_info.tarjet-1

    ori = np.array(ori)
    tar = np.array(tar)

    return np.array([ori,tar])

def load_configuration(name):
    # Cargar configuraciones desde JSON
    with open(f"{name}.json", 'r') as archivo_json:
        config = json.load(archivo_json)

    return config