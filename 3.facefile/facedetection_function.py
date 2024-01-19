import cv2
import numpy as np
import pandas as pd
import os

def face_detect(filevideo, facevideo, face_det, face_mesh):
    """detection face in video,crop it and find patch yall roll
    filevideo : path of video file
    facevideo : path of output after crop only face
    face_det : model of facedetection
    face_mesh : model of facealiment"""
    yaws, patchs, rolls = [], [], []
    scores = []
    boxs = {}
    num_people = []    
    
    cap = cv2.VideoCapture(filevideo)
    output = cv2.VideoWriter( 
                f"{facevideo}/{os.path.basename(filevideo)}", cv2.VideoWriter_fourcc(*'mp4v'), 25, (1280, 720))

    while cap.isOpened():
        success, image = cap.read()
        if success == False:
            cap.release()
            output.release() 
            return num_people, yaws, patchs, rolls

        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # To improve performance
        image.flags.writeable = False
        # Get the result
        results = face_det.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        out = np.zeros_like(image)
        # detection face
        if results.detections:
            num_people.append(len(results.detections))
            score = 0
            num = -1
            for ret in results.detections:
                face = ret.location_data.relative_bounding_box
                x_min, y_min = int(face.xmin * image.shape[1]), int(face.ymin * image.shape[0])
                x_max, y_max = x_min + int(face.width * image.shape[1]), y_min + int(face.height * image.shape[0])
                scores.append(ret.score[0])
                boxs[ret.score[0]] = [y_min, y_max, x_min, x_max]
                if ret.score[0] > score: # finding mach score and use it to face alilmet
                    score = ret.score[0]
                    num += 1

            face = results.detections[num].location_data.relative_bounding_box
            x_min, y_min = int(face.xmin * image.shape[1]), int(face.ymin * image.shape[0])
            x_max, y_max = x_min + int(face.width * image.shape[1]), y_min + int(face.height * image.shape[0])

            out[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]
            
            yaw, patch, roll = get_head_pose(out[y_min:y_max, x_min:x_max], face_mesh)
        else:
            num_people.append(0)
            yaw, patch, roll = None, None, None
        print(len(num_people), yaw, patch, roll)
        yaws.append(yaw)
        patchs.append(patch)
        rolls.append(roll)
        output.write(out)
    

def get_head_pose(image, face_mesh):
    """face aliment finding aliment of head pose astimate
    image : image of face
    face mesh : model of facealiment"""
        # face-yaw
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        results = face_mesh.process(image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360


                patch = np.round(x,2)
                yaw = np.round(y,2)
                roll = np.round(z,2) 
        else:
            yaw = None 
            patch = None 
            roll = None 

        return yaw, patch, roll 

    