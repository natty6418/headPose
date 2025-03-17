import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


cap = cv2.VideoCapture(2)

while cap.isOpened():
    success, image = cap.read()
    
    start = time.time()
    
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
    results = face_mesh.process(image)
    
    image.flags.writeable = True
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for i, landmark in enumerate(face_landmarks.landmark):
                if i == 33 or i==263 or i==1 or i==61 or i==291 or i==199:
                    if i==1:
                        nose_2d = (landmark.x * img_w, landmark.y * img_h)
                        nose_3d = (landmark.x * img_w, landmark.y * img_h, landmark.z*3000)
                    
                    x,y=int(landmark.x * img_w), int(landmark.y * img_h)
                    
                    
                    face_2d.append([x,y])
                    face_3d.append([x,y,landmark.z])
            
            face_2d = np.array(face_2d, dtype=np.float64)
            
            face_3d = np.array(face_3d, dtype=np.float64)
            
            focal_length = 1*img_w
            
            cam_matrix = np.array(
                [[focal_length, 0, img_h/2],
                [0, focal_length, img_w/2],
                [0, 0, 1]]
            )
            
            dist_matrix = np.zeros((4,1), dtype=np.float64)
            
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            
            rmat, jac = cv2.Rodrigues(rot_vec)
            
            # Get angles
            
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            
            x = angles[0]*360
            y = angles[1]*360
            z = angles[2]*360
            
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0]+y*10), int(nose_2d[1]-x*10))
            
            cv2.line(image, p1, p2, (0,255,0), 2)
            
            cv2.putText(image, f"X: {int(x)}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(image, f"Y: {int(y)}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(image, f"Z: {int(z)}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
        end = time.time()
        totalTime = end - start
        fps = 1/totalTime
        # print(f"FPS: {fps}")
        
        cv2.putText(image, f"FPS: {int(fps)}", (20,250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections = mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec               
        )
        
        cv2.imshow("Head Pose Estimation", image)
        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
            
            