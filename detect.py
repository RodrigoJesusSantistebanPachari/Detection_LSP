import numpy as np
import mediapipe as mp
import os
import csv
import cv2
from utils import refine,elm_spaces

mp_hands = mp.solutions.hands

def is_right_hand(kp):
    
    digitgroups = [
        (17,18,19,20),
        (13,14,15,16),
        (9,10,11,12),
        (5,6,7,8),
        (2,3,4) 
    ]
    
    palm_dir_vec = np.array([0,0,0], dtype=np.float64)
    for digit in digitgroups:
        for idx in digit[1:]:
            palm_dir_vec += kp[idx] - kp[digit[0]]
            
    palm_pos_vec = np.array([0,0,0], dtype=np.float64)
    for digit in digitgroups:
        palm_pos_vec += kp[digit[0]]
    palm_pos_vec /= len(digitgroups)
    
    top_palm_pos_vec = kp[9]
    
    val = np.dot(np.cross(kp[2] - palm_pos_vec, palm_dir_vec), top_palm_pos_vec - palm_pos_vec)

    if val < 0: return True
    
    return False

def ret_vector(imggg):
    mp_drawing = mp.solutions.drawing_utils 
    images = {"imagen":imggg}
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)

    for name, image in images.items():

        results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
        
        if not results.multi_hand_landmarks:
            return 0
        else:
            annotated_image = cv2.flip(image.copy(), 1)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                break
            cv2.imshow("x",cv2.flip(annotated_image, 1))
            arr=[]
            for hand_landmarks in results.multi_hand_landmarks:
                act=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)
                act=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)
                act=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)
                act=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)

                act=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)
                act=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)
                act=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)
                act=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)

                act=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)
                act=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)
                act=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)
                act=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)

                act=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)
                act=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)
                act=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)
                act=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)

                act=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)
                act=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)
                act=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)
                act=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                arr.append(act.x);arr.append(act.y);arr.append(act.z)
                    
            return arr

def get_dist(arr1,arr2):
    dist=0
    for i in range(len(arr1)):
        dist+=abs(arr1[i]-arr2[i])
    return dist

def detector():
    dataset=[]
    print("[INFO] cargando base de datos...")

    letters="abcdefghijklmnopqrstuvwxyz"
    for let in letters:
        for filename in os.listdir('./out_tr/'+let+'/'):
            tmp=[]
            with open(os.path.join('./out_tr/'+let+'/',filename)) as csvfile:
                redd = csv.reader(csvfile, delimiter=',')
                for row in redd:
                    for elem in row:
                        tmp.append(float(elem))

            dataset.append([tmp,let])

    print("[ATENCION] SE CARGO LA BASE DE DATOS")


    print("[ATENCION] INICIO DE DETECCION")
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)

    is_recording = False
    video_count = 1
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out = None

    frase = ""

    while True:
        ret, frame = cap.read()

        letter = ""
        caract=ret_vector(frame)
        if (caract!=0):
            caract=[0]+caract
            minimo=get_dist(dataset[0][0],caract)
            letter=dataset[0][1]
            for aa,bb in dataset:
                rrr=get_dist(aa,caract)
                if(rrr<minimo):
                    minimo=rrr
                    letter=bb

        cv2.putText(frame, letter, (40,50), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 255, 0))

        if is_recording:
            cv2.putText(frame, "Recording", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Webcam', frame)

        key = cv2.waitKey(1)

        if key == ord('r'):
            if not is_recording:
                is_recording = True
                video_out = cv2.VideoWriter(f'Prueba{video_count}.mp4', fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                print(f"Iniciando grabación: Prueba{video_count}.mp4")
            else:
                is_recording = False
                video_out.release()
                print(f"Grabación finalizada: Prueba{video_count}.mp4")
                
                with open(f'Prueba{video_count}.txt', 'w') as file:
                    file.write(frase)
                print(f"Archivo de texto generado: Prueba{video_count}.txt")
                frase = ""

                video_count += 1

        if is_recording:
            frase+=letter
            video_out.write(frame)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return frase



frase=detector()
print(frase)