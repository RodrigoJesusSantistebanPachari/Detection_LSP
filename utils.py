import os, re, os.path
import cv2

def limpiar_carpeta(dir):
    for root, dirs, files in os.walk(dir):
        for file in files:
            os.remove(os.path.join(root, file))

def ret_vector(imagen,motor):
    hands = motor.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7)
    results = hands.process(cv2.flip(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB), 1))
    #results = hands.process(image)

    if not results.multi_hand_landmarks:
        return 0
    else:
        #print(results.multi_hand_landmarks)
        arr=[]
        for hand_landmarks in results.multi_hand_landmarks:
            act=hand_landmarks.landmark[motor.HandLandmark.PINKY_DIP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)
            act=hand_landmarks.landmark[motor.HandLandmark.PINKY_MCP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)
            act=hand_landmarks.landmark[motor.HandLandmark.PINKY_PIP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)
            act=hand_landmarks.landmark[motor.HandLandmark.PINKY_TIP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)

            act=hand_landmarks.landmark[motor.HandLandmark.RING_FINGER_DIP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)
            act=hand_landmarks.landmark[motor.HandLandmark.RING_FINGER_MCP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)
            act=hand_landmarks.landmark[motor.HandLandmark.RING_FINGER_PIP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)
            act=hand_landmarks.landmark[motor.HandLandmark.RING_FINGER_TIP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)

            act=hand_landmarks.landmark[motor.HandLandmark.MIDDLE_FINGER_DIP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)
            act=hand_landmarks.landmark[motor.HandLandmark.MIDDLE_FINGER_MCP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)
            act=hand_landmarks.landmark[motor.HandLandmark.MIDDLE_FINGER_PIP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)
            act=hand_landmarks.landmark[motor.HandLandmark.MIDDLE_FINGER_TIP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)

            act=hand_landmarks.landmark[motor.HandLandmark.INDEX_FINGER_DIP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)
            act=hand_landmarks.landmark[motor.HandLandmark.INDEX_FINGER_MCP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)
            act=hand_landmarks.landmark[motor.HandLandmark.INDEX_FINGER_PIP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)
            act=hand_landmarks.landmark[motor.HandLandmark.INDEX_FINGER_TIP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)

            act=hand_landmarks.landmark[motor.HandLandmark.THUMB_CMC]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)
            act=hand_landmarks.landmark[motor.HandLandmark.THUMB_IP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)
            act=hand_landmarks.landmark[motor.HandLandmark.THUMB_MCP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)
            act=hand_landmarks.landmark[motor.HandLandmark.THUMB_TIP]
            arr.append(act.x);arr.append(act.y);arr.append(act.z)
                
        return arr

def agg_let(pal,let):
    if(pal==""):
        return(let)
    if(pal[-1]!=let):
        return(pal+let)
    else:
        return(pal)

def refine(strr,tol):
    refin=""

    let=""
    cont=0

    lettmp=""
    tmpcc=0

    rev=False
    for ltr in strr:
        if(not rev):
            if(ltr==let):
                cont+=1
                if(cont>=tol):
                    refin=agg_let(refin,let)
            else:
                lettmp=ltr
                tmpcc=1
                rev=True
        else:
            if(ltr==lettmp):
                tmpcc+=1
                if(tmpcc==tol):
                    let=lettmp
                    cont=tmpcc
                    rev=False
            else:
                rev=False

    return(refin)

def elm_spaces(lett):
    res=""
    for lt in lett:
        if(lt=="t"):
            res+="d"
        if(lt!="_"):
            res+=lt

        else:
            res+=" "
    return(res)

def ret_promedios():
    letters="abcdefghijklmnopqrstuvwxyz_"
    for let in letters:
        tmpn=0
        tmpx=[]
        for filename in os.listdir('./out_tr/'+let+'/'):
            tmp=[]
            with open(os.path.join('./out_tr/'+let+'/',filename)) as csvfile:
                redd = csv.reader(csvfile, delimiter=',')
                for row in redd:
                    for elem in row:
                        tmp.append(float(elem))

            dataset.append([tmp,let])


if __name__ == "__main__":
    rr=refine("eeeeeeeeeeeeeeeeeeexssszzzzzsslllllllllllllllassssiiiiiiiiiiiiiiiiizzxxxxxxxxxxxzxxsx___________________jxmsppmnppppnmsxssssallaaaaaaaaaaaaaaasssxxvkkkkvryuiiiiiiiiiiiiiiiiiutttxxxxxxxssaaaaaaaaaaaaaaasssxxxxxx",6)
    print(rr)

