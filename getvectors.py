import mediapipe as mp
import cv2
import os
from utils import limpiar_carpeta,ret_vector

mp_hands = mp.solutions.hands

def vect_to_arch(vect,dirr,name):
    if vect==0:
        return
    newname=dirr+name+".csv"
    arch=open(newname,"w")
    print(0,file=arch,end='')
    arch.close()
    arch=open(newname,"a")
    
    for p in vect:
        print(",",file=arch,end="")
        print(p,end="",file=arch)
    arch.close()
        
def run_rot(folder,out):
    limpiar_carpeta(out)
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        vect_to_arch(ret_vector(img,mp_hands),out,filename)
        

if __name__ == "__main__":
    letters="abcdefghijklmnopqrstuvwxyz"
    for let in letters:
        run_rot('./train/'+let+'/','./out_tr/'+let+'/')
    
