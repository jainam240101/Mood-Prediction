import pandas as pd
import os
import numpy as np
import cv2

os.mkdir("Testing")
os.mkdir("Testing\\Angry")
os.mkdir("Testing\\Disgust")
os.mkdir("Testing\\Fear")
os.mkdir("Testing\\Happy")
os.mkdir("Testing\\Sad")
os.mkdir("Testing\\Suprise")
os.mkdir("Testing\\Neutral")

emo_dict={0:"Angry",1:"Disgust",2:"Fear",3:"Happy",4:"Sad",5:"Suprise",6:"Neutral"}
print(emo_dict[3])

fer_data=pd.read_csv('icml_face_data.csv',delimiter=',')
new_data=fer_data.iloc[28709:,]
new_data.to_csv("Mood_Testing.csv")

fer_data=pd.read_csv('Mood_Testing.csv')
emotion=fer_data.iloc[:,[0]].values
emo=emotion.tolist()

pixels=np.array(fer_data.iloc[:,[1]].values)
df=pd.DataFrame(pixels)
df.columns=['pixels']

def save_fer_img():
    for index,row in df.iterrows():
        pixels=np.asarray(list(row['pixels'].split(' ')),dtype=np.uint8)
        value=emo[index]
        #print(emo)
        img=pixels.reshape((48,48))
        pathname=os.path.join("Testing\\"+emo_dict[value[0]],emo_dict[value[0]]+'.jpg')
        if os.path.exists(pathname):
            no=1
            while(True):
                new_pathname=os.path.join("Testing\\"+emo_dict[value[0]],emo_dict[value[0]]+'-'+str(no)+'.jpg')
                if(os.path.exists(new_pathname)):
                    no=no+1
                else:
                    pathname=new_pathname
                    break
                
        cv2.imwrite(pathname,img)
        print(index,"\t",pathname)
save_fer_img()



#For Training Dataset
os.mkdir("Training")
os.mkdir("Training\\Angry")
os.mkdir("Training\\Disgust")
os.mkdir("Training\\Fear")
os.mkdir("Training\\Happy")
os.mkdir("Training\\Sad")
os.mkdir("Training\\Suprise")
os.mkdir("Training\\Neutral")
fer_data=pd.read_csv('train.csv')
emotion=fer_data.iloc[:,[0]].values
emo=emotion.tolist()
value=emo[3]

def save_fer_img():
    for index,row in fer_data.iterrows():
        pixels=np.asarray(list(row['pixels'].split(' ')),dtype=np.uint8)
        value=emo[index]
        #print(emo)
        img=pixels.reshape((48,48))
        pathname=os.path.join("Training\\"+emo_dict[value[0]],emo_dict[value[0]]+'.jpg')
        if os.path.exists(pathname):
            no=1
            while(True):
                new_pathname=os.path.join("Training\\"+emo_dict[value[0]],emo_dict[value[0]]+'-'+str(no)+'.jpg')
                if(os.path.exists(new_pathname)):
                    no=no+1
                else:
                    pathname=new_pathname
                    break
                
        cv2.imwrite(pathname,img)
        print(index,pathname)
        
save_fer_img()