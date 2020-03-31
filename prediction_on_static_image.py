#Prediction on Random Image
import cv2
import tensorflow as tf
import numpy as np

emotion_labels=["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

def prepare(filepath):
    img_size=48
    img_array=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(img_size,img_size))
    return new_array.reshape(-1,img_size,img_size,1)

model=tf.keras.models.load_model('mood_predict.model')
prediction=model.predict([prepare('Your_Image_Here')])
print(prediction)


print(np.amax(prediction))

for i in range(7):
    if(prediction.item(i)==np.amax(prediction)):
        print(emotion_labels[i])


