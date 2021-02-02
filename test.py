from keras.preprocessing.image import img_to_array,load_img
from keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt


#LOAD MODEL
model = load_model("MODEL.h5")

#MODEL lOAD WEIGHTS 
model.load_weights('model_weights.h5') 

 
#MODEL TEST
path = "test/"
test_img_path = path + "ugur.jpg"
#woman/face_188.jpg

img_orj = load_img(test_img_path)
img = load_img(test_img_path, grayscale=False, target_size=(48, 48))

x = img_to_array(img)
x = np.expand_dims(x, axis = 0)
x = x.reshape(-1,48,48,3)/255.0



custom = model.predict(x)
#Duygu Analizi(custom[0])


#1
objects = ("man", "woman")
y_pos = np.arange(len(objects))
    
plt.bar(y_pos, custom[0], align='center', alpha=0.5, color='blue')
plt.xticks(y_pos, objects)
plt.ylabel('yüzde')
plt.title('cinsiyet')
plt.show()

#2
x = np.array(x, 'float32')
x = x.reshape([48, 48,3]);
plt.axis('off')
plt.gray()
plt.imshow(img_orj)

plt.show()
#------------------------------

