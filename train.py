from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,AveragePooling2D,Activation,Dropout,Flatten,Dense,BatchNormalization
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img


from glob import glob
import matplotlib.pyplot as plt


train_path = "train/"
test_path = "valid/"


img = load_img(test_path + "man/face_190.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()


x = img_to_array(img)
print(x.shape)

className = glob(train_path + "/*")
numberOfClass = len(className)
print(numberOfClass)

#CNN MODEL

model = Sequential()
model.add(Conv2D(64,(3,3),input_shape = x.shape,padding='same'))
model.add(Activation("relu"))
model.add(Conv2D(64,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())



model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(Conv2D(128,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())




model.add(Conv2D(256,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(Conv2D(256,(3,3),padding='same'))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D())


model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.2))

model.add(Dense(numberOfClass)) # output
model.add(Activation("softmax"))

model.compile(loss = "categorical_crossentropy",
              optimizer = "adam",
              metrics = ["accuracy"])

model.summary()

batch_size = 64
epoch = 30
#DATA GENERATOR

train_datagen = ImageDataGenerator(rescale= 1./255,
                   shear_range = 0.3,
                   horizontal_flip=True,
                   zoom_range = 0.3)

test_datagen = ImageDataGenerator(rescale= 1./255)

train_generator = train_datagen.flow_from_directory(
        train_path, 
        target_size=x.shape[:2],
        batch_size = batch_size,
        color_mode= "rgb",
        class_mode= "categorical")

test_generator = test_datagen.flow_from_directory(
        test_path, 
        target_size=x.shape[:2],
        batch_size = batch_size,
        color_mode= "rgb",
        class_mode= "categorical")



#MODEL TRAİN

hist = model.fit_generator(
        generator = train_generator,
        steps_per_epoch =  batch_size,
        epochs=epoch,
        validation_data = test_generator,
        validation_steps = batch_size)

#SAVE MODEL
model.save('MODEL.h5')

#MODEL SAVE
model.save_weights("model_weights.h5")

#SAVE HISTORY
import json
with open("history.json","w", encoding = "utf-8") as f:
    json.dump(hist.history,f)

#MODEL EVULATION
print(hist.history.keys())
plt.plot(hist.history["loss"], label = "TRAIN LOSS")
plt.plot(hist.history["val_loss"], label = "VALIDATION LOSS")
plt.legend()
plt.show()

plt.figure()
print(hist.history.keys())
plt.plot(hist.history["accuracy"], label = "TRAIN ACC")
plt.plot(hist.history["val_accuracy"], label = "VALIDATION ACC")
plt.legend()
plt.show()

