import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
from keras.models import load_model


#Wwebcam okuma
vid = cv2.VideoCapture(0)

#duygu tahmini
model = load_model("MODEL.h5")

#etiket
label = ["man", "woman"]

# face hearcasced
face_cascade = cv2.CascadeClassifier("frontalface.xml")

#yazı tonu
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    
    ret, frame  = vid.read()
    frame = cv2.flip(frame , 1)
    
    if ret is False:
        break
    
    #gray çeviriyoruz
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #face detection edelim
    faces = face_cascade.detectMultiScale(gray, 1.3, 7)
    
    #çizdirelim
    for (x,y,w,h) in faces:
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        
        #duygu modelimiz için input girişine uygun hale getiriyoruz
        roi_ = frame[y-15:y+h+15, x-15:x+w+15]
        roi_ = cv2.cvtColor(roi_, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(roi_ ,(48,48))
        roi = np.expand_dims(roi, axis = 0)
        roi = roi.reshape(-1,48,48,3).astype(np.float32)
        roi /= 255.0
        custom = model.predict(roi)
        
        dogruluk = str(int(custom[0][np.argmax(custom)]*100))
        cv2.putText(frame, (label[np.argmax(custom)]+" %"+dogruluk),(x+35,y-30) , font, 1,  (255,0,0),2)
        #man
        cv2.putText(frame, "m",(x+w,y+10) , font, 1,  (0,0,0),1)    
        cv2.line(frame,(x+w+30,y+5) ,(x+w+30+int(50*custom[0][0]),y+5) , (0,255,0),4)
        #woman
        cv2.putText(frame, "w",(x+w,y+40) , font, 1,  (0,0,0),1)
        cv2.line(frame,(x+w+30,y+35) ,(x+w+30+int(50*custom[0][1]),y+35) , (0,255,0),4)

        
    cv2.imshow("WebCam",frame)
    
    if cv2.waitKey(80) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()    