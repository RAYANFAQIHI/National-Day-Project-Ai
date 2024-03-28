import cv2 #استدعاء المكتبة
#الابعاد

MODEL_MEAN_VALUES =(78.4463377603,87.7689143744,114.895847746)
#انشاء ليست الاعمار
age_list =['(0,2)','(4,6)','(8,12)','(14,20)','(25,32)','(38,43)','(48,53)','(60,100)']
#ليست تحديد الجنس
gender_list=['Male','Female']
#استدعاء الملفات التي تتعرف على العمر والجنس
def filesGet():
    age_net = cv2.dnn.readNetFromCaffe(
         'data/deploy_age.prototxt',
         'data/age_net.caffemodel'

    )
    gender_net =cv2.dnn.readNetFromCaffe(
        'data/deploy_gender.prototxt',
        'data/gender_net.caffemodel'
    )
    return(age_net,gender_net)

def read_from_camera(age_net,gender_net):
    font= cv2.FONT_HERSHEY_SIMPLEX#نوع الخط
    imges=cv2.imread('imges/image.jpg')#استدعاء الصورة
    #الملف الخاص بتحديد الوجه
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    #تحديد نظام الالوان
    gray= cv2.cvtColor(imges, cv2.COLOR_BGR2GRAY)
    #كشف وجوه متعددة في الصورة واحدة
    faces= face_cascade.detectMultiScale(gray,1.1,5)
    if(len(faces)>0):#تحديد عدد الوجوه
        print("Found {} Faces".format(str(len(faces))))
    
    for(x,y,w,h)in faces:
        #  رسم مستطيل وتحديد الالوان 
        cv2.rectangle(imges, (x,y),(x+w, y+h),(255,255,0),2)
        #جلب وجه ونسخه ارسالها الى الخوارزمية 
        face_img = imges[y:y+h, h:h+w].copy()
        blob =cv2.dnn.blobFromImage(face_img, 1,(227,227),MODEL_MEAN_VALUES,swapRB=False)
        #توقع الجنس
        gender_net.setInput(blob)
        gender_p=gender_net.forward()#output
        gender = gender_list[gender_p[0].argmax()]
        print("Gender : " + gender)
        #توقع العمر
           #توقع الجنس
        age_net.setInput(blob)
        age_p=age_net.forward()#output
        age = age_list[age_p[0].argmax()]
        print("Age : " + gender)
        G_A = "%s %s" % (gender ,age)
        cv2.putText(imges, G_A, (x,y), font , 1 , (255,255,255),2,cv2.LINE_AA)
        cv2.imshow('RAKWAN',imges)
    cv2.waitKey(0)
if __name__ == "__main__":
    age_net, gender_net = filesGet()
    read_from_camera(age_net,gender_net)