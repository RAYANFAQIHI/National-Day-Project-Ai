import cv2

MODEL_MEAN_VALUES = (78.4463377603, 87.7689143744, 114.895847746)
age_list = ['(0,2)', '(4,6)', '(8,12)', '(15,20)', '(24,30)', '(35,43)', '(46,53)', '(60,100)']
gender_list = ['Male', 'Female']

def filesGet():
    age_net = cv2.dnn.readNetFromCaffe(
        'data/deploy_age.prototxt',
        'data/age_net.caffemodel'
    )
    gender_net = cv2.dnn.readNetFromCaffe(
        'data/deploy_gender.prototxt',
        'data/gender_net.caffemodel'
    )
    return age_net, gender_net

def read_from_camera(age_net, gender_net):
    font = cv2.FONT_HERSHEY_SIMPLEX
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

    cap = cv2.VideoCapture(0)  # Open the default camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            face_img = frame[y:y + h, x:x + w].copy()
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            # Gender prediction
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]

            # Age prediction
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            
            G_A = "{} {}".format(gender, age)
            cv2.putText(frame, G_A, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__== "__main__":
    age_net, gender_net = filesGet()
    read_from_camera(age_net,gender_net)
