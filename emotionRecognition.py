# andrean

# mengimport library yang diperlukan

# Importing required packages
from keras.models import load_model
import numpy as np
import argparse
import dlib
import cv2


# .ArgumentParser() digunakan untuk membuat ArgumentParser object yang menyimpan informasi yang diperlukan
# dan dimasukan saat menjalankan perintah Python di command line
ap = argparse.ArgumentParser()

# .add_argument digunakan untuk mengisi ArgumentParser dengan informasi tentang argumen program
ap.add_argument("-vw", "--isVideoWriter", type=bool, default=False)

# .parse_args() digunakan untuk menyimpan dan menggunakan informasi dari tahap sebelumnya
args = vars(ap.parse_args())

emotion_offsets = (20, 40)

# dictionary berisi key dan value dari emosi
emotions = {
    0: {
        "emotion": "Angry",
        "color": (193, 69, 42)
    },
    1: {
        "emotion": "Disgust",
        "color": (164, 175, 49)
    },
    2: {
        "emotion": "Fear",
        "color": (40, 52, 155)
    },
    3: {
        "emotion": "Happy",
        "color": (23, 164, 28)
    },
    4: {
        "emotion": "Sad",
        "color": (164, 93, 23)
    },
    5: {
        "emotion": "Suprise",
        "color": (218, 229, 97)
    },
    6: {
        "emotion": "Neutral",
        "color": (108, 72, 200)
    }
}


# fungsi ShapePoints digunakan untuk mengkonversi object ke dalam Numpy array
def shapePoints(shape):

    # inisialisasi  list koordinat (x,y)
    coords = np.zeros((68, 2), dtype="int")

    # melakukan perulangan 68 landmark wajah dan mengkonversi
    # ke dalam koordinat (x,y)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

# fungsi rectPoints digunakan untuk mengkonversi kotak persegi
# ke dalam format (x, y, w, h)
def rectPoints(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# faceLandmarks mengarah ke path dari file bernama shape_predictor_68_face_landmarks
# model file tersebut adalah facial point annotations yang digunakan untuk memberi anotasi 
# pada bagian-bagian wajah tertentu, seperti mata, alis, mulut, hidung, dan lain sebagainya
faceLandmarks = "faceDetection/models/dlib/shape_predictor_68_face_landmarks.dat"

# .get_frontal_face_detector digunakan untuk mendeteksi keberadaan suatu wajah dalam frame
detector = dlib.get_frontal_face_detector()

# .shape_predictor dingunakan untuk mengambil gambar yang berisi wajah atau objek
# dan anotasi-anotasi untuk menentukan pose dari wajah atau objek yang berasal dari shape_predictor_68_face_landmarks
predictor = dlib.shape_predictor(faceLandmarks)

# emotionModelPath mengarah ke path dari pre-trained model yaitu fer2013_mini_XCEPTION.110-0.65
# pre-trained model ini digunakan untuk klasifikasi emosi seseorang berdasarkan titik anotasi sebelumnya
# terdapat 7 emosi yang akan diklasifikasikan yaitu 'Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral'
emotionModelPath = 'models/emotionModel.hdf5'  # fer2013_mini_XCEPTION.110-0.65

# melakukan load dari pre-trained model
emotionClassifier = load_model(emotionModelPath, compile=False)

# input_shape untuk data awal
emotionTargetSize = emotionClassifier.input_shape[1:3]

# VideoCapture(0) mengambil video dari kamera pertama (utama) dari suatu komputer
cap = cv2.VideoCapture(0)

# ketika 'isVideoWriter' bernilai True
if args["isVideoWriter"] == True:
    # fourcc memiliki arti 4 byte untuk menentukan codec video, MJPG berarti motion JPEG
    fourrcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    # mendapatkan frame lebar dari video dengan default 640x480px
    capWidth = int(cap.get(3))
    # mendapatkan frame tinggi dari video dengan default 640x480px
    capHeight = int(cap.get(4))
    # output video dengan parameter berturut-turut adalah 
    # nama file, 4 codec, jumlah FPS, dan lebar tinggi frame video
    videoWrite = cv2.VideoWriter("output.avi", fourrcc, 22,
                                 (capWidth, capHeight))

 # rizka  
while True:
    # memuat gambar yang di input dan ubah ukurannya
    ret, frame = cap.read()
    frame = cv2.resize(frame, (720, 480))
    if not ret:
        break
        
    # convert BGR ke HSV (mengubah ke grayscale)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(grayFrame, 0)
    for rect in rects:
        # Menentukan titik-titik wajah pada area wajah
        shape = predictor(grayFrame, rect)
        # Menandai wajah dengan  koordinat (x, y)
        points = shapePoints(shape)
        # Mendefiniskan kotak pembatas di sekitar wajah yang terdeteksi pada gambar
        (x, y, w, h) = rectPoints(rect)
        # Gambar indeks wajah
        grayFace = grayFrame[y:y + h, x:x + w]
        try:
            grayFace = cv2.resize(grayFace, (emotionTargetSize))
        except:
            continue

        grayFace = grayFace.astype('float32')
        grayFace = grayFace / 255.0
        grayFace = (grayFace - 0.5) * 2.0
        grayFace = np.expand_dims(grayFace, 0)
        grayFace = np.expand_dims(grayFace, -1)
        emotion_prediction = emotionClassifier.predict(grayFace)
        emotion_probability = np.max(emotion_prediction)

        # jika emotion probability lebih dari 0.36
        if (emotion_probability > 0.36):
            # membuat prediksi dan menampungnya pada emotion_label_arg 
            # berdasarkan array emotion_prediction yang memiliki nilai index maksimum 
            emotion_label_arg = np.argmax(emotion_prediction)
            # mengakses item color pada dict emotions 
            # dengan nilai key yang ada pada emotion_label_arg
            color = emotions[emotion_label_arg]['color']
            # menggambar bounding box di sekitaran objek face yang terdeteksi
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.line(frame, (x, y + h), (x + 20, y + h + 20), color, thickness=2)
            cv2.rectangle(frame, (x + 20, y + h + 20), (x + 130, y + h + 40), color, -1)
            # menampilkan gambar face yang terdeteksi di frame 
            # dengan memberikan caption/label berupa prediksi emosi 
            cv2.putText(frame, emotions[emotion_label_arg]['emotion']+ ' '+ str(round(emotion_probability,2)),
                        (x + 25, y + h + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        # kondisi else jika kondisi pertama tidak terpenuhi
        else:
            # inisialisasi color dengan nilai dibawah
            color = (255, 255, 255)
            # membuat rectangle dengan 
            # informasi color yang didefinisikan diatas 
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # jika 'isVideoWriter' bernilai True
    if args["isVideoWriter"] == True:
        # write output frame ke file 'output.avi'
        videoWrite.write(frame)

    # menampilkan output frame
    cv2.imshow("Emotion Recognition", frame)
    # exit jika tombol esc ditekan 
    k = cv2.waitKey(1) & 0xFF #hexadecimal
    if k == 27:
        break

# release video capture dan video write jika semuanya telah selesai
cap.release()
if args["isVideoWriter"] == True:
    videoWrite.release()
# cleanup, digunakan menutup semua frame
cv2.destroyAllWindows()
