"""veri setimizi, veri klasörü içine olusturduğumuz modül"""

import cv2      #opencv kütüphanesi eklendi

#stream_url = f"rtsp://SimaProject:Yumurta1@192.168.1.117:554/stream1"
vid_cam = cv2.VideoCapture(0)   #(stream_url)video kamera tanımlandı

face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faceName = 1    #her farklı yüz için farklı numara tanımlanacak
number = 1      #çekilecek fotoğraf sayısı

while(True):
    _, frame = vid_cam.read()   #kamera okutuldu

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #gri tonlama eklendi
    
    #Eger yuze benzeyen ufak bolgeler karistiriliyorsa minSize = () eklenebilir. Default 130,130
    faces = face_detector.detectMultiScale(gray, 1.3, 5)#Resimdeki yüzlerin yerleri tespit edildi. Gri tonlamanın alt üst(koyuluk) sınırları belirlendi

    for(x, y, w, h) in faces: #yuzu algılayacak çerçeve ebatları 

        #cerceve kalınlıgı ve rengı
        cv2.rectangle(frame, (x + 5, y + 10), (x + w, y + h), (0, 0, 255), 2)
        number += 1

        #resimler veri klasörüne aşağıdaki şekilde yazdırılır
        cv2.imwrite("veri/egitim/Gulumsemeyen." + str(faceName) + '.' + str(number) + ".jpg", gray[y: y+h, x: x+w])
       
        #kameraya göster komutu atandı
        win_name = "cerceve"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 800, 600)
        cv2.imshow(win_name , frame)

        #kameradan çıkış tuşu ve gecikme süresi belirlendi
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    elif number > 1000: #çekilecek fotoğraf sayısı için üst sınır belirlendi
        break

vid_cam.release()   #kamera durduruldu
cv2.destroyAllWindows() #tum pencereler kapatıldı