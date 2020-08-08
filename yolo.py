# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 01:20:41 2020

@author: ALPEREN
"""
#!/usr/bin/python
import cv2 # OpenCV kütüphanemiz
import numpy as np

classes = []
colors = []
img = None

# Sadece çıkış katmanlarını kullanmak
# için çağırmamız gereken fonksiyon
def get_output_layers(net):
    katman_isim = net.getLayerNames()
    output_layers = [katman_isim[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers 

# Belirlenen objelerin etrafına 
# kutu çizmemiz için 
# çağırdımız fonksiyon
def kutu_ciz(sinif_id, confidence, x,y,x_2,y_2):
    
    etiket = str(classes[sinif_id]) #Dosyadan sinif ismini okuyoruz
    color = colors[sinif_id] #Sınıfa özgü renk
    cv2.rectangle(img, (x,y),(x_2,y_2),color,2) #Dikdötrgen çizimi
    #Objenin ismini diktörgenin üstüne yazdırıyoruz
    cv2.putText(img, etiket,(x-10,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
    
    

#Giriş fonksiyonumuz
def main():
    global classes
    global colors
    global img
    img = cv2.imread("horse.jpg") # Resim dosyasını oku
    (uzunluk, genislik) = img.shape[:2] #Resim boyutu
    
    #Sınıf isimlerini dosyadan okuyoruz
    with open("yolov3.txt", 'r') as f:
        
        classes = [line.strip() for line in f.readlines()]
    
    #80,3 yani her sınıf için farklı renk oluşturuyoruz
    #np.array
    colors = np.random.uniform(0,255,size=(len(classes), 3))
    
    #DNN: deep neural network
    #Önceden eğitilmiş modelimizi okuyoruz
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    
    #Resmi önişleme
    #scale_factor = 1/255
    #Ortalama Çıkarma: yok
    #(416,416) algoritma için en uygun boyutlar
    #crop: false, 416,416 boyutuna resmi kesmeden getir.
    #True: BGR yerine RGB kullanmasını istiyoruz.
    blob = cv2.dnn.blobFromImage(img, 1/255, (416,416),True,crop=False)
    
    #Ağ için girdi
    net.setInput(blob)
    
    #Obje tanıma aşaması !!
    objeler = net.forward(get_output_layers(net))
    
    class_ids = []
    confidences = []
    boxes = []
    nms_th = 0.4
    
    for obje in objeler:
        for dedected in obje: 
            #Bulunan objeden gerekli olan 
            #değerleri alıyoruz
            skor = dedected[5:]
            class_id = np.argmax(skor)
            confidence = skor[class_id] 
            if confidence > 0.5: #%50 doğruluk olasılığı
                #Dikdörtgen için gerekli hesaplamalar
                center_x = int(dedected[0] * genislik)
                center_y = int(dedected[1] * uzunluk)
                w = int(dedected[2] * genislik)
                h = int(dedected[3] * uzunluk)
                x = center_x - w / 2
                y = center_y - h / 2 
                class_ids.append(class_id) #Sınıf id
                confidences.append(float(confidence)) #Confidence değerleri
                boxes.append([x,y,w,h]) #Dikdörtgenler
                
    #Birden fazla diktörgen çizmemek için
    #NMS uyguluyoruz
    indeksler = cv2.dnn.NMSBoxes(boxes,confidences, 0.5, nms_th)
    
    for i in indeksler:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        #Dikdörtgenimizi çizdiriyoruz
        kutu_ciz(class_ids[i],confidences[i],round(x), round(y), round(x+w), round(y+h))
        
        
    #resmi görüntüle
    cv2.imshow("Tanimlanan nesneler", img)
    #resmi kaydetmek için
    #cv2.imwrite("horse2.jpg",img) 
    cv2.waitKey()
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()
    
        