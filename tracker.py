import cv2
import time

# Toplam mesafeyi 28,5 santimetre olarak kabul ediyoruz
total_distance_meters = 0.285

# Video dosyasını veya kamera kaynağını aç
cap = cv2.VideoCapture("output.mp4")  # '0' yerine video dosyası yolu da kullanılabilir

# İlk kareyi oku
ret, frame = cap.read()
if not ret:
    print("Kamera açılamadı veya video dosyası bulunamadı")
    exit()

# İzlemek istediğiniz nesnenin ilk konumunu seçin
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Tracking", 640, 480)
bbox = cv2.selectROI("Tracking", frame, False)

# Tracker oluştur
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)

# Zaman ve konum değişkenlerini başlat
start_time = time.time()
prev_bbox = bbox
no_movement_start_time = None
path_points = []

# Son geçerli kareyi saklamak için değişken
last_valid_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Yeni karede nesneyi izle
    success, bbox = tracker.update(frame)

    if success:
        # Son geçerli kareyi güncelle
        last_valid_frame = frame.copy()

        # İzlenen nesnenin merkez noktasını hesapla ve listeye ekle
        center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
        path_points.append(center)

        # İzlenen nesneyi çiz
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        # Hareket kontrolü
        if (int(bbox[0]) == int(prev_bbox[0]) and int(bbox[1]) == int(prev_bbox[1])):  # Hareket etmediği kabul edilen durum
            if no_movement_start_time is None:
                no_movement_start_time = time.time()
            elif time.time() - no_movement_start_time > 2:
                print("Nesne 2 saniye boyunca aynı konumda kaldı, videoyu bitiriyorum.")
                break
        else:
            no_movement_start_time = None

        # Zaman ve konum güncelleme
        prev_bbox = bbox
    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        print("Tracking failure detected, videoyu bitiriyorum.")
        break

    # Path noktasını çiz
    for point in path_points:
        cv2.circle(frame, point, 2, (0, 255, 0), -1)

    cv2.imshow('Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Son geçerli kareyi kaydet
if last_valid_frame is not None:
    for point in path_points:
        cv2.circle(last_valid_frame, point, 2, (0, 255, 0), -1)
    cv2.imwrite("tracked_path_last_frame.jpg", last_valid_frame)
else:
    print("Son geçerli kare bulunamadı, resim kaydedilemedi.")

cap.release()
cv2.destroyAllWindows()

# Toplam süre kullanarak ortalama hız hesaplama
end_time = time.time()
total_time_seconds = end_time - start_time

if total_time_seconds > 0:
    average_speed_meters_per_second = total_distance_meters / total_time_seconds
    print(f"Toplam Mesafe: {total_distance_meters:.2f} m"
          f"\nToplam Zaman: {total_time_seconds:.2f} s")
    print(f"Ortalama Hız: {average_speed_meters_per_second:.2f} m/s")
else:
    print("Hareket tespit edilmedi veya zaman ölçümü yapılamadı.")
