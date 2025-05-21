import cv2
import numpy as np
import csv
import matplotlib.pyplot as plt
import time
import os

# Çıktı klasörü
output_dir = "output_results"
os.makedirs(output_dir, exist_ok=True)

# Toplam mesafe sabit (örnek: 28.5 cm = 0.285 m)
total_distance_meters = 0.285

# Video yükle
cap = cv2.VideoCapture("output.mp4")
ret, frame = cap.read()
if not ret:
    print("Video açılamadı")
    exit()

# ROI seçimi
cv2.namedWindow("ROI Selection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ROI Selection", 640, 480)
roi = cv2.selectROI("ROI Selection", frame, False)
cv2.destroyWindow("ROI Selection")

# -----------------------------
# 1. CSRT TRACKER BAŞLANGICI
# -----------------------------
csrt_tracker = cv2.TrackerCSRT_create()
csrt_tracker.init(frame, roi)

csrt_frame_count = 0
csrt_prev_bbox = roi
csrt_tracking_start_time = None
csrt_tracking_end_time = None
csrt_no_movement_start_time = None
csrt_path_points = []
csrt_last_valid_frame = None

csrt_csv_path = os.path.join(output_dir, 'csrt_tracking_coordinates.csv')
csrt_csv = open(csrt_csv_path, 'w', newline='')
csrt_writer = csv.writer(csrt_csv)
csrt_writer.writerow(['Frame', 'X', 'Y', 'Time'])

# -----------------------------
# 2. OPTICAL FLOW BAŞLANGICI
# -----------------------------
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
roi_x, roi_y, roi_w, roi_h = roi
mask = np.zeros_like(old_gray)
mask[int(roi_y):int(roi_y + roi_h), int(roi_x):int(roi_x + roi_w)] = 255
p0 = cv2.goodFeaturesToTrack(old_gray, 25, 0.3, 7, mask=mask)
vector_data = []

# -----------------------------
# DÖNGÜ
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # CSRT Takibi
    success, bbox = csrt_tracker.update(frame)
    if success:
        if csrt_tracking_start_time is None:
            csrt_tracking_start_time = time.time()
        csrt_tracking_end_time = time.time()

        csrt_last_valid_frame = frame.copy()
        center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
        csrt_path_points.append(center)

        elapsed_time = csrt_tracking_end_time - csrt_tracking_start_time
        csrt_writer.writerow([csrt_frame_count, center[0], center[1], elapsed_time])

        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        if int(bbox[0]) == int(csrt_prev_bbox[0]) and int(bbox[1]) == int(csrt_prev_bbox[1]):
            if csrt_no_movement_start_time is None:
                csrt_no_movement_start_time = time.time()
            elif time.time() - csrt_no_movement_start_time > 2:
                print("CSRT: Nesne 2 saniye boyunca aynı konumda kaldı.")
                break
        else:
            csrt_no_movement_start_time = None

        csrt_prev_bbox = bbox
    else:
        print("CSRT: Takip başarısız oldu.")
        break

    # Optical Flow
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if p0 is not None:
        p1_optical, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None)
        if p1_optical is not None:
            good_new = p1_optical[st == 1]
            good_old = p0[st == 1]

            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                vector_data.append([c, d, a - c, b - d])
                cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 1, tipLength=0.3)

            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

    for pt in csrt_path_points:
        cv2.circle(frame, pt, 2, (0, 0, 255), -1)

    cv2.imshow("Tracking", frame)
    csrt_frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# KAPAT
# -----------------------------
csrt_csv.close()
cap.release()
cv2.destroyAllWindows()

# CSRT Görselleştirme
if csrt_last_valid_frame is not None:
    for pt in csrt_path_points:
        cv2.circle(csrt_last_valid_frame, pt, 2, (0, 0, 255), -1)
    csrt_img_path = os.path.join(output_dir, "csrt_tracked_path.jpg")
    cv2.imwrite(csrt_img_path, csrt_last_valid_frame)

# CSRT Ortalama Hız Hesabı
if csrt_tracking_start_time and csrt_tracking_end_time:
    total_time = csrt_tracking_end_time - csrt_tracking_start_time
    avg_speed = total_distance_meters / total_time
    print(f"CSRT Takip Süresi: {total_time:.2f} s")
    print(f"Ortalama Hız: {avg_speed:.2f} m/s")
else:
    print("CSRT: Süre hesaplanamadı.")

# Optical Flow Görselleştirme
if vector_data:
    vector_data = np.array(vector_data)
    magnitudes = np.sqrt(vector_data[:, 2]**2 + vector_data[:, 3]**2)
    mean_magnitude = np.mean(magnitudes)
    print(f"Optical Flow Ortalama Hareket Yoğunluğu: {mean_magnitude:.4f}")

    # CSV kaydet
    vector_csv_path = os.path.join(output_dir, 'optical_vector_data.csv')
    with open(vector_csv_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Start_X', 'Start_Y', 'Vector_X', 'Vector_Y'])
        writer.writerows(vector_data)

    # Quiver görseli
    plt.figure(figsize=(10, 10))
    plt.quiver(vector_data[:, 0], vector_data[:, 1],
               vector_data[:, 2], vector_data[:, 3],
               angles='xy', scale_units='xy', scale=1,
               color='blue', width=0.003)
    plt.gca().invert_yaxis()
    plt.title('Optical Flow Motion Vectors')
    vector_img_path = os.path.join(output_dir, 'optical_flow_vectors.png')
    plt.savefig(vector_img_path)
    plt.close()
else:
    print("Optical Flow: Vektör verisi yok.")
