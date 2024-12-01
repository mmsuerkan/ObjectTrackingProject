import cv2
import numpy as np

# Resmi yükle
image_path = r"/image.jpeg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Dosya yüklenemedi. Dosya yolunu ve dosyanın mevcut olduğunu kontrol edin.")
else:
    # Resmi gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur uygulayarak gürültüyü azalt
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Kenarları belirginleştirmek için Canny Edge Detection uygulayın (eşik değerlerini ayarlayın)
    edges = cv2.Canny(blurred, 30, 100)

    # Kontrastı artırmak için Histogram Equalization uygulayın
    equalized = cv2.equalizeHist(gray)


    # Filtrelenmiş resmi göster
    cv2.imshow("Filtered Image", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Filtrelenmiş resmi kaydet
    cv2.imwrite("../filtered_image.jpg", edges)
