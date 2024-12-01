import cv2
import os


def preprocess_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Error: Dosya yüklenemedi. Dosya yolunu ve dosyanın mevcut olduğunu kontrol edin.")
        return None

    # Resmi gri tonlamaya çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur uygulayarak gürültüyü azalt
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Kenarları belirginleştirmek için Canny Edge Detection uygulayın (eşik değerlerini ayarlayın)
    edges = cv2.Canny(blurred, 30, 100)

    # Kontrastı artırmak için Histogram Equalization uygulayın
    equalized = cv2.equalizeHist(gray)

    # Filtrelenmiş resmi kaydet (isteğe bağlı)
    # cv2.imwrite("filtered_image.jpg", edges)

    return edges


def create_video_from_preprocessed_frames(frames_dir, output_video_path, fps=30):
    frames = [os.path.join(frames_dir, frame) for frame in sorted(os.listdir(frames_dir)) if frame.endswith(".jpg")]
    if not frames:
        print("No frames found in the directory.")
        return

    # Read the first frame to get the size
    first_frame = preprocess_image(frames[0])
    if first_frame is None:
        print("Error processing the first frame.")
        return
    height, width = first_frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use the correct codec
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_path in frames:
        preprocessed_frame = preprocess_image(frame_path)
        if preprocessed_frame is not None:
            video.write(cv2.cvtColor(preprocessed_frame, cv2.COLOR_GRAY2BGR))

    video.release()
    print(f"Video saved at {output_video_path}")


# Example usage
frames_directory = r"C:\Users\mmert\PycharmProjects\ObjectTrackingProject\images"  # Directory containing preprocessed images
output_video_file = r"../preprocess_output.mp4"  # Output video file path

create_video_from_preprocessed_frames(frames_directory, output_video_file)
