import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class ImageComparison:
    @staticmethod
    def normalize_path(path):
        return os.path.normpath(path).encode('utf-8').decode('utf-8')

    @staticmethod
    def compare_media(file1_path, file2_path):
        file1_path = ImageComparison.normalize_path(file1_path)
        file2_path = ImageComparison.normalize_path(file2_path)

        if not os.path.exists(file1_path):
            logger.warning(f"File does not exist: {file1_path}")
            return 0
        if not os.path.exists(file2_path):
            logger.warning(f"File does not exist: {file2_path}")
            return 0

        try:
            # Determine if files are images or videos
            is_video1 = ImageComparison.is_video(file1_path)
            is_video2 = ImageComparison.is_video(file2_path)

            if is_video1 and is_video2:
                return ImageComparison.compare_videos(file1_path, file2_path)
            elif not is_video1 and not is_video2:
                return ImageComparison.compare_images(file1_path, file2_path)
            else:
                # Handle image-video comparison
                image_path = file1_path if not is_video1 else file2_path
                video_path = file2_path if not is_video1 else file1_path
                return ImageComparison.compare_image_to_video(image_path, video_path)
        except Exception as e:
            logger.error(f"Error comparing files {file1_path} and {file2_path}: {str(e)}")
            return 0

    @staticmethod
    def is_video(file_path):
        _, ext = os.path.splitext(file_path)
        return ext.lower() in ['.mov', '.mp4', '.avi', '.mkv']

    @staticmethod
    def compare_images(img1_path, img2_path):
        img1 = cv2.imdecode(np.fromfile(img1_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        img2 = cv2.imdecode(np.fromfile(img2_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

        if img1 is None or img2 is None:
            logger.warning(f"Failed to read image: {img1_path} or {img2_path}")
            return 0

        return ImageComparison.compare_image_data(img1, img2)

    @staticmethod
    def compare_videos(video1_path, video2_path):
        frame1 = ImageComparison.extract_first_frame(video1_path)
        frame2 = ImageComparison.extract_first_frame(video2_path)

        if frame1 is None or frame2 is None:
            logger.warning(f"Failed to extract frame from video: {video1_path} or {video2_path}")
            return 0

        return ImageComparison.compare_image_data(frame1, frame2)

    @staticmethod
    def compare_image_to_video(image_path, video_path):
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        video_frame = ImageComparison.extract_first_frame(video_path)

        if img is None or video_frame is None:
            logger.warning(f"Failed to read image or extract video frame: {image_path} or {video_path}")
            return 0

        return ImageComparison.compare_image_data(img, video_frame)

    @staticmethod
    def compare_image_data(img1, img2):
        # Resize images for faster processing
        img1 = cv2.resize(img1, (300, 300))
        img2 = cv2.resize(img2, (300, 300))

        # Histogram comparison
        hist_sim = ImageComparison.histogram_similarity(img1, img2)

        # Feature matching
        feature_sim = ImageComparison.feature_similarity(img1, img2)

        # Combine similarities (you can adjust weights as needed)
        combined_sim = 0.5 * hist_sim + 0.5 * feature_sim

        return combined_sim

    @staticmethod
    def extract_first_frame(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    @staticmethod
    def histogram_similarity(img1, img2):
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    @staticmethod
    def feature_similarity(img1, img2):
        # Convert images to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Initialize ORB detector
        orb = cv2.ORB_create()

        # Find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        # Create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Calculate similarity based on number of good matches
        num_good_matches = sum(1 for m in matches if m.distance < 50)
        similarity = num_good_matches / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0

        return similarity