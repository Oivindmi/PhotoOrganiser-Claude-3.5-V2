
import cv2
import numpy as np
import os
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import pickle

logger = logging.getLogger(__name__)


class ImageCache:
    def __init__(self, cache_file="image_similarity_cache.pkl"):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                return {}
        return {}

    def _save_cache(self):
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value
        self._save_cache()

    def exists(self, key):
        return key in self.cache

    def clear(self):
        self.cache = {}
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)


def compare_media_task(file_pair):
    return ImageComparison.compare_media(*file_pair)


class ImageComparison:
    _cache = ImageCache()
    TARGET_SIZE = (200, 200)

    @staticmethod
    def normalize_path(path):
        return os.path.normpath(path).encode('utf-8').decode('utf-8')

    @staticmethod
    def clear_cache():
        ImageComparison._cache.clear()
        logger.info("Image comparison cache cleared")

    @staticmethod
    def batch_compare_media(file_pairs):
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        uncached_pairs = []
        results = []

        for pair in file_pairs:
            cache_key = tuple(sorted(pair))
            if ImageComparison._cache.exists(cache_key):
                results.append(ImageComparison._cache.get(cache_key))
            else:
                uncached_pairs.append(pair)

        if uncached_pairs:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                new_results = list(executor.map(compare_media_task, uncached_pairs))

                for pair, result in zip(uncached_pairs, new_results):
                    cache_key = tuple(sorted(pair))
                    ImageComparison._cache.set(cache_key, result)

                results.extend(new_results)

        return results

    @staticmethod
    def compare_media(file1_path: str, file2_path: str) -> float:
        file1_path = ImageComparison.normalize_path(file1_path)
        file2_path = ImageComparison.normalize_path(file2_path)

        cache_key = tuple(sorted([file1_path, file2_path]))
        if ImageComparison._cache.exists(cache_key):
            return ImageComparison._cache.get(cache_key)

        if not os.path.exists(file1_path):
            logger.warning(f"File does not exist: {file1_path}")
            return 0
        if not os.path.exists(file2_path):
            logger.warning(f"File does not exist: {file2_path}")
            return 0

        try:
            is_video1 = ImageComparison.is_video(file1_path)
            is_video2 = ImageComparison.is_video(file2_path)

            if is_video1 and is_video2:
                similarity = ImageComparison.compare_videos(file1_path, file2_path)
            elif not is_video1 and not is_video2:
                similarity = ImageComparison.compare_images(file1_path, file2_path)
            else:
                image_path = file1_path if not is_video1 else file2_path
                video_path = file2_path if not is_video1 else file1_path
                similarity = ImageComparison.compare_image_to_video(image_path, video_path)

            ImageComparison._cache.set(cache_key, similarity)
            return similarity

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
        if img1 is None or img2 is None:
            return 0

        img1 = cv2.resize(img1, ImageComparison.TARGET_SIZE)
        img2 = cv2.resize(img2, ImageComparison.TARGET_SIZE)

        hist_sim = ImageComparison.histogram_similarity(img1, img2)

        if hist_sim > 0.5:
            feature_sim = ImageComparison.feature_similarity(img1, img2)
            combined_sim = 0.6 * hist_sim + 0.4 * feature_sim
        else:
            combined_sim = hist_sim

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
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=500)

        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if not kp1 or not kp2 or des1 is None or des2 is None:
            return 0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:50]

        num_good_matches = sum(1 for m in matches if m.distance < 40)
        max_possible_matches = min(len(kp1), len(kp2), 50)
        similarity = num_good_matches / max_possible_matches if max_possible_matches > 0 else 0

        return similarity