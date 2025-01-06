import os
import logging
import cv2
import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.utils.image_cache import LRUCache


class ImageComparison:
    _cache = None
    TARGET_SIZE = (100, 100)  # Reduced from 200x200
    BATCH_SIZE = 256  # Increased from 128
    MIN_GPU_MEM_REQUIRED = 512 * 1024 * 1024  # Reduced to 512MB
    _NUM_THREADS = min(16, os.cpu_count() * 2)  # More aggressive threading
    _image_cache = LRUCache(200)  # Doubled cache size
    _db_manager = None
    has_cuda = False

    @classmethod
    def init_gpu(cls):
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                cls.has_cuda = True
                logging.info("CUDA is available and enabled")
            else:
                cls.has_cuda = False
                logging.info("CUDA is not available, using CPU")
        except Exception as e:
            cls.has_cuda = False
            logging.error(f"Error initializing GPU: {str(e)}")

    @classmethod
    def set_db_manager(cls, db_manager):
        cls._db_manager = db_manager

    @classmethod
    def _compare_images_gpu(cls, img1: np.ndarray | str, img2: np.ndarray | str) -> float:
        try:
            if isinstance(img1, str):
                img1 = cls._load_image(img1)
            if isinstance(img2, str):
                img2 = cls._load_image(img2)

            if img1 is None or img2 is None:
                return 0.0

            if cls.has_cuda:
                try:
                    gpu_img1 = cv2.cuda_GpuMat()
                    gpu_img2 = cv2.cuda_GpuMat()

                    # Convert to grayscale before upload
                    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

                    gpu_img1.upload(gray1)
                    gpu_img2.upload(gray2)

                    # Calculate histogram similarity
                    hist_sim = cls._histogram_similarity_gpu(gpu_img1, gpu_img2)

                    if hist_sim > 0.5:
                        feature_sim = cls._feature_similarity_cpu(gray1, gray2)
                        similarity = 0.6 * hist_sim + 0.4 * feature_sim
                    else:
                        similarity = hist_sim

                    gpu_img1.release()
                    gpu_img2.release()

                    return similarity

                except cv2.error:
                    return cls._compare_images_cpu(img1, img2)
            else:
                return cls._compare_images_cpu(img1, img2)

        except Exception as e:
            logging.error(f"Error comparing images: {str(e)}")
            return 0.0

    @classmethod
    def _load_image(cls, file_path: str) -> np.ndarray | None:
        try:
            img = cls._image_cache.get(file_path)
            if img is not None:
                return img

            if not os.path.exists(file_path):
                return None

            file_bytes = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                return None

            img = cv2.resize(img, cls.TARGET_SIZE)
            cls._image_cache.put(file_path, img)
            return img

        except Exception as e:
            logging.error(f"Error loading image {file_path}: {str(e)}")
            return None

    @classmethod
    def batch_compare_media(cls, file_pairs: List[Tuple[str, str]]) -> List[float]:
        if not file_pairs:
            return []

        with ThreadPoolExecutor(max_workers=cls._NUM_THREADS) as executor:
            futures = []
            for i in range(0, len(file_pairs), cls.BATCH_SIZE):
                batch = file_pairs[i:i + cls.BATCH_SIZE]
                futures.append(executor.submit(cls._process_batch, batch))

            results = []
            for future in as_completed(futures):
                results.extend(future.result())

        return results

    @classmethod
    def _process_batch(cls, batch_pairs: List[Tuple[str, str]]) -> List[float]:
        return [cls._compare_images_gpu(pair[0], pair[1]) for pair in batch_pairs]

    @classmethod
    def _compare_images_cpu(cls, img1: np.ndarray, img2: np.ndarray) -> float:
        try:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            hist_sim = cls._histogram_similarity_cpu(gray1, gray2)
            if hist_sim > 0.5:
                feature_sim = cls._feature_similarity_cpu(gray1, gray2)
                return 0.6 * hist_sim + 0.4 * feature_sim
            return hist_sim

        except Exception as e:
            logging.error(f"Error in CPU comparison: {str(e)}")
            return 0.0

    @staticmethod
    def _histogram_similarity_cpu(img1: np.ndarray, img2: np.ndarray) -> float:
        try:
            hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        except Exception as e:
            logging.error(f"Error calculating histogram similarity: {str(e)}")
            return 0.0

    @staticmethod
    def _histogram_similarity_gpu(gpu_img1: cv2.cuda_GpuMat, gpu_img2: cv2.cuda_GpuMat) -> float:
        try:
            img1 = gpu_img1.download()
            img2 = gpu_img2.download()

            hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        except Exception as e:
            logging.error(f"Error calculating GPU histogram similarity: {str(e)}")
            return 0.0

    @staticmethod
    def _feature_similarity_cpu(img1: np.ndarray, img2: np.ndarray) -> float:
        try:
            orb = cv2.ORB_create(nfeatures=500)
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)

            if not kp1 or not kp2 or des1 is None or des2 is None:
                return 0.0

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:50]

            good_matches = sum(1 for m in matches if m.distance < 40)
            max_matches = min(len(kp1), len(kp2), 50)

            return good_matches / max_matches if max_matches > 0 else 0.0

        except Exception as e:
            logging.error(f"Error calculating feature similarity: {str(e)}")
            return 0.0


# Initialize GPU on module import
ImageComparison.init_gpu()