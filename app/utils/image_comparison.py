import cv2
import numpy as np
import os
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import pickle
from typing import List, Tuple, Optional
import json

logger = logging.getLogger(__name__)

def compare_media_pair(pair: Tuple[str, str]) -> float:
    from app.utils.image_comparison import ImageComparison  # Import inside function
    try:
        return ImageComparison.compare_media(pair[0], pair[1])
    except Exception as e:
        logging.error(f"Error in compare_media_pair: {str(e)}")
        return 0.0
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
        if not self.cache:
            return
        try:
            # Create a temp file first
            temp_file = f"{self.cache_file}.tmp"
            with open(temp_file, "wb") as f:
                pickle.dump(self.cache, f)
            # Then rename it to the actual cache file
            if os.path.exists(self.cache_file):
                try:
                    os.remove(self.cache_file)
                except Exception:
                    pass
            os.rename(temp_file, self.cache_file)
        except Exception as e:
            logger.error(f"Error saving cache: {str(e)}")
            # Clean up temp file if it exists
            if os.path.exists(f"{self.cache_file}.tmp"):
                try:
                    os.remove(f"{self.cache_file}.tmp")
                except Exception:
                    pass

    def safe_clear(self):
        try:
            # First clear the memory cache
            self.cache = {}
            # Try to save the empty cache
            try:
                with open(self.cache_file, "wb") as f:
                    pickle.dump({}, f)
            except:
                pass
        except Exception as e:
            logger.error(f"Error in safe_clear: {str(e)}")

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value
        self._save_cache()

    def exists(self, key):
        return key in self.cache

    def clear(self):
        try:
            self.cache = {}
            # Save empty cache first
            self._save_cache()
            # Then try to remove the file
            if os.path.exists(self.cache_file):
                try:
                    os.remove(self.cache_file)
                except PermissionError:
                    logger.warning(f"Could not remove cache file {self.cache_file} - it may be in use")
                except Exception as e:
                    logger.error(f"Error removing cache file: {str(e)}")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            # Ensure cache is empty even if file operations fail
            self.cache = {}
class ImageComparison:
    _cache = ImageCache()
    TARGET_SIZE = (200, 200)
    _has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0

    BATCH_SIZE = 128
    TARGET_SIZE = (200, 200)


    @classmethod
    def init_gpu(cls) -> None:
        try:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if device_count > 0:
                cls.has_cuda = True
                cls.device = cv2.cuda.getDevice()
                logging.info(f"GPU acceleration enabled")
            else:
                logging.warning("No CUDA devices available, using CPU")
        except Exception as e:
            logging.error(f"Error initializing GPU: {str(e)}")
            cls.has_cuda = False
            cls.device = None

    @classmethod
    def batch_compare_media(cls, file_pairs: List[Tuple[str, str]]) -> List[float]:
        results = []
        num_batches = (len(file_pairs) + cls.BATCH_SIZE - 1) // cls.BATCH_SIZE

        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(num_batches):
                start_idx = i * cls.BATCH_SIZE
                end_idx = min(start_idx + cls.BATCH_SIZE, len(file_pairs))
                batch_pairs = file_pairs[start_idx:end_idx]
                future = executor.submit(cls._process_batch, batch_pairs)
                futures.append(future)

            for future in futures:
                results.extend(future.result())

        return results

    @classmethod
    def _process_batch(cls, batch_pairs: List[Tuple[str, str]]) -> List[float]:
        batch_results = []

        # Load images for the batch
        batch_images_a = []
        batch_images_b = []
        for pair in batch_pairs:
            img_a = cls._load_image(pair[0])
            img_b = cls._load_image(pair[1])
            if img_a is not None and img_b is not None:
                batch_images_a.append(img_a)
                batch_images_b.append(img_b)

        # Transfer images to GPU
        gpu_images_a = cv2.cuda_GpuMat()
        gpu_images_b = cv2.cuda_GpuMat()
        gpu_images_a.upload(np.stack(batch_images_a))
        gpu_images_b.upload(np.stack(batch_images_b))

        # Perform comparisons on GPU
        gpu_similarities = cv2.cuda.compareHist(gpu_images_a, gpu_images_b, cv2.HISTCMP_CORREL)

        # Transfer results back to CPU
        similarities = gpu_similarities.download()

        # Process results
        for similarity in similarities:
            if similarity >= cls.SIMILARITY_THRESHOLD:
                batch_results.append(similarity)
            else:
                batch_results.append(0.0)

        return batch_results
    """
    @staticmethod
    def batch_compare_media(file_pairs: List[Tuple[str, str]]) -> List[float]:
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
                new_results = list(executor.map(compare_media_pair, uncached_pairs))

                for pair, result in zip(uncached_pairs, new_results):
                    cache_key = tuple(sorted(pair))
                    ImageComparison._cache.set(cache_key, result)

                results.extend(new_results)

        return results
    """

    @staticmethod
    def _load_image(file_path: str) -> Optional[np.ndarray]:
        try:
            # Read image file into a byte array
            with open(file_path, 'rb') as f:
                byte_array = bytearray(f.read())
                img_array = np.asarray(byte_array, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                logging.warning(f"Failed to load image: {file_path}")
                return None
            return cv2.resize(img, ImageComparison.TARGET_SIZE)
        except Exception as e:
            logging.error(f"Error loading image {file_path}: {str(e)}")
            return None

    @classmethod
    def compare_images(cls, img1_path: str, img2_path: str) -> float:
        img1 = cls._load_image(img1_path)
        img2 = cls._load_image(img2_path)

        if img1 is None or img2 is None:
            return 0.0

        try:
            if cls.has_cuda:
                return cls._compare_images_gpu(img1, img2)
            return cls._compare_images_cpu(img1, img2)
        except Exception as e:
            logging.error(f"Error comparing images: {str(e)}")
            return 0.0

    @classmethod
    def _compare_images_gpu(cls, img1: np.ndarray, img2: np.ndarray) -> float:
        try:
            # Upload images to GPU
            gpu_img1 = cv2.cuda_GpuMat()
            gpu_img2 = cv2.cuda_GpuMat()
            gpu_img1.upload(img1)
            gpu_img2.upload(img2)

            # Convert to grayscale
            gray1 = cv2.cuda.cvtColor(gpu_img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cuda.cvtColor(gpu_img2, cv2.COLOR_BGR2GRAY)

            # Calculate histograms
            hist_sim = cls._histogram_similarity_gpu(gpu_img1, gpu_img2)

            if hist_sim > 0.5:
                # Download for feature matching (not available on GPU)
                cpu_gray1 = gray1.download()
                cpu_gray2 = gray2.download()
                feature_sim = cls._feature_similarity_cpu(cpu_gray1, cpu_gray2)
                return 0.6 * hist_sim + 0.4 * feature_sim

            return hist_sim
        except cv2.error as e:
            logging.error(f"GPU processing error: {str(e)}, falling back to CPU")
            return cls._compare_images_cpu(img1, img2)

    @staticmethod
    def _compare_images_cpu(img1: np.ndarray, img2: np.ndarray) -> float:
        hist_sim = ImageComparison._histogram_similarity_cpu(img1, img2)

        if hist_sim > 0.5:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            feature_sim = ImageComparison._feature_similarity_cpu(gray1, gray2)
            return 0.6 * hist_sim + 0.4 * feature_sim

        return hist_sim

    @staticmethod
    def _histogram_similarity_gpu(gpu_img1: cv2.cuda_GpuMat, gpu_img2: cv2.cuda_GpuMat) -> float:
        try:
            hist1 = cv2.cuda.calcHist(gpu_img1)
            hist2 = cv2.cuda.calcHist(gpu_img2)

            # Download histograms for comparison (CPU)
            cpu_hist1 = hist1.download()
            cpu_hist2 = hist2.download()

            return cv2.compareHist(cpu_hist1, cpu_hist2, cv2.HISTCMP_CORREL)
        except Exception:
            # Fallback to CPU if GPU histogram fails
            img1 = gpu_img1.download()
            img2 = gpu_img2.download()
            return ImageComparison._histogram_similarity_cpu(img1, img2)

    @staticmethod
    def _histogram_similarity_cpu(img1: np.ndarray, img2: np.ndarray) -> float:
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

    @staticmethod
    def _feature_similarity_cpu(gray1: np.ndarray, gray2: np.ndarray) -> float:
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)

        if not kp1 or not kp2 or des1 is None or des2 is None:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:50]

        good_matches = sum(1 for m in matches if m.distance < 40)
        max_matches = min(len(kp1), len(kp2), 50)
        return good_matches / max_matches if max_matches > 0 else 0.0


# Initialize GPU on module import
print("Available methods:", [method for method in dir(ImageComparison) if not method.startswith('_')])
ImageComparison.init_gpu()