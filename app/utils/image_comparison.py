import cv2
import numpy as np
import os
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict
import json
import pickle
from app.models.database_manager import FileMetadata
from app.utils.image_cache import LRUCache
import time

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
    _cache = None  # Remove ImageCache reference since we're using LRUCache now
    TARGET_SIZE = (200, 200)
    BATCH_SIZE = 128
    MIN_GPU_MEM_REQUIRED = 1024 * 1024 * 1024  # 1GB
    _db_manager = None
    _image_cache = LRUCache(100)  # Cache last 100 images
    _NUM_THREADS = 4  # Adjust based on CPU cores

    # Add database manager as class variable
    _db_manager = None

    @classmethod
    def set_db_manager(cls, db_manager):
        cls._db_manager = db_manager

    @classmethod
    def _compare_pair(cls, pair: Tuple[str, str]) -> float:
        try:
            img1 = cls._load_cached_image(pair[0])
            img2 = cls._load_cached_image(pair[1])

            if img1 is None or img2 is None:
                return 0.0

            return cls._compare_images_gpu(img1, img2)
        except Exception as e:
            logging.error(f"Error comparing pair {pair}: {str(e)}")
            return 0.0

    @classmethod
    def _load_cached_image(cls, path: str) -> Optional[np.ndarray]:
        img = cls._image_cache.get(path)
        if img is not None:
            return img

        img = cls._load_image(path)
        if img is not None:
            cls._image_cache.put(path, img)
        return img


    @classmethod
    def init_gpu(cls) -> None:
        cls.has_cuda = False
        cls.device = None
        cls.gpu_mem_available = 0

        try:
            device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if device_count <= 0:
                logging.warning("No CUDA devices available, using CPU")
                return

            max_memory = 0
            best_device = 0

            for device in range(device_count):
                cv2.cuda.setDevice(device)
                device_info = cv2.cuda.DeviceInfo()
                free_memory = device_info.freeMemory()

                if free_memory > max_memory:
                    max_memory = free_memory
                    best_device = device

                logging.info(f"GPU {device}")
                logging.info(f"- Total memory: {device_info.totalMemory() / (1024 * 1024):.2f} MB")
                logging.info(f"- Free memory: {free_memory / (1024 * 1024):.2f} MB")
                logging.info(f"- Compute capability: {device_info.majorVersion()}.{device_info.minorVersion()}")

            cls.device = best_device
            cv2.cuda.setDevice(cls.device)
            cls.gpu_mem_available = max_memory

            test_sizes = [(64, 64), (128, 128), (256, 256)]
            max_test_size = None

            for size in test_sizes:
                try:
                    test_img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
                    gpu_mat = cv2.cuda_GpuMat()
                    gpu_mat.upload(test_img)
                    processed = cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_BGR2GRAY)
                    processed.download()
                    gpu_mat.release()
                    processed.release()
                    max_test_size = size
                except cv2.error as e:
                    logging.warning(f"Failed GPU test at size {size}: {str(e)}")
                    break

            if max_test_size:
                cls.has_cuda = True
                logging.info(f"GPU acceleration enabled - Max test size: {max_test_size}")
                logging.info(f"Using GPU {cls.device} with {cls.gpu_mem_available / (1024 * 1024):.2f} MB free memory")
            else:
                raise RuntimeError("GPU tests failed at minimum size")

        except Exception as e:
            logging.error(f"Error initializing GPU: {str(e)}")
            cls.has_cuda = False
            cls.device = None
            cls.gpu_mem_available = 0

    @classmethod
    def check_gpu_memory(cls, required_memory: int) -> bool:
        if not cls.has_cuda:
            return False

        try:
            info = cv2.cuda.DeviceInfo()
            free_mem = info.freeMemory()
            return free_mem >= required_memory
        except Exception as e:
            logging.error(f"Error checking GPU memory: {str(e)}")
            return False

    @classmethod
    def can_use_gpu(cls, image_size: tuple[int, int]) -> bool:
        if not cls.has_cuda:
            return False

        required_mem = image_size[0] * image_size[1] * 3 * 4  # Estimate memory needed
        return cls.check_gpu_memory(required_mem)

    @classmethod
    def batch_compare_media(cls, file_pairs: List[Tuple[str, str]]) -> List[float]:
        if not file_pairs:
            return []

        start_time = time.time()
        logging.info(f"Starting comparison of {len(file_pairs)} pairs")

        results = []
        batch_size = cls.BATCH_SIZE
        num_batches = (len(file_pairs) + batch_size - 1) // batch_size

        for i in range(num_batches):
            batch_start = time.time()
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(file_pairs))
            batch_pairs = file_pairs[start_idx:end_idx]

            with ThreadPoolExecutor(max_workers=cls._NUM_THREADS) as executor:
                batch_results = list(executor.map(cls._compare_pair, batch_pairs))
                results.extend(batch_results)

            logging.info(f"Batch {i + 1}/{num_batches} took: {time.time() - batch_start:.2f} seconds")

        logging.info(f"Total comparison time: {time.time() - start_time:.2f} seconds")
        return results

    @classmethod
    def _compare_frames(cls, frames1: List[str], frames2: List[str]) -> float:
        start_time = time.time()
        max_similarity = 0.0
        loaded_frames = {}  # Cache for loaded frames

        # Load first frame of each list
        try:
            first1 = cls._load_and_cache_frame(frames1[0], loaded_frames)
            first2 = cls._load_and_cache_frame(frames2[0], loaded_frames)
            if first1 is not None and first2 is not None:
                similarity = cls._compare_images_gpu(first1, first2)
                if similarity > 0.8:  # Early exit on good match
                    return similarity
                max_similarity = max(max_similarity, similarity)
        except Exception as e:
            logging.error(f"Error comparing first frames: {str(e)}")

        # Only continue if first comparison wasn't good enough
        for frame1_path in frames1[1:]:
            frame1 = cls._load_and_cache_frame(frame1_path, loaded_frames)
            if frame1 is None:
                continue

            for frame2_path in frames2[1:]:
                frame2 = cls._load_and_cache_frame(frame2_path, loaded_frames)
                if frame2 is None:
                    continue

                try:
                    similarity = cls._compare_images_gpu(frame1, frame2)
                    max_similarity = max(max_similarity, similarity)

                    if max_similarity > 0.8:  # Early exit on good match
                        return max_similarity
                except Exception as e:
                    logging.error(f"Error comparing frames {frame1_path} and {frame2_path}: {str(e)}")

        logging.info(f"Frame comparison took: {time.time() - start_time:.2f} seconds")
        return max_similarity
    @classmethod
    def _compare_with_video(cls, file1: FileMetadata, file2: FileMetadata) -> float:
        if file1.is_video and file2.is_video:
            return cls._compare_videos(file1.video_frames, file2.video_frames)
        elif file1.is_video:
            return cls._compare_video_with_image(file1.video_frames, file2.file_path)
        else:
            return cls._compare_video_with_image(file2.video_frames, file1.file_path)

    @classmethod
    def _load_and_cache_frame(cls, frame_path: str, cache: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        if frame_path not in cache:
            frame = cls._load_image(frame_path)
            if frame is not None:
                cache[frame_path] = frame
        return cache.get(frame_path)

    @classmethod
    def _compare_video_with_image(cls, video_frames: List[str], image_path: str) -> float:
        max_similarity = 0.0

        for frame in video_frames:
            similarity = cls._compare_images_gpu(frame, image_path)
            max_similarity = max(max_similarity, similarity)

            if max_similarity > 0.8:  # Early exit if we find a very good match
                return max_similarity

        return max_similarity

    @classmethod
    def _load_and_cache_image(cls, path: str, cache: Dict[str, cv2.cuda_GpuMat]) -> Optional[cv2.cuda_GpuMat]:
        if path not in cache:
            img = cls._load_image(path)
            if img is None:
                return None
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(img)
            cache[path] = gpu_mat
        return cache[path]

    @classmethod
    def _process_batch_cpu(cls, batch_pairs: List[Tuple[str, str]]) -> List[float]:
        results = []
        for pair in batch_pairs:
            img_a = cls._load_image(pair[0])
            img_b = cls._load_image(pair[1])
            if img_a is None or img_b is None:
                results.append(0.0)
                continue
            similarity = cls._compare_images_cpu(img_a, img_b)
            results.append(similarity)
        return results

    @classmethod
    def _process_batch(cls, batch_pairs: List[Tuple[str, str]]) -> List[float]:
        batch_results = []

        try:
            for pair in batch_pairs:
                try:
                    img_a = cls._load_image(pair[0])
                    img_b = cls._load_image(pair[1])

                    if img_a is None or img_b is None:
                        batch_results.append(0.0)
                        continue

                    if cls.has_cuda and cls.can_use_gpu(img_a.shape[:2]):
                        try:
                            # Convert to grayscale first
                            gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
                            gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

                            # Upload individual images
                            gpu_a = cv2.cuda_GpuMat()
                            gpu_b = cv2.cuda_GpuMat()
                            gpu_a.upload(gray_a)
                            gpu_b.upload(gray_b)

                            # Calculate and combine similarities
                            hist_sim = cls._histogram_similarity_gpu(gpu_a, gpu_b)
                            if hist_sim > 0.5:
                                feature_sim = cls._feature_similarity_cpu(gray_a, gray_b)
                                similarity = 0.6 * hist_sim + 0.4 * feature_sim
                            else:
                                similarity = hist_sim

                            gpu_a.release()
                            gpu_b.release()

                        except cv2.error as e:
                            logging.warning(f"GPU comparison failed, falling back to CPU: {str(e)}")
                            similarity = cls._compare_images_cpu(img_a, img_b)
                    else:
                        similarity = cls._compare_images_cpu(img_a, img_b)

                    batch_results.append(similarity)
                except Exception as e:
                    logging.error(f"Error processing image pair: {str(e)}")
                    batch_results.append(0.0)

        except Exception as e:
            logging.error(f"Error in _process_batch: {str(e)}")

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

    @classmethod
    def _load_image(cls, file_path: str) -> Optional[np.ndarray]:
        try:
            if not os.path.exists(file_path):
                logging.warning(f"File does not exist: {file_path}")
                return None

            # Normalize path to handle potential issues
            file_path = os.path.normpath(file_path)

            # Always use binary reading for better compatibility
            with open(file_path, 'rb') as f:
                file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img is None:
                    logging.warning(f"Failed to decode image: {file_path}")
                    return None

                try:
                    return cv2.resize(img, cls.TARGET_SIZE)
                except Exception as resize_error:
                    logging.error(f"Error resizing image {file_path}: {str(resize_error)}")
                    return None

        except PermissionError:
            logging.error(f"Permission denied accessing file: {file_path}")
            return None
        except IOError as io_error:
            logging.error(f"IO Error reading file {file_path}: {str(io_error)}")
            return None
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
            if cls.has_cuda and cls.can_use_gpu(img1.shape[:2]):
                try:
                    return cls._compare_images_gpu(img1, img2)
                except (cv2.error, RuntimeError) as e:
                    logging.warning(f"GPU comparison failed, falling back to CPU: {str(e)}")
                    return cls._compare_images_cpu(img1, img2)
            return cls._compare_images_cpu(img1, img2)
        except Exception as e:
            logging.error(f"Error comparing images: {str(e)}")
            return 0.0

    @classmethod
    def _compare_images_gpu(cls, img1: np.ndarray | str, img2: np.ndarray | str) -> float:
        try:
            # Load images if paths are provided
            if isinstance(img1, str):
                img1 = cls._load_image(img1)
            if isinstance(img2, str):
                img2 = cls._load_image(img2)

            if img1 is None or img2 is None:
                return 0.0

            if cls.has_cuda and cls.can_use_gpu(img1.shape[:2]):
                try:
                    # Upload to GPU
                    gpu_img1 = cv2.cuda_GpuMat()
                    gpu_img2 = cv2.cuda_GpuMat()
                    gpu_img1.upload(img1)
                    gpu_img2.upload(img2)

                    # Convert to grayscale on GPU
                    gray1 = cv2.cuda.cvtColor(gpu_img1, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cuda.cvtColor(gpu_img2, cv2.COLOR_BGR2GRAY)

                    # Download for histogram comparison
                    cpu_gray1 = gray1.download()
                    cpu_gray2 = gray2.download()

                    # Calculate histograms
                    hist_sim = cls._histogram_similarity_cpu(cpu_gray1, cpu_gray2)

                    if hist_sim > 0.5:
                        feature_sim = cls._feature_similarity_cpu(cpu_gray1, cpu_gray2)
                        similarity = 0.6 * hist_sim + 0.4 * feature_sim
                    else:
                        similarity = hist_sim

                    # Cleanup GPU memory
                    gpu_img1.release()
                    gpu_img2.release()
                    gray1.release()
                    gray2.release()

                    return similarity

                except cv2.error as e:
                    logging.error(f"GPU processing error: {str(e)}, falling back to CPU")
                    return cls._compare_images_cpu(img1, img2)
            else:
                return cls._compare_images_cpu(img1, img2)

        except Exception as e:
            logging.error(f"Error comparing images: {str(e)}")
            return 0.0

    @classmethod
    def _compare_images_cpu(cls, img1: np.ndarray, img2: np.ndarray) -> float:
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Calculate histogram similarity
            hist_sim = cls._histogram_similarity_cpu(gray1, gray2)

            if hist_sim > 0.5:
                feature_sim = cls._feature_similarity_cpu(gray1, gray2)
                return 0.6 * hist_sim + 0.4 * feature_sim

            return hist_sim

        except Exception as e:
            logging.error(f"Error in CPU comparison: {str(e)}")
            return 0.0

    @classmethod
    def _feature_similarity_gpu(cls, gray1: cv2.cuda_GpuMat, gray2: cv2.cuda_GpuMat) -> float:
        try:
            # Download for feature detection - ORB is not available on GPU
            cpu_gray1 = gray1.download()
            cpu_gray2 = gray2.download()

            orb = cv2.ORB_create(nfeatures=500)
            kp1, des1 = orb.detectAndCompute(cpu_gray1, None)
            kp2, des2 = orb.detectAndCompute(cpu_gray2, None)

            if not kp1 or not kp2 or des1 is None or des2 is None:
                return 0.0

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)[:50]

            good_matches = sum(1 for m in matches if m.distance < 40)
            max_matches = min(len(kp1), len(kp2), 50)

            return good_matches / max_matches if max_matches > 0 else 0.0

        except cv2.error as e:
            logging.error(f"GPU feature detection error: {str(e)}")
            return 0.0

    @staticmethod
    def _histogram_similarity_gpu(gpu_img1: cv2.cuda_GpuMat, gpu_img2: cv2.cuda_GpuMat) -> float:
        try:
            # Calculate histograms on CPU - more reliable than GPU for histograms
            img1 = gpu_img1.download()
            img2 = gpu_img2.download()

            histSize = [256]
            ranges = [0, 256]

            hist1 = cv2.calcHist([img1], [0], None, histSize, ranges)
            hist2 = cv2.calcHist([img2], [0], None, histSize, ranges)

            # Normalize
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

        except Exception as e:
            logging.warning(f"Histogram comparison failed: {str(e)}")
            return 0.0

    @classmethod
    def _histogram_similarity_cpu(cls, img1: np.ndarray, img2: np.ndarray) -> float:
        try:
            # Calculate histograms for single-channel (grayscale) images
            hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

            # Normalize the histograms
            cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            # Compare histograms
            similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return max(0.0, float(similarity))  # Ensure non-negative result

        except Exception as e:
            logging.error(f"Error calculating histogram similarity: {str(e)}")
            return 0.0

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
# print("Available methods:", [method for method in dir(ImageComparison) if not method.startswith('_')])
ImageComparison.init_gpu()