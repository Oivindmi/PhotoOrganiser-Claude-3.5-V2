import cv2
import numpy as np
import os
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import pickle
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class ImageComparison:
    _cache = {}
    TARGET_SIZE = (200, 200)
    _has_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0

    @staticmethod
    def has_cuda() -> bool:
        return ImageComparison._has_cuda

    @staticmethod
    def init_gpu():
        if ImageComparison._has_cuda:
            cv2.cuda.setDevice(0)
            logger.info("CUDA device initialized")
            logger.info(f"Using GPU: {cv2.cuda.getDevice()}")
        else:
            logger.warning("No CUDA device available, falling back to CPU")

    @staticmethod
    def compare_image_data(img1: np.ndarray, img2: np.ndarray) -> float:
        if img1 is None or img2 is None:
            return 0

        img1 = cv2.resize(img1, ImageComparison.TARGET_SIZE)
        img2 = cv2.resize(img2, ImageComparison.TARGET_SIZE)

        if ImageComparison._has_cuda:
            return ImageComparison._compare_image_data_gpu(img1, img2)
        else:
            return ImageComparison._compare_image_data_cpu(img1, img2)

    @staticmethod
    def _compare_image_data_gpu(img1: np.ndarray, img2: np.ndarray) -> float:
        try:
            # Convert images to GPU
            gpu_img1 = cv2.cuda_GpuMat()
            gpu_img2 = cv2.cuda_GpuMat()
            gpu_img1.upload(img1)
            gpu_img2.upload(img2)

            # Convert to grayscale on GPU
            gpu_gray1 = cv2.cuda.cvtColor(gpu_img1, cv2.COLOR_BGR2GRAY)
            gpu_gray2 = cv2.cuda.cvtColor(gpu_img2, cv2.COLOR_BGR2GRAY)

            # Calculate histograms on GPU
            hist_sim = ImageComparison._histogram_similarity_gpu(gpu_img1, gpu_img2)

            if hist_sim > 0.5:
                # Download for CPU feature matching if needed
                gray1 = gpu_gray1.download()
                gray2 = gpu_gray2.download()
                feature_sim = ImageComparison._feature_similarity_cpu(gray1, gray2)
                return 0.6 * hist_sim + 0.4 * feature_sim

            return hist_sim

        except cv2.error as e:
            logger.error(f"GPU processing error: {str(e)}")
            return ImageComparison._compare_image_data_cpu(img1, img2)

    @staticmethod
    def _histogram_similarity_gpu(gpu_img1: cv2.cuda_GpuMat, gpu_img2: cv2.cuda_GpuMat) -> float:
        try:
            hist1 = cv2.cuda.calcHist(gpu_img1)
            hist2 = cv2.cuda.calcHist(gpu_img2)

            # Download histograms for comparison
            cpu_hist1 = hist1.download()
            cpu_hist2 = hist2.download()

            return cv2.compareHist(cpu_hist1, cpu_hist2, cv2.HISTCMP_CORREL)
        except cv2.error:
            # Fallback to CPU if GPU histogram fails
            img1 = gpu_img1.download()
            img2 = gpu_img2.download()
            return ImageComparison._histogram_similarity_cpu(img1, img2)

    @staticmethod
    def _compare_image_data_cpu(img1: np.ndarray, img2: np.ndarray) -> float:
        hist_sim = ImageComparison._histogram_similarity_cpu(img1, img2)

        if hist_sim > 0.5:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            feature_sim = ImageComparison._feature_similarity_cpu(gray1, gray2)
            return 0.6 * hist_sim + 0.4 * feature_sim

        return hist_sim

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
            return 0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:50]

        num_good_matches = sum(1 for m in matches if m.distance < 40)
        max_possible_matches = min(len(kp1), len(kp2), 50)
        return num_good_matches / max_possible_matches if max_possible_matches > 0 else 0