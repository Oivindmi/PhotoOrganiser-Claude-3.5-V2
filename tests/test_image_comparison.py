import pytest
import cv2
import numpy as np
import tempfile
import psutil
import gc
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from app.utils.image_comparison import ImageComparison


class TestImageComparison:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def create_test_image(self, path: Path, color: tuple = (255, 255, 255)) -> Path:
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:] = color
        cv2.imwrite(str(path), img)
        return path

    def test_compare_identical_images(self, temp_dir):
        img1_path = temp_dir / "img1.jpg"
        img2_path = temp_dir / "img2.jpg"
        self.create_test_image(img1_path)
        self.create_test_image(img2_path)

        similarity = ImageComparison.compare_images(str(img1_path), str(img2_path))
        assert similarity > 0.5  # Should be reasonably similar

    def test_compare_different_images(self, temp_dir):
        img1_path = temp_dir / "img1.jpg"
        img2_path = temp_dir / "img2.jpg"
        self.create_test_image(img1_path, color=(255, 255, 255))  # White
        self.create_test_image(img2_path, color=(0, 0, 0))  # Black

        similarity = ImageComparison.compare_images(str(img1_path), str(img2_path))
        assert similarity < 0.1  # Should be very different

    def test_batch_comparison(self, temp_dir):
        img_paths = []
        colors = [(255, 255, 255), (0, 0, 0), (128, 128, 128)]

        for i, color in enumerate(colors):
            path = temp_dir / f"img{i}.jpg"
            self.create_test_image(path, color)
            img_paths.append(path)

        pairs = [(str(img_paths[0]), str(img_paths[1])),
                 (str(img_paths[1]), str(img_paths[2])),
                 (str(img_paths[0]), str(img_paths[2]))]

        similarities = ImageComparison.batch_compare_media(pairs)
        assert len(similarities) == len(pairs)
        assert all(0 <= sim <= 1 for sim in similarities)

    def test_invalid_images(self, temp_dir):
        valid_img = temp_dir / "valid.jpg"
        self.create_test_image(valid_img)

        nonexistent = temp_dir / "nonexistent.jpg"
        similarity = ImageComparison.compare_images(str(valid_img), str(nonexistent))
        assert similarity == 0.0

        corrupt_img = temp_dir / "corrupt.jpg"
        with open(corrupt_img, 'wb') as f:
            f.write(b'not an image')

        similarity = ImageComparison.compare_images(str(valid_img), str(corrupt_img))
        assert similarity == 0.0

    def test_gpu_acceleration(self, temp_dir):
        assert hasattr(ImageComparison, 'has_cuda')

        img1_path = temp_dir / "img1.jpg"
        img2_path = temp_dir / "img2.jpg"
        self.create_test_image(img1_path)
        self.create_test_image(img2_path)

        if ImageComparison.has_cuda:
            similarity = ImageComparison._compare_images_gpu(
                ImageComparison._load_image(str(img1_path)),
                ImageComparison._load_image(str(img2_path))
            )
            assert similarity > 0.5  # Should be reasonably similar

    def test_feature_matching(self, temp_dir):
        img1 = np.zeros((200, 200, 3), dtype=np.uint8)
        img2 = np.zeros((200, 200, 3), dtype=np.uint8)

        cv2.circle(img1, (100, 100), 30, (255, 255, 255), -1)
        cv2.circle(img2, (110, 110), 30, (255, 255, 255), -1)  # Slightly offset

        img1_path = temp_dir / "feat1.jpg"
        img2_path = temp_dir / "feat2.jpg"
        cv2.imwrite(str(img1_path), img1)
        cv2.imwrite(str(img2_path), img2)

        similarity = ImageComparison.compare_images(str(img1_path), str(img2_path))
        assert similarity > 0.5  # Should detect similar features

    def test_histogram_comparison(self, temp_dir):
        img1 = np.zeros((200, 200, 3), dtype=np.uint8)
        img2 = np.zeros((200, 200, 3), dtype=np.uint8)

        for i in range(200):
            img1[i, :] = [i, i, i]
            img2[i, :] = [i, i, i]

        img1_path = temp_dir / "grad1.jpg"
        img2_path = temp_dir / "grad2.jpg"
        cv2.imwrite(str(img1_path), img1)
        cv2.imwrite(str(img2_path), img2)

        similarity = ImageComparison._histogram_similarity_cpu(
            cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        )
        assert similarity > 0.95  # Histogram should be nearly identical

    def test_memory_management(self, temp_dir):
        process = psutil.Process()
        initial_memory = process.memory_info().rss

        img_paths = []
        for i in range(20):
            path = temp_dir / f"img_{i}.jpg"
            img = np.zeros((1000, 1000, 3), dtype=np.uint8)
            img[:] = (i * 10, i * 10, i * 10)
            cv2.imwrite(str(path), img)
            img_paths.append(path)

        for _ in range(3):
            for path in img_paths:
                img = ImageComparison._load_image(str(path))
                assert img is not None
                del img
                gc.collect()

        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory

        assert memory_increase < 100 * 1024 * 1024
        assert len(ImageComparison._image_cache.cache) <= 100

    def test_cache_eviction(self, temp_dir):
        cache_size = ImageComparison._image_cache.capacity
        img_paths = []

        for i in range(cache_size + 10):
            path = temp_dir / f"cache_test_{i}.jpg"
            self.create_test_image(path, color=(i * 10, i * 10, i * 10))
            img_paths.append(path)

        loaded_images = []
        for path in img_paths:
            img = ImageComparison._load_cached_image(str(path))
            loaded_images.append(img)

        assert len(ImageComparison._image_cache.cache) <= cache_size

        first_path = str(img_paths[0])
        first_image = ImageComparison._load_cached_image(first_path)

        assert np.array_equal(first_image, loaded_images[0])

    def test_cache_performance(self, temp_dir):
        test_path = temp_dir / "perf_test.jpg"
        self.create_test_image(test_path)

        start_time = time.time()
        first_load = ImageComparison._load_cached_image(str(test_path))
        first_load_time = time.time() - start_time

        start_time = time.time()
        second_load = ImageComparison._load_cached_image(str(test_path))
        second_load_time = time.time() - start_time

        assert second_load_time < first_load_time
        assert first_load is second_load

    def test_cache_thread_safety(self, temp_dir):
        img_paths = []
        for i in range(10):
            path = temp_dir / f"thread_test_{i}.jpg"
            self.create_test_image(path)
            img_paths.append(str(path))

        results = []
        lock = Lock()

        def load_and_verify(path):
            img = ImageComparison._load_cached_image(path)
            with lock:
                results.append(img is not None)
            return img

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for _ in range(3):
                for path in img_paths:
                    futures.append(executor.submit(load_and_verify, path))

            for future in futures:
                future.result()

        assert all(results)
        assert len(ImageComparison._image_cache.cache) <= ImageComparison._image_cache.capacity

    def test_large_image_handling(self, temp_dir):
        large_img = np.zeros((2000, 2000, 3), dtype=np.uint8)
        img_path = temp_dir / "large.jpg"
        cv2.imwrite(str(img_path), large_img)

        loaded_img = ImageComparison._load_image(str(img_path))
        assert loaded_img is not None
        assert loaded_img.shape == ImageComparison.TARGET_SIZE + (3,)

    def test_image_cache_behavior(self, temp_dir):
        img_paths = []
        for i in range(5):
            path = temp_dir / f"cache_test_{i}.jpg"
            self.create_test_image(path)
            img_paths.append(path)

        first_loads = [ImageComparison._load_cached_image(str(path))
                       for path in img_paths]
        assert all(img is not None for img in first_loads)

        second_loads = [ImageComparison._load_cached_image(str(path))
                        for path in img_paths]
        assert all(first is second for first, second in zip(first_loads, second_loads))

    def test_error_handling(self, temp_dir):
        # Create a tiny valid image
        tiny_img = np.zeros((1, 1, 3), dtype=np.uint8)
        tiny_path = temp_dir / "tiny.jpg"
        cv2.imwrite(str(tiny_path), tiny_img)

        corrupt_path = temp_dir / "corrupt.jpg"
        with open(corrupt_path, "wb") as f:
            f.write(b"not an image")

        loaded_tiny = ImageComparison._load_image(str(tiny_path))
        loaded_corrupt = ImageComparison._load_image(str(corrupt_path))
        assert loaded_tiny is not None
        assert loaded_corrupt is None

    def test_concurrent_processing(self, temp_dir):
        img_paths = []
        for i in range(10):
            path = temp_dir / f"concurrent_{i}.jpg"
            self.create_test_image(path, color=(i * 25, i * 25, i * 25))
            img_paths.append(path)

        pairs = [(str(img_paths[i]), str(img_paths[i + 1]))
                 for i in range(len(img_paths) - 1)]

        results = ImageComparison.batch_compare_media(pairs)
        assert len(results) == len(pairs)
        assert all(0 <= r <= 1 for r in results)


if __name__ == "__main__":
    pytest.main(["-v", __file__])