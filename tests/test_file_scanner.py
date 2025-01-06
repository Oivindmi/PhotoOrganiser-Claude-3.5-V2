import os
import pytest
import tempfile
import cv2
import numpy as np
import psutil
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.utils.file_scanner import FileScanner


class TestFileScanner:
    @pytest.fixture
    def scanner(self):
        return FileScanner()

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield Path(tmp_dir)

    def create_test_file(self, dir_path: Path, filename: str) -> Path:
        file_path = dir_path / filename
        file_path.touch()
        return file_path

    def test_is_media_file(self, scanner):
        assert scanner.is_media_file("test.jpg")
        assert scanner.is_media_file("test.JPG")
        assert scanner.is_media_file("test.mp4")
        assert not scanner.is_media_file("test.txt")
        assert not scanner.is_media_file("test")

    def test_is_video(self, scanner):
        assert scanner.is_video("test.mp4")
        assert scanner.is_video("test.MP4")
        assert scanner.is_video("test.avi")
        assert not scanner.is_video("test.jpg")
        assert not scanner.is_video("test.txt")

    def test_scan_folders(self, scanner, temp_dir):
        # Create test files
        jpg_file = self.create_test_file(temp_dir, "test.jpg")
        mp4_file = self.create_test_file(temp_dir, "video.mp4")
        txt_file = self.create_test_file(temp_dir, "ignore.txt")

        # Create subfolder with files
        sub_dir = temp_dir / "subfolder"
        sub_dir.mkdir()
        sub_jpg = self.create_test_file(sub_dir, "sub.jpg")

        # Scan directory
        all_files, file_info_dict = scanner.scan_folders([str(temp_dir)])

        # Verify results
        expected_files = {str(jpg_file), str(mp4_file), str(sub_jpg)}
        assert set(all_files) == expected_files
        assert len(file_info_dict) == len(expected_files)

    def test_get_file_info(self, scanner, temp_dir):
        # Create test file
        test_file = self.create_test_file(temp_dir, "test.jpg")

        # Get file info twice
        info1 = scanner.get_file_info(str(test_file))
        info2 = scanner.get_file_info(str(test_file))

        # Verify it returns consistent results
        assert info1 == info2
        assert isinstance(info1, str)
        assert len(info1) > 0

    def create_test_video(self, path: Path, duration: int = 2, fps: int = 30, resolution: tuple = (640, 480)) -> Path:
        width, height = resolution
        video_path = str(path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        try:
            for frame_idx in range(duration * fps):
                # Create a frame with a number on it
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(frame, str(frame_idx), (width // 2, height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                out.write(frame)
        finally:
            out.release()

        return Path(video_path)

    def test_extract_video_frames(self, scanner, temp_dir):
        # Test non-video files return empty list
        jpg_file = self.create_test_file(temp_dir, "test.jpg")
        frames = scanner.extract_video_frames(str(jpg_file))
        assert frames == []

        # Test actual video frame extraction
        video_path = temp_dir / "test.mp4"
        test_video = self.create_test_video(video_path)
        frames = scanner.extract_video_frames(str(test_video))

        # Should extract exactly 5 frames (0%, 25%, 50%, 75%, 100%)
        assert len(frames) == 5

        # Verify frames exist and are readable
        for frame_path in frames:
            assert os.path.exists(frame_path)
            img = cv2.imread(frame_path)
            assert img is not None
            assert img.shape[2] == 3  # Should be BGR image

    def test_extract_video_frames_corrupted(self, scanner, temp_dir):
        # Test with corrupted video file
        corrupt_video = temp_dir / "corrupt.mp4"
        with open(corrupt_video, 'wb') as f:
            f.write(b'not a video file')

        frames = scanner.extract_video_frames(str(corrupt_video))
        assert frames == []

    def test_verify_video_frames(self, scanner, temp_dir):
        # Create test video and extract frames
        video_path = temp_dir / "test.mp4"
        test_video = self.create_test_video(video_path)
        frames = scanner.extract_video_frames(str(test_video))

        # Test with all frames existing
        valid_frames = scanner.verify_video_frames(frames)
        assert len(valid_frames) == len(frames)

        # Remove one frame and verify it's detected
        if frames:
            os.remove(frames[0])
            valid_frames = scanner.verify_video_frames(frames)
            assert len(valid_frames) == len(frames) - 1

    def test_clean_old_frames(self, scanner, temp_dir):
        # Create test video and extract frames
        video_path = temp_dir / "test.mp4"
        test_video = self.create_test_video(video_path)
        frames = scanner.extract_video_frames(str(test_video))

        # Verify frames directory exists
        assert os.path.exists(scanner.frames_dir)

        # Clean frames and verify directory still exists
        scanner.clean_old_frames()
        assert os.path.exists(scanner.frames_dir)

    def test_different_resolutions(self, scanner, temp_dir):
        resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        for width, height in resolutions:
            video_path = temp_dir / f"test_{width}x{height}.mp4"
            self.create_test_video(video_path, duration=1, fps=30, resolution=(width, height))
            frames = scanner.extract_video_frames(str(video_path))
            assert len(frames) == 5

            # Verify first frame dimensions
            if frames:
                img = cv2.imread(frames[0])
                assert img is not None

    def test_concurrent_extraction(self, scanner, temp_dir):
        video_files = []
        for i in range(3):
            video_path = temp_dir / f"test_video_{i}.mp4"
            video_files.append(self.create_test_video(video_path, duration=1))

        all_frames = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(scanner.extract_video_frames, str(video))
                       for video in video_files]
            for future in as_completed(futures):
                frames = future.result()
                all_frames.extend(frames)

        assert len(all_frames) == len(video_files) * 5

    def test_large_video_performance(self, scanner, temp_dir):
        # Create a 30-second HD video
        video_path = temp_dir / "large_test.mp4"
        self.create_test_video(video_path, duration=30, fps=30, resolution=(1920, 1080))

        # Measure memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss

        # Measure time
        start_time = time.time()
        frames = scanner.extract_video_frames(str(video_path))
        processing_time = time.time() - start_time

        # Measure memory after
        mem_after = process.memory_info().rss
        mem_increase = mem_after - mem_before

        # Memory increase should be reasonable (less than 1GB)
        assert mem_increase < 1024 * 1024 * 1024

        # Processing time should be reasonable (less than 30 seconds)
        assert processing_time < 30

        # Should still extract exactly 5 frames
        assert len(frames) == 5

        # Verify frame quality
        for frame_path in frames:
            img = cv2.imread(frame_path)
            assert img is not None
            # Verify image has reasonable dimensions
            height, width = img.shape[:2]
            assert 100 <= height <= 1080  # Reasonable height range
            assert 100 <= width <= 1920  # Reasonable width range

    def test_batch_processing_performance(self, scanner, temp_dir):
        # Create multiple videos
        video_paths = []
        for i in range(5):
            video_path = temp_dir / f"test_video_{i}.mp4"
            self.create_test_video(video_path, duration=5, fps=30, resolution=(1280, 720))
            video_paths.append(video_path)

        # Measure batch processing time
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(scanner.extract_video_frames, str(video))
                       for video in video_paths]
            all_frames = []
            for future in as_completed(futures):
                frames = future.result()
                all_frames.extend(frames)

        processing_time = time.time() - start_time

        # Total frames should be number of videos * 5 frames each
        assert len(all_frames) == len(video_paths) * 5

        # Average time per video should be reasonable
        avg_time_per_video = processing_time / len(video_paths)
        assert avg_time_per_video < 10

    def test_memory_cleanup(self, scanner, temp_dir):
        video_path = temp_dir / "cleanup_test.mp4"
        self.create_test_video(video_path, duration=10, fps=30, resolution=(1920, 1080))

        # Initial memory
        process = psutil.Process()
        initial_mem = process.memory_info().rss

        # Process video multiple times
        for _ in range(3):
            frames = scanner.extract_video_frames(str(video_path))
            assert len(frames) == 5

        # Final memory
        final_mem = process.memory_info().rss
        mem_increase = final_mem - initial_mem

        # Memory increase should be minimal after multiple runs
        assert mem_increase < 100 * 1024 * 1024  # Less than 100MB

    def test_mixed_media_performance(self, scanner, temp_dir):
        # Create mix of videos and images
        num_each = 10
        media_files = []

        # Create videos with different resolutions
        resolutions = [(640, 480), (1280, 720), (1920, 1080)]
        for i in range(num_each):
            video_path = temp_dir / f"video_{i}.mp4"
            resolution = resolutions[i % len(resolutions)]
            self.create_test_video(video_path, duration=2, resolution=resolution)
            media_files.append(video_path)

        # Create image files
        for i in range(num_each):
            img_path = temp_dir / f"image_{i}.jpg"
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
            media_files.append(img_path)

        # Test scanning performance
        start_time = time.time()
        all_files, file_info_dict = scanner.scan_folders([str(temp_dir)])
        scan_time = time.time() - start_time

        assert len(all_files) == num_each * 2
        assert scan_time < 5  # Should be quick

        # Test frame extraction for all videos
        total_frames = []
        frame_extraction_times = []

        for file_path in all_files:
            if scanner.is_video(file_path):
                start_time = time.time()
                frames = scanner.extract_video_frames(file_path)
                frame_extraction_times.append(time.time() - start_time)
                total_frames.extend(frames)

        avg_extraction_time = sum(frame_extraction_times) / len(frame_extraction_times)
        assert avg_extraction_time < 5  # Average extraction under 5 seconds

    def test_long_running_performance(self, scanner, temp_dir):
        # Test sustained performance over multiple operations
        iterations = 5
        video_path = temp_dir / "test_video.mp4"
        self.create_test_video(video_path, duration=5, resolution=(1280, 720))

        process = psutil.Process()
        initial_mem = process.memory_info().rss
        extraction_times = []

        for i in range(iterations):
            start_time = time.time()
            frames = scanner.extract_video_frames(str(video_path))
            extraction_times.append(time.time() - start_time)

            assert len(frames) == 5
            # Sleep briefly to simulate real-world usage
            time.sleep(0.1)

        final_mem = process.memory_info().rss
        mem_increase = final_mem - initial_mem

        # Check performance consistency
        time_variance = np.var(extraction_times)
        assert time_variance < 1.0  # Times should be consistent
        assert mem_increase < 200 * 1024 * 1024  # Less than 200MB increase

    def test_parallel_mixed_operations(self, scanner, temp_dir):
        # Test parallel scanning and frame extraction
        num_videos = 3
        video_paths = []

        # Create test videos
        for i in range(num_videos):
            video_path = temp_dir / f"parallel_test_{i}.mp4"
            self.create_test_video(video_path, duration=3, resolution=(1280, 720))
            video_paths.append(video_path)

        start_time = time.time()
        all_frames = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            # Start frame extraction tasks
            frame_futures = [
                executor.submit(scanner.extract_video_frames, str(path))
                for path in video_paths
            ]

            # While frames are being extracted, do file scanning
            scan_future = executor.submit(scanner.scan_folders, [str(temp_dir)])

            # Collect frame results
            for future in as_completed(frame_futures):
                frames = future.result()
                all_frames.extend(frames)

            # Get scan results
            all_files, file_info_dict = scan_future.result()

        total_time = time.time() - start_time

        # Verify results
        assert len(all_frames) == num_videos * 5  # 5 frames per video
        assert len(all_files) == num_videos
        assert total_time < 30  # Should complete in reasonable time


if __name__ == "__main__":
    pytest.main(["-v", __file__])