from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree


class MediaMatcher:
    def __init__(self):
        self.media_items = []
        self.kdtree = None

    def add_media_item(self, media_item):
        self.media_items.append(media_item)

    def build_index(self):
        hash_vectors = [list(item.perceptual_hash.hash.flatten()) for item in self.media_items]
        self.kdtree = cKDTree(hash_vectors)

    def find_matches(self, query_item, max_distance=10):
        query_vector = list(query_item.perceptual_hash.hash.flatten())
        distances, indices = self.kdtree.query(query_vector, k=10, distance_upper_bound=max_distance)

        matches = []
        for dist, idx in zip(distances, indices):
            if idx < len(self.media_items):  # Check if the index is valid
                matches.append((self.media_items[idx], dist))

        return matches

    def group_by_time_window(self, window_size_minutes=10):
        time_windows = defaultdict(list)
        for item in self.media_items:
            window = item.creation_time.replace(
                minute=item.creation_time.minute - item.creation_time.minute % window_size_minutes,
                second=0, microsecond=0)
            time_windows[window].append(item)
        return time_windows

    def compare_keyframes(self, item1, item2):
        if not (item1.is_video and item2.is_video):
            return 0

        # Compare keyframes using perceptual hash
        similarities = []
        for kf1 in item1.keyframes:
            for kf2 in item2.keyframes:
                hash1 = imagehash.phash(Image.fromarray(cv2.cvtColor(kf1, cv2.COLOR_BGR2RGB)))
                hash2 = imagehash.phash(Image.fromarray(cv2.cvtColor(kf2, cv2.COLOR_BGR2RGB)))
                similarities.append(1 - (hash1 - hash2) / len(hash1.hash) ** 2)

        return max(similarities) if similarities else 0


    def compare_audio_fingerprints(self, item1, item2):
        if not (item1.is_video and item2.is_video):
            return 0

        # Placeholder for audio fingerprint comparison
        # This should be replaced with actual audio fingerprint comparison logic
        return 1 if item1.audio_fingerprint == item2.audio_fingerprint else 0