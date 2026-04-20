import numpy as np
import cv2
import os
import mediapipe as mp
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle
from typing import Literal
from time import perf_counter
import torch
from torch.utils.data import Dataset, DataLoader

class PoseDataset(Dataset):
    def __init__(self, X, y_pose, num_poses):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y_pose = torch.from_numpy(y_pose).long()
        self.num_poses = num_poses
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_pose[idx]

class Timer:
    def __init__(self, func):
        self.func = func
        self._exec_times = {}  

    @property
    def exec_time(self):
        return None

    def __call__(self, *args, **kwargs):
        s = perf_counter()
        result = self.func(*args, **kwargs)
        e = perf_counter()
        self._exec_times[id(self)] = e - s
        return result

    def __get__(self, obj, objtype=None):
        """Support instance methods by implementing descriptor protocol"""
        if obj is None:
            return self
        import types
        bound = types.MethodType(self, obj)
        # Attach a per-instance exec_time property via a small wrapper
        wrapper = _BoundTimer(self, obj)
        return wrapper


class _BoundTimer:
    """Wrapper returned by Timer.__get__ that tracks exec_time per owner instance."""
    def __init__(self, timer: 'Timer', obj):
        self._timer = timer
        self._obj = obj
        self._key = id(obj)

    def __call__(self, *args, **kwargs):
        s = perf_counter()
        result = self._timer.func(self._obj, *args, **kwargs)
        e = perf_counter()
        self._timer._exec_times[self._key] = e - s
        return result

    @property
    def exec_time(self):
        return self._timer._exec_times.get(self._key)

# MediaPipe keypoint confidence threshold — landmarks below this
# visibility score are considered unreliable and trigger rejection logic.
KEYPOINT_CONFIDENCE_THRESHOLD = 0.5

# Minimum fraction of keypoints that must be above the threshold for a
# frame to be considered "valid".  If fewer than this fraction are visible
# the frame is marked as rejected and replaced by the nearest valid frame.
MIN_VALID_KEYPOINT_FRACTION = 0.5   # i.e. at least 17 / 33 keypoints visible


class VideoDataLoader:
    def __init__(self, dataset_path, sequence_length=16,
                 mediapipe_model_complexity: int = 1,
                 pool_frames=True, output_dir=None):
        """
        Args:
            dataset_path: Root directory of the dataset.
            sequence_length: Number of frames to sample per video.
            mediapipe_model_complexity: MediaPipe Pose model complexity
                (0 = lite, 1 = full, 2 = heavy).
            pool_frames: Whether to temporally average keypoints after extraction.
            output_dir: Directory for checkpoints / processed data.
        """
        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        self.mediapipe_model_complexity = mediapipe_model_complexity
        self.pool_frames = pool_frames
        self.output_dir = output_dir
        self.train_size = None

        # Storage for data
        self.videos = []
        self.pose_labels = []
        self.pose_names = []

        self._load_dataset_info()
    

    def _load_dataset_info(self):
        """Load dataset structure and file paths.
        
        If output_dir already contains a partial/full run, previously processed
        video paths are loaded from metadata so load_all_videos can skip them.
        """
        # Restore the set of already-processed video paths (empty by default).
        # 'processed_poses' always stores full video paths for fine-grained
        # checkpointing; we derive completed *pose folder names* from those paths
        # for backward-compatible coarse skipping (see below).
        self.processed_poses: set[str] = set()

        metadata_path = (
            os.path.join(self.output_dir, 'metadata.pkl')
            if self.output_dir else None
        )
        if (
            metadata_path
            and os.path.isfile(metadata_path)
        ):
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                self.processed_poses = set(metadata.get('processed_poses', []))
                if self.processed_poses:
                    print(
                        f"Resuming: {len(self.processed_poses)} video(s) already "
                        f"processed and will be skipped."
                    )
            except (pickle.UnpicklingError, EOFError, KeyError) as e:
                print(f"Warning: could not read existing metadata ({e}). Starting fresh.")
                self.processed_poses = set()

        # Derive completed pose *folder names* from the stored video paths.
        # A pose folder is considered complete when every video file inside it
        # appears in processed_poses.  This comparison is purely by folder name
        # so it works with both old checkpoints (that stored folder names) and
        # new ones (that store video paths).
        all_pose_folders = sorted([f for f in os.listdir(self.dataset_path)
                                   if os.path.isdir(os.path.join(self.dataset_path, f))])

        # Build a helper: pose_name -> set of all its video paths on disk
        pose_to_videos: dict[str, set[str]] = {}
        for pose_name in all_pose_folders:
            pose_path = os.path.join(self.dataset_path, pose_name)
            pose_to_videos[pose_name] = {
                os.path.join(pose_path, f)
                for f in os.listdir(pose_path)
                if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))
            }

        # A pose is fully done when ALL its videos are in processed_poses.
        completed_pose_names: set[str] = {
            name
            for name, vids in pose_to_videos.items()
            if vids and vids.issubset(self.processed_poses)
        }

        # Only queue poses that are NOT fully completed yet.
        pose_folders = [f for f in all_pose_folders if f not in completed_pose_names]

        self.pose_names = all_pose_folders   # full list preserved for label consistency
        self.num_poses = len(all_pose_folders)

        for pose_idx, pose_name in enumerate(all_pose_folders):
            if pose_name in completed_pose_names:
                continue                     # every video already in checkpoint
            pose_path = os.path.join(self.dataset_path, pose_name)
            video_files = [f for f in os.listdir(pose_path)
                           if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            for video_file in video_files:
                video_path = os.path.join(pose_path, video_file)
                self.videos.append(video_path)
                self.pose_labels.append(pose_idx)

        pending = len(self.videos)
        print(f"Found {len(all_pose_folders)} pose(s) total, "
              f"{len(completed_pose_names)} fully processed, "
              f"{len(pose_folders)} pending ({pending} video(s) to process)")
        print(f"Poses ({self.num_poses}): {self.pose_names}")
    
    @Timer
    def load_video_frames(self, video_path):
        """Load and preprocess video frames for MediaPipe Pose.

        Frames are decoded as uint8 RGB — MediaPipe handles its own
        internal resizing so we keep frames at native resolution.
        """
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # MediaPipe expects RGB uint8
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.uint8)
            frames.append(frame)

        cap.release()

        if not frames:
            raise ValueError(f"No frames decoded from {video_path}")

        # Sample / pad to self.sequence_length
        if len(frames) >= self.sequence_length:
            indices = np.linspace(0, len(frames) - 1, self.sequence_length).astype(int)
            frames = [frames[i] for i in indices]
        else:
            while len(frames) < self.sequence_length:
                frames.append(frames[-1])

        return frames   # list of H×W×3 uint8 arrays

    @staticmethod
    def _load_model(mediapipe_model_complexity: int = 1):
        """Instantiate a MediaPipe Pose estimator.

        Returns the mp.solutions.pose.Pose object; the caller is
        responsible for using it as a context manager or closing it.
        """
        pose = mp.solutions.pose.Pose(
            static_image_mode=False,          # optimised for video streams
            model_complexity=mediapipe_model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return pose

    @Timer
    def extract_keypoints(self, frames) -> np.ndarray:
        """Extract 33-keypoint MediaPipe Pose landmarks from a list of frames.

        Confidence / rejection logic
        ----------------------------
        Each landmark carries a *visibility* score in [0, 1].  A frame is
        deemed **valid** when at least ``MIN_VALID_KEYPOINT_FRACTION`` of
        its 33 landmarks have visibility ≥ ``KEYPOINT_CONFIDENCE_THRESHOLD``.

        Rejected frames are **imputed** by copying keypoints from the
        nearest valid frame found by scanning forward then backward.
        If *every* frame in the sequence is rejected the raw (low-confidence)
        keypoints are kept and a warning is printed.

        Returns
        -------
        np.ndarray of shape (3, T, 33)
            Axis 0 — channels: [x_norm, y_norm, visibility]
            Axis 1 — time steps
            Axis 2 — keypoints (33 MediaPipe landmarks)
        """
        pose_estimator = self._load_model(self.mediapipe_model_complexity)

        NUM_KP = 33
        T = len(frames)
        raw_kp   = np.zeros((T, NUM_KP, 3), dtype=np.float32)  # (T, 33, 3)
        valid    = np.zeros(T, dtype=bool)

        try:
            for t, frame in enumerate(frames):
                result = pose_estimator.process(frame)
                if result.pose_landmarks:
                    kp = np.array(
                        [[lm.x, lm.y, lm.visibility]
                         for lm in result.pose_landmarks.landmark],
                        dtype=np.float32
                    )  # (33, 3)
                    # Apply per-keypoint threshold
                    low_conf = kp[:, 2] < KEYPOINT_CONFIDENCE_THRESHOLD
                    kp[low_conf, :2] = 0.0   # zero out unreliable positions
                    visible_fraction = (~low_conf).mean()
                    raw_kp[t] = kp
                    valid[t] = visible_fraction >= MIN_VALID_KEYPOINT_FRACTION
                else:
                    # No pose detected — frame stays zeroed, marked invalid
                    valid[t] = False
        finally:
            pose_estimator.close()

        # ── Rejection / imputation logic ────────────────────────────────── #
        num_rejected = int((~valid).sum())
        if num_rejected > 0:
            print(
                f"  [MediaPipe] {num_rejected}/{T} frame(s) rejected "
                f"(visibility < {KEYPOINT_CONFIDENCE_THRESHOLD:.0%}). "
                "Imputing from nearest valid frame."
            )

        video_keypoints = raw_kp.copy()  # will be filled in-place

        if valid.all():
            pass  # nothing to impute
        elif not valid.any():
            # Edge case: every frame is below threshold — keep raw data
            print(
                "  [MediaPipe] WARNING: no valid frames found for this video. "
                "Using raw (low-confidence) keypoints."
            )
        else:
            # Build a lookup: for each frame index find the nearest valid idx
            valid_indices = np.where(valid)[0]
            for t in range(T):
                if not valid[t]:
                    # Find nearest valid frame (prefer forward, then backward)
                    diffs = np.abs(valid_indices - t)
                    nearest = valid_indices[np.argmin(diffs)]
                    video_keypoints[t] = raw_kp[nearest]

        # Transpose to (channels=3, time=T, joints=33)
        video_keypoints = video_keypoints.transpose(2, 0, 1)  # (3, T, 33)
        return video_keypoints
    
    def load_all_videos(self, max_videos=None):
        """Load all videos into numpy arrays, skipping already-processed ones.

        Previously processed keypoints are loaded from output_dir and merged
        with any newly extracted ones so the final arrays are always complete.
        """
        if max_videos:
            video_paths = self.videos[:max_videos]
            pose_labels = self.pose_labels[:max_videos]
        else:
            video_paths = self.videos
            pose_labels = self.pose_labels

        # ------------------------------------------------------------------ #
        # Load cached keypoints that were saved in a previous (partial) run.  #
        # ------------------------------------------------------------------ #
        cache_X: list = []
        cache_y: list = []

        cache_path = (
            os.path.join(self.output_dir, 'partial_X.npy')
            if self.output_dir else None
        )
        cache_y_path = (
            os.path.join(self.output_dir, 'partial_y.npy')
            if self.output_dir else None
        )
        if (
            cache_path and os.path.isfile(cache_path)
            and cache_y_path and os.path.isfile(cache_y_path)
            and self.processed_poses
        ):
            try:
                cache_X = np.load(cache_path, allow_pickle=False)
                cache_y = np.load(cache_y_path, allow_pickle=False)
                print(f"Loaded {cache_X.shape[0]} cached keypoint(s) from previous run.")
            except Exception as e:
                print(f"Warning: could not load cached keypoints ({e}). Re-processing all videos.")
                cache_X, cache_y = [], []
                self.processed_poses = set()

        # ------------------------------------------------------------------ #
        # Process only videos that haven't been handled yet.                  #
        # ------------------------------------------------------------------ #
        pending_paths  = [p for p in video_paths if p not in self.processed_poses]
        pending_labels = [
            pose_labels[i]
            for i, p in enumerate(video_paths)
            if p not in self.processed_poses
        ]

        X_new: list = []
        y_new: list = []

        etkp_time = 0.0
        lf_time = 0.0

        print(
            f"Loading {len(pending_paths)} new video(s) "
            f"({len(self.processed_poses)} poses skipped — already processed)..."
        )

        for i, video_path in enumerate(pending_paths):
            if i % 10 == 0:
                print(f"Loading video {i+1}/{len(pending_paths)}")
                print(f"Extract Keypoints took {etkp_time:.3f}s for last 10 videos")
                print(f"Loading Frames   took {lf_time:.3f}s for last 10 videos")
                etkp_time = 0.0
                lf_time = 0.0

            try:
                frames = self.load_video_frames(video_path)
                keypoints = self.extract_keypoints(frames)
                etkp_time += self.extract_keypoints.exec_time
                lf_time += self.load_video_frames.exec_time
                X_new.append(keypoints)
                y_new.append(pending_labels[i])
                self.processed_poses.add(video_path)
            except Exception as e:
                print(f"Error loading {video_path}: {e}")
                continue

            # ---------------------------------------------------------------- #
            # Incremental checkpoint every 10 videos so a crash only loses     #
            # work done since the last checkpoint.                              #
            # ---------------------------------------------------------------- #
            if self.output_dir and (i + 1) % 10 == 0:
                self._save_partial_progress(
                    self._safe_concat_X(cache_X, X_new),
                    self._safe_concat_y(cache_y, y_new),
                )

        # Final merge of cached + newly extracted data
        if not X_new and not (isinstance(cache_X, np.ndarray) and cache_X.ndim == 4):
            raise RuntimeError('No keypoints loaded — check dataset path and video files.')
        X      = self._safe_concat_X(cache_X, X_new)
        y_pose = self._safe_concat_y(cache_y, y_new)

        # Persist the completed (or updated) partial cache
        if self.output_dir and X_new:
            self._save_partial_progress(X, y_pose)

        if self.pool_frames:
            X = X.mean(axis=2)
            print(f"Frame-pooled data shape: {X.shape}")

        print(f"Loaded {len(X)} videos successfully")
        print(f"Video data shape:  {X.shape}")
        print(f"Pose labels shape: {y_pose.shape}")

        return X, y_pose

    # ------------------------------------------------------------------ #
    # Safe concatenation helpers                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _safe_concat_X(cache_X, new_list: list) -> np.ndarray:
        """Concatenate a cached 4-D ndarray with a list of new arrays.

        Handles the three degenerate cases that previously caused the
        ``ValueError: all input arrays must have same number of dimensions``
        crash:
          - cache empty (list), new_list has items
          - cache is a valid ndarray, new_list is empty
          - both have items
        """
        parts = []
        if isinstance(cache_X, np.ndarray) and cache_X.ndim == 4:
            parts.append(cache_X)
        if new_list:
            parts.append(np.array(new_list))   # uniform shape guaranteed by extract_keypoints
        if not parts:
            raise RuntimeError("No keypoint data available to concatenate.")
        return np.concatenate(parts, axis=0)

    @staticmethod
    def _safe_concat_y(cache_y, new_list: list) -> np.ndarray:
        parts = []
        if isinstance(cache_y, np.ndarray) and cache_y.ndim >= 1:
            parts.append(cache_y)
        if new_list:
            parts.append(np.array(new_list))
        if not parts:
            raise RuntimeError("No label data available to concatenate.")
        return np.concatenate(parts, axis=0)

    def load_dataset(self, max_videos=None):
        """Return (X, y_pose) — from disk cache if complete, otherwise run pipeline.

        Strategy
        --------
        1. If ``output_dir`` contains a finalised dataset (``full_X.npy`` /
           ``full_y.npy`` **or** both ``train_X.npy`` + ``val_X.npy``) load and
           return it directly, skipping the MoveNet pipeline entirely.
        2. If ``output_dir`` contains a *partial* checkpoint (``partial_X.npy``),
           ``load_all_videos`` will pick it up automatically and only process the
           remaining videos.
        3. If nothing is cached, run ``load_all_videos`` from scratch.

        This is the recommended entry-point for callers that previously called
        ``load_all_videos`` directly.
        """
        # ── 1. Complete finalised dataset ───────────────────────────────── #
        if self.output_dir:
            full_X_path = os.path.join(self.output_dir, 'full_X.npy')
            full_y_path = os.path.join(self.output_dir, 'full_y.npy')
            train_X_path = os.path.join(self.output_dir, 'train_X.npy')
            val_X_path   = os.path.join(self.output_dir, 'val_X.npy')

            if os.path.isfile(full_X_path) and os.path.isfile(full_y_path):
                print(f"[load_dataset] Found complete dataset at '{self.output_dir}'. Loading from disk.")
                X      = np.load(full_X_path, allow_pickle=False)
                y_pose = np.load(full_y_path, allow_pickle=False)
                print(f"[load_dataset] Loaded {len(X)} samples  shape={X.shape}")
                return X, y_pose

            if (os.path.isfile(train_X_path) and os.path.isfile(val_X_path)):
                print(f"[load_dataset] Found split dataset at '{self.output_dir}'. Merging splits.")
                X_train = np.load(train_X_path, allow_pickle=False)
                y_train = np.load(os.path.join(self.output_dir, 'train_y.npy'), allow_pickle=False)
                X_val   = np.load(val_X_path, allow_pickle=False)
                y_val   = np.load(os.path.join(self.output_dir, 'val_y.npy'), allow_pickle=False)
                X      = np.concatenate([X_train, X_val], axis=0)
                y_pose = np.concatenate([y_train, y_val], axis=0)
                print(f"[load_dataset] Merged {len(X)} samples  shape={X.shape}")
                return X, y_pose

        # ── 2 & 3. Partial checkpoint or cold start ──────────────────────── #
        print("[load_dataset] No complete cache found — running extraction pipeline.")
        return self.load_all_videos(max_videos=max_videos)

    def _save_partial_progress(self, X: np.ndarray, y: np.ndarray):
        """Checkpoint keypoints and the processed-video set to output_dir."""
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        np.save(os.path.join(self.output_dir, 'partial_X.npy'), np.array(X))
        np.save(os.path.join(self.output_dir, 'partial_y.npy'), np.array(y))
        # Update only the 'processed_poses' key so we don't clobber any
        # finalized split data that save_processed_data may have written.
        metadata_path = os.path.join(self.output_dir, 'metadata.pkl')
        existing: dict = {}
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path, 'rb') as f:
                    existing = pickle.load(f)
            except Exception:
                pass
        existing['processed_poses'] = list(self.processed_poses)
        with open(metadata_path, 'wb') as f:
            pickle.dump(existing, f)
        
    def create_balanced_split(self, X, y_pose, train_size=0.8, random_state=None):
        """Create balanced train/val splits with fallback for small datasets"""
        val_size = 1.0 - train_size
        self.train_size = train_size
       
        label_counts = Counter(y_pose)
        print("\nLabel distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count} videos")
        
        num_classes = len(label_counts)
        min_samples_per_class = min(label_counts.values())
        min_val_samples = max(1, int(len(X) * val_size))
        
        # Try stratified split first
        try:
            if min_samples_per_class >= 2 and len(X) >= num_classes * 2:
                X_train, X_val, y_pose_train, y_pose_val = train_test_split(
                    X, y_pose, 
                    train_size=train_size,
                    stratify=y_pose,
                    random_state=random_state
                )
            else:
                raise ValueError("Too few samples for stratified split")
                
        except ValueError as e:
            print(f"Stratified split failed: {e}")
            print("Falling back to random split...")
            X_train, X_val, y_pose_train, y_pose_val = train_test_split(
                X, y_pose,
                train_size=train_size,
                random_state=random_state
            )
        
        return {
            'train': (X_train, y_pose_train),
            'val': (X_val, y_pose_val)
        }
    
    def create_tensorflow_dataset(self, X, y_pose, batch_size=4, shuffle=True, prefetch=True):
        """Convert numpy arrays to TensorFlow Dataset"""
        y_pose_onehot = tf.keras.utils.to_categorical(y_pose, num_classes=self.num_poses)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            X.astype(np.float32),
            y_pose_onehot.astype(np.float32),
        ))
        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        
        dataset = dataset.batch(batch_size)
        
        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def create_pytorch_dataloader(self, X, y_pose, batch_size=4, shuffle=True, num_workers=2):
        """Convert numpy arrays to PyTorch DataLoader"""
        dataset = PoseDataset(X, y_pose, self.num_poses)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers, 
            pin_memory=True,  
            persistent_workers=True if num_workers > 0 else False 
        )
        
        return dataloader
    
    def save_processed_data(self, data_splits, save_path, batch_size):
        """Save processed data to disk"""
        os.makedirs(save_path, exist_ok=True)

        metadata = {
            'num_poses': self.num_poses,
            'pose_names': list(self.pose_names),
            'sequence_length': self.sequence_length,
            'mediapipe_model_complexity': self.mediapipe_model_complexity,
            'batch_size': batch_size,
            'pool_frames': self.pool_frames,
            'train_size': self.train_size,
        }

        with open(os.path.join(save_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

        # Save data splits
        for split_name, (X, y) in data_splits.items():
            if X is not None:
                np.save(os.path.join(save_path, f'{split_name}_X.npy'), X)
                np.save(os.path.join(save_path, f'{split_name}_y.npy'), y)

        print(f"Data saved to {save_path}")
    
    def load_processed_data(self, save_path):
        """Load processed data from disk"""
        with open(os.path.join(save_path, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        self.num_poses = metadata['num_poses']
        self.pose_names = metadata['pose_names']
        self.sequence_length = metadata['sequence_length']
        # Support both old (movenet_variant) and new (mediapipe_model_complexity) keys
        self.mediapipe_model_complexity = metadata.get(
            'mediapipe_model_complexity',
            metadata.get('movenet_variant', 1)   # graceful fallback
        )

        data_splits = {}
        possible_splits = ['train', 'val', 'full']

        for split_name in possible_splits:
            X_path = os.path.join(save_path, f'{split_name}_X.npy')
            if os.path.exists(X_path):
                X = np.load(X_path)
                y = np.load(os.path.join(save_path, f'{split_name}_y.npy'))
                data_splits[split_name] = (X, y)

        return data_splits


def verify_metadata(saved_path, new_metadata: dict):
    """Return True if the saved run used the same hyperparameters as new_metadata.

    'processed_poses' is intentionally excluded from the comparison because it
    changes between runs and is not a hyperparameter — including it would cause
    a spurious cache-miss that throws away valid cached data.
    """
    _IGNORE_KEYS = {'processed_poses'}

    with open(os.path.join(saved_path, 'metadata.pkl'), 'rb') as f:
        saved_metadata = pickle.load(f)

    saved_cmp = {k: v for k, v in saved_metadata.items() if k not in _IGNORE_KEYS}
    new_cmp   = {k: v for k, v in new_metadata.items()   if k not in _IGNORE_KEYS}

    return saved_cmp == new_cmp

def create_train_val_dataloaders(
        dataset_path,
        mediapipe_model_complexity: int = 1,
        batch_size: int = 4,
        train_size: float = 0.8,
        sequence_length: int = 16,
        max_videos=None,
        load_processed=None,
        save_processed=None,
        random_state=None,
        pool_frames: bool = False,
        output_format: Literal['pytorch', 'tensorflow'] = 'pytorch',
):
    """Build train/val data-loaders using MediaPipe Pose (33 keypoints).

    Args:
        dataset_path: Root directory of the dataset.
        mediapipe_model_complexity: 0 (lite), 1 (full, default), 2 (heavy).
        batch_size: Mini-batch size for the returned loader.
        train_size: Fraction of data used for training.
        sequence_length: Number of frames sampled per video.
        max_videos: Cap on total videos processed (useful for debugging).
        load_processed: Path to look for a cached processed dataset.
        save_processed: Path where processed data (and checkpoints) are saved.
        random_state: Seed for reproducible splits.
        pool_frames: Temporally average keypoints when True.
        output_format: 'pytorch' or 'tensorflow'.
    """
    print("=== Preparing data for Train/Validation Split (MediaPipe Pose) ===")
    loader = VideoDataLoader(
        dataset_path,
        mediapipe_model_complexity=mediapipe_model_complexity,
        sequence_length=sequence_length,
        pool_frames=pool_frames,
        output_dir=save_processed,
    )

    pose_names = sorted(os.listdir(dataset_path))

    metadata = {
        'num_poses': len(pose_names),
        'pose_names': pose_names,
        'sequence_length': sequence_length,
        'mediapipe_model_complexity': mediapipe_model_complexity,
        'batch_size': batch_size,
        'pool_frames': pool_frames,
        'train_size': train_size,
    }

    if (
        load_processed
        and os.path.exists(load_processed)
        and verify_metadata(saved_path=load_processed, new_metadata=metadata)
    ):
        print("Loading processed data from disk...")
        data_splits = loader.load_processed_data(load_processed)

        if 'train' in data_splits and 'val' in data_splits:
            X_train, y_pose_train = data_splits['train']
            X_val, y_pose_val = data_splits['val']
        elif 'full' in data_splits:
            print("Creating train/val split from full dataset...")
            X, y_pose = data_splits['full']
            data_splits = loader.create_balanced_split(X, y_pose, train_size, random_state)
            X_train, y_pose_train = data_splits['train']
            X_val, y_pose_val = data_splits['val']
            if save_processed:
                loader.save_processed_data(data_splits, save_processed, batch_size)
        else:
            raise ValueError("No suitable data found in processed files")
    else:
        print("Loading videos from disk...")
        X, y_pose = loader.load_dataset(max_videos=max_videos)

        print("Creating balanced train/validation split...")
        data_splits = loader.create_balanced_split(X, y_pose, train_size, random_state)
        X_train, y_pose_train = data_splits['train']
        X_val, y_pose_val = data_splits['val']

        if save_processed:
            print("Saving processed data...")
            loader.save_processed_data(data_splits, save_processed, batch_size)

    if output_format == 'tensorflow':
        from tensorflow import keras  # lazy import — only needed when requested  # noqa: F401
        import tensorflow as tf
        train_dataset = loader.create_tensorflow_dataset(
            X_train, y_pose_train, batch_size=batch_size, shuffle=True
        )
        val_dataset = loader.create_tensorflow_dataset(
            X_val, y_pose_val, batch_size=batch_size, shuffle=False
        )
    elif output_format == 'pytorch':
        train_dataset = loader.create_pytorch_dataloader(
            X_train, y_pose_train, batch_size=batch_size, shuffle=True
        )
        val_dataset = loader.create_pytorch_dataloader(
            X_val, y_pose_val, batch_size=batch_size, shuffle=False
        )
    else:
        raise ValueError('Unsupported output format.')

    print(f"Train/Val datasets created: {len(X_train)} train, {len(X_val)} val samples")
    return train_dataset, val_dataset, loader.num_poses, loader


if __name__ == "__main__":
    try:
        # Test regular training approach with MediaPipe Pose
        print("Creating train/val split approach (MediaPipe Pose):")
        train_ds, val_ds, num_poses, loader = create_train_val_dataloaders(
            dataset_path="dataset",
            mediapipe_model_complexity=1,   # 0=lite, 1=full, 2=heavy
            batch_size=16,
            sequence_length=256,
            max_videos=None,
            save_processed="mediapipe_data",
            pool_frames=False,
        )

        print(f"\nDataset info:")
        print(f"Number of poses: {num_poses}")
        print(f"Pose names: {loader.pose_names}")

        print(f"\nTesting train dataset:")
        for videos, labels in train_ds:
            print(f"Videos shape: {videos.shape}")
            print(f"Pose labels shape: {labels.shape}")
            break

        print("\n" + "=" * 50)

    except Exception as e:
        print(f"Error testing data loading: {e}")
        import traceback
        traceback.print_exc()