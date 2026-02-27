import numpy as np
import cv2
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle
from typing import Literal

class VideoDataLoader:
    def __init__(self, dataset_path, sequence_length=16, movenet_variant: Literal['thunder', 'lightning']='thunder', pool_frames=True):
        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        self.movenet_variant = str.lower(movenet_variant)
        self.pool_frames = pool_frames
        self.target_size = None
        self.batch_size = None
        self.train_size = None
        
        # Storage for data
        self.videos = []
        self.pose_labels = []
        self.pose_names = []
        
        self._load_dataset_info()
    
    def _load_dataset_info(self):
        """Load dataset structure and file paths"""
        # Get pose folders
        pose_folders = sorted([f for f in os.listdir(self.dataset_path) 
                              if os.path.isdir(os.path.join(self.dataset_path, f))])
        
        self.pose_names = pose_folders
        self.num_poses = len(pose_folders)
                
        for pose_idx, pose_name in enumerate(pose_folders):
            pose_path = os.path.join(self.dataset_path, pose_name)
                     
            video_files = [f for f in os.listdir(pose_path) 
                          if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                
            for video_file in video_files:
                video_path = os.path.join(pose_path, video_file)
                self.videos.append(video_path)
                self.pose_labels.append(pose_idx)
        
        
        print(f"Found {len(self.videos)} videos")
        print(f"Poses ({self.num_poses}): {self.pose_names}")
    
    def load_video_frames(self, video_path):
        """Load and preprocess video frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []

        variant_to_size = {
            'lightning': (192, 192),
            'thunder': (256, 256)
        }

        self.target_size = variant_to_size.get(self.movenet_variant, (256, 256))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize and convert to RGB
            frame = cv2.resize(frame, self.target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.int32) 
            frames.append(frame)
        
        cap.release()
        
        # Handle sequence length
        if len(frames) >= self.sequence_length:
            # Sample frames evenly across the video
            indices = np.linspace(0, len(frames)-1, self.sequence_length).astype(int)
            frames = [frames[i] for i in indices]
        else:
            # Repeat last frame if not enough frames
            while len(frames) < self.sequence_length:
                frames.append(frames[-1])
        
        frames = np.array(frames)  
        
        return frames
    
    def extract_keypoints(self, frames)-> np.ndarray:
        variant_to_path = {
            'lightning': './movenet/singlepose-lightning/4',
            'thunder': './movenet/singlepose-thunder/4'
        }

        path = variant_to_path.get(self.movenet_variant, 'thunder')
        model = tf.saved_model.load(path)

        movenet = model.signatures['serving_default']
        video_keypoints = []

        for frame in frames:
            frame = tf.expand_dims(frame, axis=0)
            output = movenet(frame)
            kp = output['output_0']
            kp = tf.reshape(kp[0, 0], shape=(-1,)).numpy()
            video_keypoints.append(kp)

        video_keypoints = np.array(video_keypoints)

        return video_keypoints
    
    def load_all_videos(self, max_videos=None):
        """Load all videos into numpy arrays"""
        if max_videos:
            video_paths = self.videos[:max_videos]
            pose_labels = self.pose_labels[:max_videos]
        else:
            video_paths = self.videos
            pose_labels = self.pose_labels
        
        X = []
        y_pose = []
        
        print(f"Loading {len(video_paths)} videos...")
        
        for i, video_path in enumerate(video_paths):
            if i % 10 == 0:
                print(f"Loading video {i+1}/{len(video_paths)}")
            
            try:
                frames = self.load_video_frames(video_path)
                keypoints = self.extract_keypoints(frames)
                X.append(keypoints)
                y_pose.append(pose_labels[i])
            except Exception as e:
                print(f"Error loading {video_path}: {e}")
                continue
        
        X = np.array(X)  
        y_pose = np.array(y_pose)

        if self.pool_frames:
            X = X.mean(axis=1)  
            print(f"Frame-pooled data shape: {X.shape}")
        
        print(f"Loaded {len(X)} videos successfully")
        print(f"Video data shape: {X.shape}")
        print(f"Pose labels shape: {y_pose.shape}")
        
        return X, y_pose
        
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
        
        use_stratification = (min_samples_per_class >= 2 and 
                             min_val_samples >= num_classes and 
                             len(X) > num_classes * 2)
        
        if not use_stratification:
            print(f"\nWarning: Dataset too small for stratified splitting.")
            print(f"  Total samples: {len(X)}, Classes: {num_classes}")
            print(f"  Min samples per class: {min_samples_per_class}")
            print("Falling back to random splitting without stratification.")
            
            if len(X) < 2:
                raise ValueError(f"Dataset too small ({len(X)} samples) to create train/val split")
            
            # Use random splitting instead of stratified
            X_train, X_val, y_pose_train, y_pose_val = train_test_split(
                X, y_pose,
                train_size=train_size,
                random_state=random_state
            )
        else:
            print(f"\nUsing stratified splitting (min {min_samples_per_class} samples per class).")
            
            try:
                X_train, X_val, y_pose_train, y_pose_val = train_test_split(
                    X, y_pose,
                    train_size=train_size,
                    stratify=y_pose,
                    random_state=random_state
                )
            except ValueError as e:
                print(f"Stratified split failed: {e}")
                print("Falling back to random splitting...")
                X_train, X_val, y_pose_train, y_pose_val = train_test_split(
                    X, y_pose,
                    train_size=train_size,
                    random_state=random_state
                )
        
        # Print split statistics
        print(f"\nDataset split:")
        print(f"  Train: {len(X_train)} videos")
        print(f"  Val: {len(X_val)} videos")
        
        # Print class distribution
        self._print_split_distribution(y_pose_train, "Train")
        self._print_split_distribution(y_pose_val, "Val")
        
        return {
            'train': (X_train, y_pose_train),
            'val': (X_val, y_pose_val)
        }
    
    def _print_split_distribution(self, y_pose, split_name):
        """Print class distribution for a data split"""
        label_counts = Counter(y_pose)
        
        print(f"\n{split_name} split distribution:")
        for label, count in sorted(label_counts.items()):
            print(f"{label}: {count} videos")
    
    def create_tensorflow_dataset(self, X, y_pose, batch_size=4, shuffle=True, prefetch=True):
        """Convert numpy arrays to TensorFlow Dataset"""
        y_pose_onehot = tf.keras.utils.to_categorical(y_pose, num_classes=self.num_poses)
        
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices((
            X.astype(np.float32),
            y_pose_onehot.astype(np.float32),
        ))
        
        self.batch_size = batch_size

        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(X))
        
        dataset = dataset.batch(batch_size)
        
        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def save_processed_data(self, data_splits, save_path):
        """Save processed data to disk"""
        os.makedirs(save_path, exist_ok=True)
        
        metadata = {
            'num_poses': self.num_poses,
            'pose_names': self.pose_names,
            'sequence_length': self.sequence_length,
            'movenet_variant': self.movenet_variant,
            'batch_size': self.batch_size,
            'pool_frames': self.pool_frames,
            'train_size': self.train_size
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
        '''with open(os.path.join(save_path, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        self.num_poses = metadata['num_poses']
        self.pose_names = metadata['pose_names']
        self.sequence_length = metadata['sequence_length']
        self.movenet_variant = metadata['movenet_variant']'''
        
        data_splits = {}
        possible_splits = ['train', 'val', 'full']
        
        for split_name in possible_splits:
            X_path = os.path.join(save_path, f'{split_name}_X.npy')
            if os.path.exists(X_path):
                X = np.load(X_path)
                y = np.load(os.path.join(save_path, f'{split_name}_y.npy'))
                data_splits[split_name] = (X, y)
        
        return data_splits


def verify_metadata(saved_path , new_metadata:dict):
    with open(os.path.join(saved_path, 'metadata.pkl'), 'rb') as f:
        saved_metadata = pickle.load(f)

    if saved_metadata == new_metadata:
        return True
    
    return False

def create_train_val_dataloaders(dataset_path, movenet_variant: Literal['thunder', 'lightning']='thunder', batch_size=4, train_size=0.8, 
                               sequence_length=16, max_videos=None,
                               load_processed=None, save_processed=None, random_state=None, pool_frames=False):

    print("=== Preparing data for Train/Validation Split ===")
    loader = VideoDataLoader(dataset_path, movenet_variant=movenet_variant, sequence_length=sequence_length, pool_frames=pool_frames)

    pose_names = os.listdir(dataset_path)
    metadata = {
            'num_poses': len(pose_names),
            'pose_names': pose_names,
            'sequence_length': sequence_length,
            'movenet_variant': movenet_variant,
            'batch_size': batch_size,
            'pool_frames': pool_frames,
            'train_size': train_size
    }
    
    if load_processed and os.path.exists(load_processed) and verify_metadata(saved_path=load_processed, new_metadata=metadata):
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
                loader.save_processed_data(data_splits, save_processed)
        else:
            raise ValueError("No suitable data found in processed files")
    else:
        print("Loading videos from disk...")
        X, y_pose = loader.load_all_videos(max_videos=max_videos)
        
        print("Creating balanced train/validation split...")
        data_splits = loader.create_balanced_split(X, y_pose, train_size, random_state)
        X_train, y_pose_train = data_splits['train']
        X_val, y_pose_val = data_splits['val']
        
        if save_processed:
            print("Saving processed data...")
            loader.save_processed_data(data_splits, save_processed)
    
    train_dataset = loader.create_tensorflow_dataset(
        X_train, y_pose_train, 
        batch_size=batch_size, shuffle=True
    )
    
    val_dataset = loader.create_tensorflow_dataset(
        X_val, y_pose_val,
        batch_size=batch_size, shuffle=False
    )
    
    print(f"Train/Val datasets created: {len(X_train)} train, {len(X_val)} val samples")
    return train_dataset, val_dataset, loader.num_poses, loader


if __name__ == "__main__":
    try:
        # Test regular training approach
        print("Testing train/val split approach:")
        train_ds, val_ds, num_poses, loader = create_train_val_dataloaders(
            "dataset",
            movenet_variant='thunder', 
            batch_size=16,
            sequence_length=64, 
            max_videos=None,
            save_processed="thunder_data",
            pool_frames=True
        )
        
        print(f"\nDataset info:")
        print(f"Number of poses: {num_poses}")
        print(f"Pose names: {loader.pose_names}")
        
        print(f"\nTesting train dataset:")
        for videos, labels in train_ds.take(1):
            print(f"Videos shape: {videos.shape}")
            print(f"Pose labels shape: {labels.shape}")
            break
            
        print("\n" + "="*50)
            
    except Exception as e:
        print(f"Error testing data loading: {e}")
        import traceback
        traceback.print_exc()