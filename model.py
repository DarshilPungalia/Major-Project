import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, Optional


class SpatialGraphConv(nn.Module):
    """Spatial Graph Convolution layer for skeleton joints"""
    def __init__(self, in_channels, out_channels, num_joints=17):
        super(SpatialGraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_joints = num_joints
        
        # Learnable weight matrices for different adjacency partitions
        self.W_root = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.W_close = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.W_far = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_root)
        nn.init.xavier_uniform_(self.W_close)
        nn.init.xavier_uniform_(self.W_far)
        nn.init.zeros_(self.bias)
        
    def get_adjacency_matrix(self, device):
        """
        Creates adjacency matrix for PoseNet skeleton (17 keypoints)
        Keypoint order: nose, left_eye, right_eye, left_ear, right_ear,
                       left_shoulder, right_shoulder, left_elbow, right_elbow,
                       left_wrist, right_wrist, left_hip, right_hip,
                       left_knee, right_knee, left_ankle, right_ankle
        """
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (0, 5), (0, 6),  # Shoulders to nose
            (5, 6),  # Shoulders connected
            (5, 7), (7, 9),  # Left arm
            (6, 8), (8, 10),  # Right arm
            (5, 11), (6, 12),  # Torso
            (11, 12),  # Hips connected
            (11, 13), (13, 15),  # Left leg
            (12, 14), (14, 16),  # Right leg
        ]
        
        A = np.zeros((self.num_joints, self.num_joints))
        for i, j in edges:
            A[i, j] = 1
            A[j, i] = 1
        
        # Normalize adjacency matrix: D^(-0.5) * A * D^(-0.5)
        A = A + np.eye(self.num_joints)  # Add self-connections
        D = np.sum(A, axis=1)
        D_inv_sqrt = np.power(D, -0.5)
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0.
        D_mat = np.diag(D_inv_sqrt)
        A_normalized = D_mat @ A @ D_mat
        
        return torch.FloatTensor(A_normalized).to(device)
    
    def forward(self, x):
        # x shape: (batch, time, num_joints, features)
        batch_size, time_steps, num_joints, features = x.shape
        device = x.device
        
        # Get adjacency matrix
        A = self.get_adjacency_matrix(device)
        
        # Create partition masks (root, close neighbors, far neighbors)
        A_root = torch.eye(self.num_joints, device=device)
        A_close = (A > 0).float() * (1 - A_root)
        A_far = (torch.matrix_power(A, 2) > 0).float() * (1 - A_close - A_root)
        
        # Reshape for matrix operations
        x_reshaped = x.view(-1, num_joints, features)
        
        # Apply graph convolution for each partition
        out_root = torch.matmul(A_root, x_reshaped)
        out_root = torch.matmul(out_root, self.W_root)
        
        out_close = torch.matmul(A_close, x_reshaped)
        out_close = torch.matmul(out_close, self.W_close)
        
        out_far = torch.matmul(A_far, x_reshaped)
        out_far = torch.matmul(out_far, self.W_far)
        
        # Combine and add bias
        out = out_root + out_close + out_far + self.bias
        
        # Reshape back to original temporal structure
        out = out.view(batch_size, time_steps, num_joints, self.out_channels)
        
        return out


class TemporalConv(nn.Module):
    """Temporal convolution along the time axis"""
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(TemporalConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(padding, 0)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        # x shape: (batch, time, num_joints, features)
        # Conv2d expects: (batch, channels, height, width)
        # Permute to: (batch, features, time, num_joints)
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.bn(x)
        # Permute back to: (batch, time, num_joints, features)
        x = x.permute(0, 2, 3, 1)
        return x


class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolution Block"""
    def __init__(self, in_channels, out_channels, num_joints=17, stride=1, residual=True):
        super(STGCNBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_joints = num_joints
        self.stride = stride
        self.residual = residual
        
        # Spatial graph convolution
        self.gcn = SpatialGraphConv(in_channels, out_channels, num_joints=num_joints)
        
        # Temporal convolution
        self.tcn = TemporalConv(out_channels, out_channels, kernel_size=9, stride=stride)
        
        # Activation and regularization
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        
        # Residual connection
        if not residual:
            self.residual_conv = None
        elif in_channels != out_channels or stride != 1:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(1, 1),
                    stride=(stride, 1),
                    padding=0
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual_conv = None
            
    def forward(self, x):
        residual = x
        
        # Spatial graph convolution
        x = self.gcn(x)
        x = self.relu(x)
        
        # Temporal convolution
        x = self.tcn(x)
        
        # Residual connection
        if self.residual:
            if self.residual_conv is not None:
                # Permute for Conv2d: (batch, features, time, joints)
                residual = residual.permute(0, 3, 1, 2)
                residual = self.residual_conv(residual)
                # Permute back: (batch, time, joints, features)
                residual = residual.permute(0, 2, 3, 1)
            x = x + residual
            
        x = self.relu(x)
        x = self.dropout(x)
        
        return x


class VideoModel(nn.Module):
    def __init__(self, num_poses, input_shape, num_joints=17, learning_rate=1e-4):
        """
        ST-GCN based model for skeleton-based action recognition
        
        Args:
            num_poses: Number of action classes
            input_shape: (sequence_length, num_joints * features_per_joint) 
                        For PoseNet: (seq_len, 17*2) for (x,y) or (seq_len, 17*3) for (x,y,confidence)
            num_joints: Number of skeleton keypoints (17 for PoseNet)
            learning_rate: Learning rate for optimizer
        """
        super(VideoModel, self).__init__()
        
        self.input_shape = input_shape
        self.num_poses = num_poses
        self.num_joints = num_joints
        self.learning_rate = learning_rate
        self.fitted = False
        
        # Infer features per joint from input shape
        self.features_per_joint = input_shape[1] // num_joints
        self.seq_length = input_shape[0]
        
        # Set device (GPU if available)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build model
        self._build_model()
        
        # Move model to device
        self.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"Model initialized on device: {self.device}")
        
    def _build_model(self):
        # Initial projection to higher dimension
        self.initial_projection = nn.Linear(self.features_per_joint, 64)
        
        # ST-GCN blocks with increasing channels
        self.stgcn_block1 = STGCNBlock(64, 64, num_joints=self.num_joints, stride=1, residual=True)
        self.stgcn_block2 = STGCNBlock(64, 64, num_joints=self.num_joints, stride=1, residual=True)
        self.stgcn_block3 = STGCNBlock(64, 128, num_joints=self.num_joints, stride=2, residual=True)
        self.stgcn_block4 = STGCNBlock(128, 128, num_joints=self.num_joints, stride=1, residual=True)
        self.stgcn_block5 = STGCNBlock(128, 256, num_joints=self.num_joints, stride=2, residual=True)
        self.stgcn_block6 = STGCNBlock(256, 256, num_joints=self.num_joints, stride=1, residual=True)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification head
        self.fc1 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, self.num_poses)
        
    def forward(self, x):
        # Input: (batch, sequence_length, num_joints * features_per_joint)
        batch_size = x.shape[0]
        
        # Reshape to (batch, sequence_length, num_joints, features_per_joint)
        x = x.view(batch_size, self.seq_length, self.num_joints, self.features_per_joint)
        
        # Initial projection
        x = self.initial_projection(x)
        
        # ST-GCN blocks
        x = self.stgcn_block1(x)
        x = self.stgcn_block2(x)
        x = self.stgcn_block3(x)
        x = self.stgcn_block4(x)
        x = self.stgcn_block5(x)
        x = self.stgcn_block6(x)
        
        # Global pooling: (batch, time, joints, features) -> (batch, features, time, joints)
        x = x.permute(0, 3, 1, 2)
        x = self.global_pool(x)
        x = x.view(batch_size, -1)
        
        # Classification head
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def fit(self, x: Union[torch.utils.data.DataLoader, np.ndarray], 
            y: Optional[np.ndarray] = None,
            validation_data: Optional[Union[torch.utils.data.DataLoader, Tuple[np.ndarray, np.ndarray]]] = None,
            epochs: int = 10,
            verbose: bool = True,
            batch_size: int = 8,
            steps_per_epoch: Optional[int] = None,
            validation_steps: Optional[int] = None):
        """
        Train the model
        
        Args:
            x: Training data (DataLoader or numpy array)
            y: Training labels (if x is numpy array)
            validation_data: Validation data
            epochs: Number of training epochs
            verbose: Whether to print training progress
            batch_size: Batch size (if x is numpy array)
            steps_per_epoch: Steps per epoch
            validation_steps: Validation steps
        """
        history = {
            'loss': [],
            'accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        # Handle numpy arrays
        if isinstance(x, np.ndarray):
            if y is None:
                raise ValueError('y is necessary when x is of type ndarray')
            
            # Convert to PyTorch tensors
            x_tensor = torch.FloatTensor(x).to(self.device)
            y_tensor = torch.LongTensor(y).to(self.device)
            
            # Create dataset and dataloader
            train_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            
            # Handle validation data
            if validation_data is not None:
                if not isinstance(validation_data, tuple):
                    raise TypeError(f'Expected tuple, got {type(validation_data)}')
                
                val_x, val_y = validation_data
                val_x_tensor = torch.FloatTensor(val_x).to(self.device)
                val_y_tensor = torch.LongTensor(val_y).to(self.device)
                
                val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_y_tensor)
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False
                )
            else:
                val_loader = None
        
        elif isinstance(x, torch.utils.data.DataLoader):
            train_loader = x
            val_loader = validation_data
        else:
            raise TypeError(f'Expected x to be DataLoader or ndarray, got {type(x)}')
        
        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Convert labels from one-hot to class indices if needed
                if labels.dim() > 1 and labels.shape[1] > 1:
                    labels = torch.argmax(labels, dim=1)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if steps_per_epoch and batch_idx >= steps_per_epoch:
                    break
            
            epoch_loss = train_loss / len(train_loader)
            epoch_acc = train_correct / train_total
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            
            # Validation phase
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_idx, (inputs, labels) in enumerate(val_loader):
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        
                        # Convert labels from one-hot to class indices if needed
                        if labels.dim() > 1 and labels.shape[1] > 1:
                            labels = torch.argmax(labels, dim=1)
                        
                        outputs = self(inputs)
                        loss = self.criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                        
                        if validation_steps and batch_idx >= validation_steps:
                            break
                
                val_epoch_loss = val_loss / len(val_loader)
                val_epoch_acc = val_correct / val_total
                history['val_loss'].append(val_epoch_loss)
                history['val_accuracy'].append(val_epoch_acc)
                
                if verbose:
                    print(f'Epoch {epoch+1}/{epochs} - '
                          f'loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - '
                          f'val_loss: {val_epoch_loss:.4f} - val_acc: {val_epoch_acc:.4f}')
            else:
                if verbose:
                    print(f'Epoch {epoch+1}/{epochs} - '
                          f'loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}')
        
        self.fitted = True
        return history
    
    def predict(self, x):
        """
        Make predictions
        
        Args:
            x: Input data (numpy array or tensor)
            
        Returns:
            Predictions as numpy array
        """
        if not self.fitted:
            raise ValueError('Call .fit() first')
        
        self.eval()
        
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x).to(self.device)
            elif isinstance(x, torch.Tensor):
                x = x.to(self.device)
            
            outputs = self(x)
            predictions = F.softmax(outputs, dim=1)
            
        return predictions.cpu().numpy()
    
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_shape': self.input_shape,
            'num_poses': self.num_poses,
            'num_joints': self.num_joints,
            'fitted': self.fitted
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fitted = checkpoint['fitted']
        print(f"Model loaded from {filepath}")