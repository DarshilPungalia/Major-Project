import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader
import numpy as np


class SpatialGraphConv(MessagePassing):
    """Spatial Graph Convolution using PyTorch Geometric"""
    def __init__(self, in_channels, out_channels, num_partitions=3):
        super(SpatialGraphConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_partitions = num_partitions
        
        # Learnable weight matrices for different partitions
        self.weight = nn.Parameter(torch.FloatTensor(num_partitions, in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: Node features (batch_size * num_joints * time_steps, in_channels)
            edge_index: Graph connectivity (2, num_edges)
            edge_attr: Edge partition labels (num_edges,) indicating partition type
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        # x_j: features of neighbor nodes (num_edges, in_channels)
        # edge_attr: partition labels (num_edges,)
        
        # Apply partition-specific weights
        out = torch.zeros(x_j.size(0), self.out_channels, device=x_j.device)
        for i in range(self.num_partitions):
            mask = (edge_attr == i)
            if mask.any():
                out[mask] = torch.matmul(x_j[mask], self.weight[i])
        
        return out
    
    def update(self, aggr_out):
        # Add bias after aggregation
        return aggr_out + self.bias


class TemporalConv(nn.Module):
    """Temporal Convolution along time axis"""
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dropout=0.0):
        super(TemporalConv, self).__init__()
        
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            stride=(stride, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class STGCNBlock(nn.Module):
    """Spatial-Temporal Graph Convolution Block"""
    def __init__(self, in_channels, out_channels, num_joints=17, stride=1, 
                 residual=True, dropout=0.3):
        super(STGCNBlock, self).__init__()
        
        self.gcn = SpatialGraphConv(in_channels, out_channels)
        self.tcn = TemporalConv(out_channels, out_channels, kernel_size=9, 
                                stride=stride, dropout=dropout)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, 
                                        kernel_size=1, stride=stride)
    
    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x: (batch_size, in_channels, time_steps, num_joints)
            edge_index: Graph connectivity
            edge_attr: Edge partition labels
        """
        res = self.residual(x)
        
        # Spatial graph convolution
        batch_size, in_channels, time_steps, num_joints = x.size()
        
        # Reshape for graph convolution: (batch * time * joints, channels)
        x_graph = x.permute(0, 2, 3, 1).contiguous()  # (B, T, V, C)
        x_graph = x_graph.view(-1, in_channels)  # (B*T*V, C)
        
        # Apply graph convolution
        x_graph = self.gcn(x_graph, edge_index, edge_attr)
        x_graph = self.relu(x_graph)
        
        # Reshape back: (B, C, T, V)
        x = x_graph.view(batch_size, time_steps, num_joints, -1)
        x = x.permute(0, 3, 1, 2).contiguous()
        
        # Temporal convolution
        x = self.tcn(x)
        
        # Residual connection
        x = x + res
        x = self.relu(x)
        
        return x


class STGCNModel(nn.Module):
    """ST-GCN Model for Action Recognition"""
    def __init__(self, num_classes, num_joints=17, in_channels=2, 
                 edge_importance_weighting=True, dropout=0.5):
        super(STGCNModel, self).__init__()
        
        self.num_classes = num_classes
        self.num_joints = num_joints
        self.in_channels = in_channels
        
        # Build graph structure (edge_index and edge_attr)
        self.edge_index, self.edge_attr = self._build_graph()
        
        # Data batch normalization
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        
        # ST-GCN blocks
        self.st_gcn_blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64, num_joints, stride=1, residual=False, dropout=0.0),
            STGCNBlock(64, 64, num_joints, stride=1, residual=True, dropout=dropout),
            STGCNBlock(64, 64, num_joints, stride=1, residual=True, dropout=dropout),
            STGCNBlock(64, 128, num_joints, stride=2, residual=True, dropout=dropout),
            STGCNBlock(128, 128, num_joints, stride=1, residual=True, dropout=dropout),
            STGCNBlock(128, 256, num_joints, stride=2, residual=True, dropout=dropout),
            STGCNBlock(256, 256, num_joints, stride=1, residual=True, dropout=dropout),
        ])
        
        # Edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.edge_index.size(1)))
                for _ in self.st_gcn_blocks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_blocks)
        
        # Classification head
        self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def _build_graph(self):
        """
        Build graph structure for PoseNet skeleton (17 keypoints)
        Returns edge_index and edge_attr (partition labels)
        """
        # PoseNet keypoints (COCO format):
        # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
        # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
        # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
        
        neighbor_link = [
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
        
        self_link = [(i, i) for i in range(self.num_joints)]
        
        # Create edge_index (bidirectional)
        edges = []
        edge_attrs = []
        
        # Self connections (partition 0)
        for i, j in self_link:
            edges.append([i, j])
            edge_attrs.append(0)
        
        # Neighbor connections (partition 1)
        for i, j in neighbor_link:
            edges.append([i, j])
            edges.append([j, i])  # Bidirectional
            edge_attrs.extend([1, 1])
        
        # Second-order neighbors (partition 2)
        # Compute 2-hop neighbors
        adjacency = np.zeros((self.num_joints, self.num_joints))
        for i, j in neighbor_link:
            adjacency[i, j] = 1
            adjacency[j, i] = 1
        
        # A^2 for second-order
        hop_2 = np.linalg.matrix_power(adjacency, 2)
        for i in range(self.num_joints):
            for j in range(i + 1, self.num_joints):
                if hop_2[i, j] > 0 and adjacency[i, j] == 0:  # 2-hop but not 1-hop
                    edges.append([i, j])
                    edges.append([j, i])
                    edge_attrs.extend([2, 2])
        
        edge_index = torch.LongTensor(edges).t()
        edge_attr = torch.LongTensor(edge_attrs)
        
        return edge_index, edge_attr
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, in_channels, time_steps, num_joints)
        """
        # Handle TensorFlow tensors (convert to PyTorch)
        if not isinstance(x, torch.Tensor) and hasattr(x, 'numpy'):  # TensorFlow tensor
            x = torch.from_numpy(x.numpy()).float()
            if torch.cuda.is_available():
                x = x.cuda()
        
        batch_size, channels, time_steps, num_joints = x.size()
        
        # Data normalization
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, T, V, C)
        x = x.view(batch_size, time_steps, -1)  # (B, T, V*C)
        x = self.data_bn(x.permute(0, 2, 1))  # (B, V*C, T)
        x = x.view(batch_size, num_joints, channels, time_steps)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, C, T, V)
        
        # Move graph to same device as input
        edge_index = self.edge_index.to(x.device)
        edge_attr = self.edge_attr.to(x.device)
        
        # ST-GCN blocks
        for gcn, importance in zip(self.st_gcn_blocks, self.edge_importance):
            # Apply edge importance weighting
            if isinstance(importance, nn.Parameter):
                weighted_edge_attr = edge_attr  # Partition labels remain same
                # Importance is applied during message passing
            x = gcn(x, edge_index, weighted_edge_attr)
        
        # Global pooling
        x = F.avg_pool2d(x, x.size()[2:])  # (B, 256, 1, 1)
        
        # Classification
        x = self.fcn(x)  # (B, num_classes, 1, 1)
        x = x.view(batch_size, -1)  # (B, num_classes)
        
        return x


class VideoModel:
    """Wrapper class to maintain API compatibility with original TensorFlow code"""
    def __init__(self, num_poses, input_shape, num_joints=17, 
                 learning_rate=1e-3, device=None):
        """
        Args:
            num_poses: Number of action classes
            input_shape: (sequence_length, num_joints * features_per_joint)
                        For PoseNet: (seq_len, 17*2) for (x,y) or (seq_len, 17*3) for (x,y,conf)
            num_joints: Number of skeleton keypoints (17 for PoseNet)
            learning_rate: Learning rate for optimizer
            device: 'cuda' or 'cpu', auto-detected if None
        """
        self.num_poses = num_poses
        self.input_shape = input_shape
        self.num_joints = num_joints
        self.learning_rate = learning_rate
        self.fitted = False
        
        # Auto-detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Infer features per joint
        self.features_per_joint = input_shape[1] // num_joints
        self.sequence_length = input_shape[0]
        
        # Build model
        self.model = STGCNModel(
            num_classes=num_poses,
            num_joints=num_joints,
            in_channels=self.features_per_joint,
            dropout=0.5
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.0001
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        print(f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def _prepare_data(self, x, y=None):
        """Convert numpy arrays to PyTorch tensors and reshape"""
        if isinstance(x, np.ndarray):
            # x shape: (batch, seq_len, num_joints * features)
            # Target shape: (batch, features, seq_len, num_joints)
            batch_size = x.shape[0]
            x_reshaped = x.reshape(batch_size, self.sequence_length, 
                                  self.num_joints, self.features_per_joint)
            x_reshaped = x_reshaped.transpose(0, 3, 1, 2)  # (B, C, T, V)
            x_tensor = torch.FloatTensor(x_reshaped).to(self.device)
        else:
            x_tensor = x.to(self.device)
        
        if y is not None:
            if isinstance(y, np.ndarray):
                # Convert one-hot to class indices if needed
                if len(y.shape) > 1 and y.shape[1] > 1:
                    y = np.argmax(y, axis=1)
                y_tensor = torch.LongTensor(y).to(self.device)
            else:
                y_tensor = y.to(self.device)
            return x_tensor, y_tensor
        
        return x_tensor
    
    def fit(self, x, y=None, validation_data=None, epochs=10, 
            verbose=True, batch_size=8, steps_per_epoch=None, validation_steps=None):
        """
        Train the model
        
        Args:
            x: Training data (numpy array or DataLoader)
            y: Training labels (numpy array, required if x is numpy)
            validation_data: Tuple of (x_val, y_val) or DataLoader
            epochs: Number of training epochs
            verbose: Whether to print training progress
            batch_size: Batch size (used when x is numpy array)
        """
        self.model.train()
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        # Prepare data
        if isinstance(x, np.ndarray):
            if y is None:
                raise ValueError('y is necessary when x is of type ndarray')
            
            dataset = torch.utils.data.TensorDataset(
                *self._prepare_data(x, y)
            )
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            train_loader = x
        
        # Prepare validation data
        if validation_data is not None:
            if isinstance(validation_data, tuple):
                val_x, val_y = validation_data
                val_dataset = torch.utils.data.TensorDataset(
                    *self._prepare_data(val_x, val_y)
                )
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            else:
                val_loader = validation_data
        
        # Training loop
        for epoch in range(epochs):
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                # Move batch to device
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Calculate metrics
            epoch_loss = train_loss / len(train_loader)
            epoch_acc = 100 * train_correct / train_total
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            
            # Validation
            if validation_data is not None:
                val_loss, val_acc = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                if verbose:
                    print(f'Epoch [{epoch+1}/{epochs}] '
                          f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% '
                          f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%')
            else:
                if verbose:
                    print(f'Epoch [{epoch+1}/{epochs}] '
                          f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
        
        self.fitted = True
        return history
    
    def _validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                # Move batch to device
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        self.model.train()
        return val_loss / len(val_loader), 100 * val_correct / val_total
    
    def predict(self, x):
        """
        Make predictions
        
        Args:
            x: Input data (numpy array)
            
        Returns:
            Predictions as numpy array (softmax probabilities)
        """
        if not self.fitted:
            raise ValueError('Call .fit() first')
        
        self.model.eval()
        
        with torch.no_grad():
            x_tensor = self._prepare_data(x)
            outputs = self.model(x_tensor)
            predictions = F.softmax(outputs, dim=1)
        
        return predictions.cpu().numpy()
    
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_poses': self.num_poses,
            'input_shape': self.input_shape,
            'num_joints': self.num_joints,
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.fitted = True
        print(f"Model loaded from {filepath}")