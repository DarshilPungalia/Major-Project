import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
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
    def __init__(self, in_channels, out_channels, num_joints=33, stride=1, 
                 residual=True, dropout=0.2):
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
    def __init__(self, num_classes, num_joints=33, in_channels=3,
                 edge_importance_weighting=True, dropout=0.3):
        super(STGCNModel, self).__init__()
        
        self.num_classes = num_classes
        self.num_joints = num_joints
        self.in_channels = in_channels
        
        # Build graph structure (edge_index and edge_attr)
        edge_index, edge_attr = self._build_graph()
        self.register_buffer('edge_index', edge_index)
        self.register_buffer('edge_attr', edge_attr)
        
        # FIXED: Correct batch normalization for (B, C, T, V) format
        self.data_bn = nn.BatchNorm2d(in_channels)
        
        # FIXED: Simplified architecture for small datasets
        self.st_gcn_blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64, num_joints, stride=1, residual=False, dropout=0.0),
            STGCNBlock(64, 64, num_joints, stride=1, residual=True, dropout=dropout),
            STGCNBlock(64, 64, num_joints, stride=1, residual=True, dropout=dropout),
            STGCNBlock(64, 128, num_joints, stride=2, residual=True, dropout=dropout),
            STGCNBlock(128, 128, num_joints, stride=1, residual=True, dropout=dropout),
        ])
        
        # Edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.edge_index.size(1)))
                for _ in self.st_gcn_blocks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_blocks)
        
        # Global average pooling + classifier
        self.fcn = nn.Conv2d(128, num_classes, kernel_size=1)
        
    def _build_graph(self):
        """
        Build graph structure for MediaPipe Pose skeleton (33 keypoints).
        Returns edge_index and edge_attr (partition labels).

        MediaPipe Pose landmark indices:
          0: nose
          1: left_eye_inner,  2: left_eye,   3: left_eye_outer
          4: right_eye_inner, 5: right_eye,  6: right_eye_outer
          7: left_ear,        8: right_ear
          9: mouth_left,     10: mouth_right
         11: left_shoulder,  12: right_shoulder
         13: left_elbow,     14: right_elbow
         15: left_wrist,     16: right_wrist
         17: left_pinky,     18: right_pinky
         19: left_index,     20: right_index
         21: left_thumb,     22: right_thumb
         23: left_hip,       24: right_hip
         25: left_knee,      26: right_knee
         27: left_ankle,     28: right_ankle
         29: left_heel,      30: right_heel
         31: left_foot_index,32: right_foot_index
        """
        neighbor_link = [
            # Face connections
            (0, 1), (1, 2), (2, 3),           # nose → left eye chain
            (0, 4), (4, 5), (5, 6),           # nose → right eye chain
            (3, 7), (6, 8),                   # outer eyes → ears
            (9, 10),                          # mouth
            # Shoulders
            (11, 12),
            # Left arm
            (11, 13), (13, 15),
            (15, 17), (15, 19), (15, 21),    # wrist → pinky/index/thumb
            (17, 19),                         # pinky ↔ index (hand)
            # Right arm
            (12, 14), (14, 16),
            (16, 18), (16, 20), (16, 22),    # wrist → pinky/index/thumb
            (18, 20),                         # pinky ↔ index (hand)
            # Torso
            (11, 23), (12, 24),
            (23, 24),                         # hips connected
            # Left leg
            (23, 25), (25, 27),
            (27, 29), (27, 31),              # ankle → heel/foot_index
            (29, 31),                         # heel ↔ foot_index
            # Right leg
            (24, 26), (26, 28),
            (28, 30), (28, 32),              # ankle → heel/foot_index
            (30, 32),                         # heel ↔ foot_index
        ]

        self_link = [(i, i) for i in range(self.num_joints)]

        # Create edge_index (bidirectional)
        edges = []
        edge_attrs = []

        # Self connections (partition 0)
        for i, j in self_link:
            edges.append([i, j])
            edge_attrs.append(0)

        # Neighbor connections (partition 1 — bidirectional)
        for i, j in neighbor_link:
            edges.append([i, j])
            edge_attrs.append(1)
            edges.append([j, i])   # reverse direction
            edge_attrs.append(1)

        # Second-order neighbors (partition 2)
        adj_matrix = {i: set() for i in range(self.num_joints)}
        for i, j in neighbor_link:
            adj_matrix[i].add(j)
            adj_matrix[j].add(i)

        second_order = set()
        for i in range(self.num_joints):
            for neighbor in adj_matrix[i]:
                for second_neighbor in adj_matrix[neighbor]:
                    if second_neighbor != i and second_neighbor not in adj_matrix[i]:
                        second_order.add((i, second_neighbor))

        for i, j in second_order:
            edges.append([i, j])
            edge_attrs.append(2)

        edge_index = torch.LongTensor(edges).t()
        edge_attr  = torch.LongTensor(edge_attrs)

        return edge_index, edge_attr
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, in_channels, time_steps, num_joints)
               For MediaPipe Pose: (B, 3, T, 33)
        """
        # Apply batch normalization
        x = self.data_bn(x)
        
        # Apply ST-GCN blocks
        for gcn, importance in zip(self.st_gcn_blocks, self.edge_importance):
            # FIXED: Create batched edge index for each frame
            batch_size, _, time_steps, _ = x.size()
            
            # Replicate edge_index for batch*time_steps graphs
            num_graphs = batch_size * time_steps
            edge_index = self.edge_index
            edge_attr = self.edge_attr
            
            # Create batched edge_index
            edge_index_batch = edge_index.repeat(1, num_graphs)
            offset = torch.arange(num_graphs, device=x.device) * self.num_joints
            offset = offset.repeat_interleave(edge_index.size(1))
            edge_index_batch = edge_index_batch + offset
            
            # Create batched edge_attr
            edge_attr_batch = edge_attr.repeat(num_graphs)
            
            # Apply importance weighting
            if isinstance(importance, nn.Parameter):
                edge_weight = importance.repeat(num_graphs)
            else:
                edge_weight = None
            
            x = gcn(x, edge_index_batch, edge_attr_batch)
        
        # Global average pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1, 1, 1)
        
        # Classification
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        
        return x


class VideoModel:
    """Wrapper class compatible with your training code"""
    def __init__(self, num_poses, input_shape, num_joints=33,
                 learning_rate=0.001, device=None):
        """
        Args:
            num_poses: Number of action classes
            input_shape: Tuple (features_per_joint, time_steps)
                        For MediaPipe Pose: (3, T) where 3 = [x_norm, y_norm, visibility]
            num_joints: Number of skeleton joints (33 for MediaPipe Pose)
            learning_rate: Learning rate for optimizer
        """
        self.num_poses = num_poses
        self.input_shape = input_shape
        self.num_joints = num_joints
        self.fitted = False
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.features_per_joint = input_shape[0]  # 3 for MediaPipe Pose (x_norm, y_norm, visibility)
        self.sequence_length = input_shape[1]     # number of frames
        
        print(f"Features per joint: {self.features_per_joint}")
        print(f"Sequence length: {self.sequence_length}")
        
        # Build model
        self.model = STGCNModel(
            num_classes=num_poses,
            num_joints=num_joints,
            in_channels=self.features_per_joint,
            dropout=0.3  # FIXED: Reduced from 0.5
        ).to(self.device)
        
        # FIXED: Better optimizer settings
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.0001
        )
        
        # FIXED: Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5,
            min_lr=1e-6
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model created with {num_params:,} parameters")
    
    def _prepare_data(self, x, y=None):
        if isinstance(x, np.ndarray):
            x_tensor = torch.FloatTensor(x)  
        else:
            x_tensor = x
        
        if y is not None:
            if isinstance(y, np.ndarray):
                if len(y.shape) > 1 and y.shape[1] > 1:
                    y = np.argmax(y, axis=1)
                y_tensor = torch.LongTensor(y)  
            else:
                y_tensor = y
            return x_tensor, y_tensor
        
        return x_tensor
    
    def fit(self, x, y=None, validation_data=None, epochs=10,
            verbose=True, batch_size=8, steps_per_epoch=None, validation_steps=None,
            early_stopping_patience=None, early_stopping_monitor='val_loss'):
        """
        Train the model.

        Args:
            early_stopping_patience: Number of epochs with no improvement after which
                training will be stopped. None disables early stopping.
            early_stopping_monitor: Metric to monitor for early stopping.
                One of 'val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_balanced_accuracy'.
        """
        self.model.train()
        history = {
            'loss': [], 'accuracy': [],
            'val_loss': [], 'val_accuracy': [],
            'val_precision': [], 'val_recall': [],
            'val_balanced_accuracy': [],
        }

        # Early stopping state
        es_patience = early_stopping_patience
        es_best = float('inf') if 'loss' in early_stopping_monitor else -float('inf')
        es_counter = 0
        es_improve = (lambda cur, best: cur < best) if 'loss' in early_stopping_monitor \
            else (lambda cur, best: cur > best)
        best_model_state = None
        
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
                
                # FIXED: Add gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
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
                val_loss, val_acc, val_prec, val_rec, val_bal_acc = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                history['val_precision'].append(val_prec)
                history['val_recall'].append(val_rec)
                history['val_balanced_accuracy'].append(val_bal_acc)

                # Update learning rate based on validation loss
                self.scheduler.step(val_loss)

                if verbose:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f'Epoch [{epoch+1}/{epochs}] '
                          f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% '
                          f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}% '
                          f'Val Prec: {val_prec:.4f} Val Rec: {val_rec:.4f} '
                          f'Val Bal Acc: {val_bal_acc:.4f} '
                          f'LR: {current_lr:.6f}')

                # Early stopping
                if es_patience is not None:
                    metric_map = {
                        'val_loss': val_loss, 'val_accuracy': val_acc,
                        'val_precision': val_prec, 'val_recall': val_rec,
                        'val_balanced_accuracy': val_bal_acc,
                    }
                    current_metric = metric_map[early_stopping_monitor]
                    if es_improve(current_metric, es_best):
                        es_best = current_metric
                        es_counter = 0
                        best_model_state = {
                            k: v.cpu().clone() for k, v in self.model.state_dict().items()
                        }
                    else:
                        es_counter += 1
                        if verbose:
                            print(f'  Early stopping counter: {es_counter}/{es_patience}')
                        if es_counter >= es_patience:
                            if verbose:
                                print(f'Early stopping triggered after epoch {epoch+1}.')
                            if best_model_state is not None:
                                self.model.load_state_dict(
                                    {k: v.to(self.device) for k, v in best_model_state.items()}
                                )
                            break
            else:
                if verbose:
                    print(f'Epoch [{epoch+1}/{epochs}] '
                          f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%')
        
        self.fitted = True
        return history
    
    def _validate(self, val_loader):
        """Validate the model. Returns loss, accuracy, macro precision, macro recall, balanced accuracy."""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_preds.append(predicted.cpu())
                all_labels.append(batch_y.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        num_classes = self.num_poses
        tp = torch.zeros(num_classes)
        fp = torch.zeros(num_classes)
        fn = torch.zeros(num_classes)

        for c in range(num_classes):
            pred_c = (all_preds == c)
            true_c = (all_labels == c)
            tp[c] = (pred_c & true_c).sum().float()
            fp[c] = (pred_c & ~true_c).sum().float()
            fn[c] = (~pred_c & true_c).sum().float()

        precision_per_class = tp / (tp + fp + 1e-8)
        recall_per_class = tp / (tp + fn + 1e-8)

        macro_precision = precision_per_class.mean().item()
        macro_recall = recall_per_class.mean().item()
        
        # Balanced accuracy is the average of recall per class
        balanced_accuracy = recall_per_class.mean().item()

        val_correct = (all_preds == all_labels).sum().item()
        val_total = all_labels.size(0)
        val_acc = 100 * val_correct / val_total

        self.model.train()
        return val_loss / len(val_loader), val_acc, macro_precision, macro_recall, balanced_accuracy
    
    def predict(self, x):
        """Make predictions"""
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
            'scheduler_state_dict': self.scheduler.state_dict(),
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
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.fitted = True
        print(f"Model loaded from {filepath}")