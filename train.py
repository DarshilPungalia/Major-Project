import os
from datetime import datetime
import json
from ingestion import create_train_val_dataloaders, create_kfold_dataloaders
from model import VideoModel
from typing import Literal
import numpy as np
import gc
from torch.cuda import empty_cache

def save_training_config(config, model_dir):
    """Save training configuration"""
    os.makedirs(model_dir, exist_ok=True)
    config_path = os.path.join(model_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to {config_path}")

def instantiate_model(dataset_path, batch_size, epochs, train_size, sequence_length, 
                      num_poses, learning_rate, model_dir, max_videos, pool_frames,
                      movenet_variant: Literal['thunder', 'lightning']='thunder',
                      n_splits=None) -> VideoModel:
    
    input_shape = (3, sequence_length) if not pool_frames else (3,)
    num_joints = 17  

    model = VideoModel(
        input_shape=input_shape,
        num_poses=num_poses,
        num_joints=num_joints,
        learning_rate=learning_rate,
    )
    
    config = {
        'dataset_path': dataset_path,
        'batch_size': batch_size,
        'epochs': epochs,
        'train_size': train_size,
        'max_videos': max_videos,
        'sequence_length': sequence_length,
        'num_poses': num_poses,
        'input_shape': input_shape,
        'learning_rate': learning_rate,
        'movenet_variant': movenet_variant,
        'n_splits': n_splits,
        'use_kfold': n_splits is not None
    }

    save_training_config(config=config, model_dir=model_dir)

    return model

def save_model(model: VideoModel, dir, fold=None):
    if fold is not None:
        path = os.path.join(dir, f'{model.model.__class__.__name__}_fold{fold}.pt')
    else:
        path = os.path.join(dir, f'{model.model.__class__.__name__}.pt')
    
    model.save(path)
    print(f'Saved {model.model.__class__.__name__} to {path}')

def save_fold_results(fold_results, model_dir):
    """Save k-fold results to JSON"""
    results_path = os.path.join(model_dir, 'kfold_results.json')
    with open(results_path, 'w') as f:
        json.dump(fold_results, f, indent=2)
    print(f"K-fold results saved to {results_path}")

def train_model(dataset_path, 
                batch_size=4, 
                epochs=50,
                train_size=0.8,
                learning_rate=1e-4,
                max_videos=None,
                save_processed=None,
                load_processed=None,
                pool_frames=False,
                model_dir="models",
                sequence_length=16,
                movenet_variant: Literal['thunder', 'lightning']='thunder',
                random_state=None,
                use_kfold=False,
                n_splits=5,
                save_all_folds=False
                ):
    """
    Main training function with optional k-fold cross-validation
    
    Args:
        dataset_path: Path to video dataset
        batch_size: Batch size for training
        epochs: Number of training epochs
        train_size: Proportion of data for training (ignored if use_kfold=True)
        learning_rate: Learning rate for optimizer
        max_videos: Maximum number of videos to load (for testing)
        save_processed: Path to save processed data
        load_processed: Path to load processed data from
        pool_frames: Whether to pool frames temporally
        model_dir: Directory to save model and logs
        sequence_length: Number of frames per video sequence
        movenet_variant: Which MoveNet model to use ('thunder' or 'lightning')
        random_state: Random seed for reproducibility
        use_kfold: Whether to use k-fold cross-validation
        n_splits: Number of folds for k-fold CV
        save_all_folds: Whether to save models from all folds (default: only best)
    """
    
    # Create model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if use_kfold:
        model_dir = os.path.join(model_dir, f"kfold_{n_splits}_{timestamp}")
    else:
        model_dir = os.path.join(model_dir, f"model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Model will be saved to: {model_dir}")
    
    print("Loading and preparing data...")
    
    if use_kfold:
        return train_with_kfold(
            dataset_path=dataset_path,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            max_videos=max_videos,
            save_processed=save_processed,
            load_processed=load_processed,
            pool_frames=pool_frames,
            model_dir=model_dir,
            sequence_length=sequence_length,
            movenet_variant=movenet_variant,
            random_state=random_state,
            n_splits=n_splits,
            save_all_folds=save_all_folds
        )
    else:
        return train_single_split(
            dataset_path=dataset_path,
            batch_size=batch_size,
            epochs=epochs,
            train_size=train_size,
            learning_rate=learning_rate,
            max_videos=max_videos,
            save_processed=save_processed,
            load_processed=load_processed,
            pool_frames=pool_frames,
            model_dir=model_dir,
            sequence_length=sequence_length,
            movenet_variant=movenet_variant,
            random_state=random_state
        )

def train_single_split(dataset_path, batch_size, epochs, train_size, learning_rate,
                      max_videos, save_processed, load_processed, pool_frames, 
                      model_dir, sequence_length, movenet_variant, random_state):
    """Train with single train/val split"""
    
    train_dataset, val_dataset, num_poses, loader = create_train_val_dataloaders(
        dataset_path=dataset_path,
        movenet_variant=movenet_variant,
        batch_size=batch_size,
        train_size=train_size,
        sequence_length=sequence_length,
        max_videos=max_videos,
        save_processed=save_processed,
        load_processed=load_processed,
        random_state=random_state,
        pool_frames=pool_frames,
        output_format='pytorch'
    )
    
    try:          
        model = instantiate_model(
            dataset_path=dataset_path, 
            batch_size=batch_size,
            epochs=epochs,
            train_size=train_size,
            sequence_length=sequence_length,
            movenet_variant=movenet_variant,
            num_poses=num_poses,
            max_videos=max_videos,
            learning_rate=learning_rate,
            model_dir=model_dir,
            pool_frames=pool_frames
        )

        training_result = model.fit(
            x=train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            verbose=True,
            early_stopping_patience=6,
            early_stopping_monitor='val_loss'
        )
        
        final_results = training_result    
        save_model(model, model_dir)
                
        print("\n" + "="*50)
        print("TRAINING COMPLETED")
        print("="*50)
            
        if final_results and isinstance(final_results, dict):
            print("Final training results:")
            for key, value in final_results.items():
                if isinstance(value, list) and len(value) > 0:
                    if isinstance(value[-1], (int, float)):
                        print(f"{key}: {value[-1]:.4f}")
                elif isinstance(value, (int, float)):
                    print(f"{key}: {value:.4f}")
            
        print(f"\nModel and logs saved to: {model_dir}")
            
        return model, final_results, model_dir
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None, model_dir

def train_with_kfold(dataset_path, batch_size, epochs, learning_rate,
                    max_videos, save_processed, load_processed, pool_frames, 
                    model_dir, sequence_length, movenet_variant, random_state,
                    n_splits, save_all_folds):
    """Train with k-fold cross-validation"""
    
    print(f"\n{'='*60}")
    print(f"K-FOLD CROSS-VALIDATION WITH {n_splits} FOLDS")
    print(f"{'='*60}\n")
    
    fold_dataloaders, num_poses, loader = create_kfold_dataloaders(
        dataset_path=dataset_path,
        movenet_variant=movenet_variant,
        batch_size=batch_size,
        n_splits=n_splits,
        sequence_length=sequence_length,
        max_videos=max_videos,
        save_processed=save_processed,
        load_processed=load_processed,
        random_state=random_state,
        pool_frames=pool_frames,
        output_format='pytorch'
    )
    
    fold_results = {
        'folds': [],
        'summary': {}
    }
    
    best_fold = None
    best_val_loss = float('inf')
    
    try:
        for fold_idx, fold_data in enumerate(fold_dataloaders):
            print(f"\n{'='*60}")
            print(f"TRAINING FOLD {fold_idx + 1}/{n_splits}")
            print(f"{'='*60}\n")
            
            model = instantiate_model(
                dataset_path=dataset_path,
                batch_size=batch_size,
                epochs=epochs,
                train_size=None,  
                sequence_length=sequence_length,
                movenet_variant=movenet_variant,
                num_poses=num_poses,
                max_videos=max_videos,
                learning_rate=learning_rate,
                model_dir=model_dir,
                pool_frames=pool_frames,
                n_splits=n_splits
            )
            
            # Train on this fold
            history = model.fit(
                x=fold_data['train'],
                validation_data=fold_data['val'],
                epochs=epochs,
                verbose=True,
                early_stopping_patience=6,
                early_stopping_monitor='val_loss'
            )
            
            # Extract final metrics
            fold_result = {
                'fold': fold_idx + 1,
                'final_train_loss': history['loss'][-1] if history['loss'] else None,
                'final_train_acc': history['accuracy'][-1] if history['accuracy'] else None,
                'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
                'final_val_acc': history['val_accuracy'][-1] if history['val_accuracy'] else None,
                'final_val_precision': history['val_precision'][-1] if history['val_precision'] else None,
                'final_val_recall': history['val_recall'][-1] if history['val_recall'] else None,
                'best_val_loss': min(history['val_loss']) if history['val_loss'] else None,
                'best_val_acc': max(history['val_accuracy']) if history['val_accuracy'] else None,
            }
            
            fold_results['folds'].append(fold_result)
            
            # Track best fold
            if fold_result['final_val_loss'] and fold_result['final_val_loss'] < best_val_loss:
                best_val_loss = fold_result['final_val_loss']
                best_fold = fold_idx
            
            # Save model if requested or if it's the best so far
            if save_all_folds or fold_idx == best_fold:
                save_model(model, model_dir, fold=fold_idx + 1)
            
            print(f"\nFold {fold_idx + 1} Results:")
            print(f"  Val Loss: {fold_result['final_val_loss']:.4f}")
            print(f"  Val Accuracy: {fold_result['final_val_acc']:.2f}%")
            print(f"  Val Precision: {fold_result['final_val_precision']:.4f}")
            print(f"  Val Recall: {fold_result['final_val_recall']:.4f}")

            del model
            del fold_data
            del history
            gc.collect()
            empty_cache()

            import time
            time.sleep(5)
            
        
        # Calculate summary statistics
        val_losses = [f['final_val_loss'] for f in fold_results['folds'] if f['final_val_loss']]
        val_accs = [f['final_val_acc'] for f in fold_results['folds'] if f['final_val_acc']]
        val_precs = [f['final_val_precision'] for f in fold_results['folds'] if f['final_val_precision']]
        val_recs = [f['final_val_recall'] for f in fold_results['folds'] if f['final_val_recall']]
        
        fold_results['summary'] = {
            'mean_val_loss': np.mean(val_losses),
            'std_val_loss': np.std(val_losses),
            'mean_val_accuracy': np.mean(val_accs),
            'std_val_accuracy': np.std(val_accs),
            'mean_val_precision': np.mean(val_precs),
            'std_val_precision': np.std(val_precs),
            'mean_val_recall': np.mean(val_recs),
            'std_val_recall': np.std(val_recs),
            'best_fold': best_fold + 1,
            'best_val_loss': best_val_loss,
        }
        
        # Save results
        save_fold_results(fold_results, model_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("K-FOLD CROSS-VALIDATION COMPLETED")
        print("="*60)
        print(f"\nSummary Statistics Across {n_splits} Folds:")
        print(f"  Mean Val Loss: {fold_results['summary']['mean_val_loss']:.4f} ± {fold_results['summary']['std_val_loss']:.4f}")
        print(f"  Mean Val Accuracy: {fold_results['summary']['mean_val_accuracy']:.2f}% ± {fold_results['summary']['std_val_accuracy']:.2f}%")
        print(f"  Mean Val Precision: {fold_results['summary']['mean_val_precision']:.4f} ± {fold_results['summary']['std_val_precision']:.4f}")
        print(f"  Mean Val Recall: {fold_results['summary']['mean_val_recall']:.4f} ± {fold_results['summary']['std_val_recall']:.4f}")
        print(f"\nBest Fold: {fold_results['summary']['best_fold']} (Val Loss: {fold_results['summary']['best_val_loss']:.4f})")
        print(f"\nResults saved to: {model_dir}")
        
        return model, fold_results, model_dir
        
    except Exception as e:
        print(f"Error during k-fold training: {e}")
        import traceback
        traceback.print_exc()
        return None, None, model_dir

def main():
    """Main function for direct usage"""
    ''' print("Running with default parameters...")
    
    # Example 1: Standard single split training
    print("\n" + "="*60)
    print("EXAMPLE 1: Single Train/Val Split")
    print("="*60)
    
    model, results, model_dir = train_model(
        dataset_path="dataset",
        epochs=50,
        batch_size=16,
        train_size=0.8,
        sequence_length=256,
        learning_rate=1e-4,
        max_videos=None,
        load_processed="thunder_data",  
        save_processed="thunder_data",
        model_dir="Thunder",
        movenet_variant='thunder',
        random_state=42,
        pool_frames=False,
        use_kfold=False  # Single split
    )'''
    
    # Example 2: K-fold cross-validation
    # Uncomment to use k-fold instead:
    
    print("\n" + "="*60)
    print("EXAMPLE 2: K-Fold Cross-Validation")
    print("="*60)
    
    model, results, model_dir = train_model(
        dataset_path="dataset",
        epochs=50,
        batch_size=16,
        sequence_length=256,
        learning_rate=1e-4,
        max_videos=None,
        load_processed="thunder_data",
        save_processed="thunder_data",
        model_dir="Thunder",
        movenet_variant='thunder',
        random_state=42,
        pool_frames=False,
        use_kfold=True,  
        n_splits=5,  
        save_all_folds=False  
    )
    
        
    if model_dir:
        print("\nTraining completed successfully!")
        return model_dir
    else:
        print("\nTraining failed!")
        return None

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    model_dir = main()