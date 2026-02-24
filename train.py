import os
from datetime import datetime
import json
from ingestion import create_train_val_dataloaders
from model import VideoModel
from typing import Literal


def save_training_config(config, model_dir):
    """Save training configuration"""
    os.makedirs(model_dir, exist_ok=True)
    config_path = os.path.join(model_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to {config_path}")

def instantiate_model(dataset_path, batch_size, epochs, train_size, sequence_length, target_size, 
                      num_poses, learning_rate, model_dir, max_videos, movenet_variant: Literal['thunder', 'lightning']='thunder') -> VideoModel:
    
    # Create model with dynamic classes
    input_shape = (sequence_length, target_size[0], target_size[1], 3)

    model = VideoModel(
        input_shape=input_shape,
        num_poses=num_poses,
        learning_rate=learning_rate,
        movenet_variant=movenet_variant
    )
    
    config = {
        'dataset_path': dataset_path,
        'batch_size': batch_size,
        'epochs': epochs,
        'train_size': train_size,
        'max_videos': max_videos,
        'sequence_length': sequence_length,
        'target_size': target_size,
        'num_poses': num_poses,
        'input_shape': input_shape,
        'learning_rate': learning_rate,
        'movenet_variant': movenet_variant
    }

    save_training_config(config=config, model_dir=model_dir)

    return model

def save_model(model, dir):
        path = os.path.join(dir, f'{model.name}.keras')
        model.save(path)
        print(f'Saved {model.name} to {path}')

def train_model(dataset_path, 
                batch_size=4, 
                epochs=50,
                train_size=0.8,
                learning_rate=1e-4,
                max_videos=None,
                save_processed=None,
                load_processed=None,
                model_dir="models",
                sequence_length=16,
                movenet_variant: Literal['thunder', 'lightning']='thunder',
                random_state=None       
                ):
    """
    Main training function
    
    Args:
        dataset_path: Path to video dataset
        batch_size: Batch size for training
        epochs: Number of training epochs
        train_size: Proportion of data for training (ignored if use_kfold=True)
        max_videos: Maximum number of videos to load (for testing)
        save_processed: Path to save processed data
        load_processed: Path to load processed data from
        model_dir: Directory to save model and logs
        sequence_length: Number of frames per video sequence
        target_size: Target frame size (height, width)
        use_kfold: Whether to use K-fold cross-validation
        k_folds: Number of folds for K-fold CV
    """
    
    # Create model directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(model_dir, f"model_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Model will be saved to: {model_dir}")
    
    # Load data
    print("Loading and preparing data...")

    variant_to_size = {
        'thunder': (256, 256),
        'lightning': (192, 192)
    }

    target_size = variant_to_size.get(str.lower(movenet_variant), (256, 256))
    
    train_dataset, val_dataset, num_poses, loader = create_train_val_dataloaders(
            dataset_path=dataset_path,
            target_size=target_size,
            batch_size=batch_size,
            train_size=train_size,
            sequence_length=sequence_length,
            max_videos=max_videos,
            save_processed=save_processed,
            load_processed=load_processed,
            random_state= random_state
        )
    
    try:          
        model = instantiate_model(dataset_path=dataset_path, 
                                  batch_size=batch_size,
                                  epochs=epochs,
                                  train_size=train_size,
                                  sequence_length=sequence_length,
                                  movenet_variant=movenet_variant,
                                  num_poses=num_poses,
                                  target_size=target_size,
                                  max_videos=max_videos,
                                  learning_rate=learning_rate,
                                  model_dir=model_dir)
        
        training_result = model.fit(
                    x=train_dataset,
                    validation_data=val_dataset,
                    epochs=epochs,
                    verbose=True,
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

def main():
    """Main function for direct usage"""
    print("Running with default parameters...")
    
    model, results, model_dir = train_model(
        dataset_path="dataset",
        batch_size=16,
        epochs=15,
        train_size=0.7,
        sequence_length=16,
        learning_rate=1e-3,
        max_videos=None,  
        load_processed="processed_data",
        save_processed="processed_data",
        model_dir="Thunder",
        movenet_variant='thunder',
        random_state=42
    )
        
    if model_dir:
        print("Training completed successfully!")
        return model_dir
    else:
        print("Training failed!")
        return None

if __name__ == "__main__":
    model_dir = main()