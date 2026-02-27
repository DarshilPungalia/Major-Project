import tensorflow as tf
from numpy import ndarray

class VideoModel:
    def __init__(self, num_poses,
                 input_shape=(16, 51),
                 learning_rate=1e-4):
        """
        Args:
            num_poses: Number of classes
            input_shape: (sequence_length, keypoints).
        """
        self.input_shape = input_shape
        self.num_poses = num_poses
        self.learning_rate = learning_rate
        self.fitted = False

        self.model = self.build_model()

    def build_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape, name='video_input')


        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        outputs = tf.keras.layers.Dense(self.num_poses, activation='softmax')(x)

        estimation_model = tf.keras.Model(
            inputs= inputs, outputs=outputs, name='estimation'
        )

        return estimation_model

    def _compile_model(self) -> tf.keras.Model:
        """Compile the model"""
        model = self.model

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy', 'precision', 'recall']
        )

        return model

    def fit(self, x: tf.data.Dataset | ndarray, y: tf.data.Dataset | ndarray=None, validation_data: tf.data.Dataset | tuple[ndarray, ndarray]=None, 
            epochs=10, verbose=True, batch_size=8, steps_per_epoch=None, validation_steps=None):
        
        self.model = self._compile_model()
        model = self.model

        if isinstance(x, tf.data.Dataset):
            history = model.fit(x,
                      validation_data=validation_data,
                      epochs=epochs,
                      steps_per_epoch=steps_per_epoch,
                      validation_steps=validation_steps,
                      verbose=verbose)
        
        elif isinstance(x, ndarray):
            if y is None:
                raise ValueError('y is neccessary when x is of type ndarray')

            else:
                if validation_data:
                    if type(validation_data) is not tuple:
                        raise TypeError(f'Expected tuple, got {type(validation_data)}')
                    
                    for data in validation_data:
                        if not isinstance(data, ndarray):
                            raise TypeError(f'Expected ndarray, got {type(data)}')
                        
                history = model.fit(x, y,
                          validation_data=validation_data,
                          epochs=epochs,
                          steps_per_epoch=steps_per_epoch,
                          batch_size=batch_size,
                          validation_steps=validation_steps,
                          verbose=verbose)
        
        else:
            raise TypeError(f'Expected x to be Dataset or ndarray, got {type(x)}')
        
        self.fitted = True
        self.model = model

        return history.history

    def predict(self, x):
        if not self.fitted:
            raise ValueError('Call .fit() first')
        
        else:
            predictions = self.model.predict(x)
        
        return predictions