import tensorflow as tf
import tensorflow_hub as hub
from typing import Tuple


class MoveNetLayer(tf.keras.layers.Layer):
    """
    Wraps MoveNet (Thunder or Lightning) from TF Hub as a Keras layer.
    MoveNet expects uint8 input of shape (1, H, W, 3) and returns
    keypoints of shape (1, 1, 17, 3) — [y, x, confidence] per keypoint.
    We flatten to (51,) per frame.
    """
    MODELS = {
        'thunder': 'https://tfhub.dev/google/movenet/singlepose/thunder/4',
        'lightning': 'https://tfhub.dev/google/movenet/singlepose/lightning/4',
    }

    def __init__(self, variant='thunder', **kwargs):
        super().__init__(**kwargs)
        if variant not in self.MODELS:
            raise ValueError(f"variant must be one of {list(self.MODELS.keys())}")
        self.variant = variant
        self.movenet = hub.load(self.MODELS[variant]).signatures['serving_default']

    def call(self, frame):
        """
        Args:
            frame: float32 tensor of shape (H, W, 3), values in [0, 1]
        Returns:
            keypoints: float32 tensor of shape (51,)
        """
        frame_uint8 = tf.cast(frame * 255.0, tf.int32)
        frame_uint8 = tf.expand_dims(frame_uint8, axis=0)           # (1, H, W, 3)

        outputs = self.movenet(input=frame_uint8)
        keypoints = outputs['output_0']                              # (1, 1, 17, 3)
        keypoints = tf.reshape(keypoints, [-1])                      
        return keypoints

    def get_config(self):
        config = super().get_config()
        config.update({'variant': self.variant})
        return config


class VideoModel:
    def __init__(self, num_poses,
                 input_shape=(16, 192, 192, 3),
                 learning_rate=1e-4,
                 movenet_variant='thunder'):
        """
        Args:
            input_shape: (sequence_length, H, W, 3).
                         MoveNet Thunder expects 256×256, Lightning expects 192×192.
                         Pass the appropriate H/W for your chosen variant.
            movenet_variant: 'thunder' (more accurate) or 'lightning' (faster).
            use_lstm: If True, an LSTM models temporal patterns over keypoints.
                      If False, keypoints are simply flattened and pooled.
        """
        self.input_shape = input_shape
        self.num_poses = num_poses
        self.learning_rate = learning_rate
        self.fitted = False
        self.movenet_variant = movenet_variant

        self.model = self.build_model()

    def build_model(self):
        sequence_length = self.input_shape[0]   
        NUM_KEYPOINTS = 17
        KEYPOINT_DIM = 3                         # y, x, confidence
        keypoint_features = NUM_KEYPOINTS * KEYPOINT_DIM  

        inputs = tf.keras.layers.Input(shape=self.input_shape, name='video_input')
        movenet_layer = MoveNetLayer(variant=self.movenet_variant, name='movenet')

        keypoints_seq = tf.keras.layers.Lambda(
            lambda x: tf.map_fn(
                lambda frame_seq: tf.map_fn(
                    movenet_layer,
                    frame_seq,
                    fn_output_signature=tf.float32
                ),
                x,
                fn_output_signature=tf.float32
            ),
            name='keypoint_extraction'
        )(inputs)

        x = tf.keras.layers.LSTM(
            128, return_sequences=False
        )(keypoints_seq)
       
        x = tf.keras.layers.Dropout(0.35)(x)
        x = tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-2))(x)

        est = tf.keras.layers.Dropout(0.5)(x)
        pose_estimation = tf.keras.layers.Dense(
            self.num_poses, activation='softmax'
        )(est)

        estimation_model = tf.keras.Model(
            inputs= inputs, outputs=pose_estimation, name='estimation'
        )

        return estimation_model

    def compile_models(self) -> tf.keras.Model:
        """Compile the model"""
        model= self.model

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(
            optimizer=self.optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )


        return model

    def fit(self, x, y=None, validation_data=None, epochs=10, verbose=True,
            batch_size=8, steps_per_epoch=None, validation_steps=None):
        pass

    def predict(self, x):
        if not self.fitted:
            raise ValueError('Call .fit() first')