import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback, CSVLogger

import segmentation_models as sm
from segmentation_models.metrics import iou_score


class DisplaySegmentationResults(Callback):

    def __init__(self, dataset):
        super(DisplaySegmentationResults, self).__init__()
        self.dataset = dataset

    def display(self, display_list, epoch=None, iou=None, size=(8,3)):
        # expects every element of the display list shape to be (B, H, W, C)
        plt.figure(figsize=size)

        title = ['Input Image', 'Ground Truth Mask', 'Predicted Mask']
        
        if iou is not None:
            title = ['Input Image', 'Ground Truth Mask', f'Pred. Mask iou: {iou:.2f}']
            
        for i in range(len(display_list)):
            plt.subplot(1, len(display_list), i+1)
            plt.title(title[i])
            plt.imshow(tf.keras.utils.array_to_img(display_list[i][0]))
            plt.axis('off')

        if epoch is not None:
            plt.suptitle(f"Epoch {epoch+1}")
        plt.show()
    
    def on_epoch_end(self, epoch, logs=None):
        # Choosing small buffer_size to see same couple of validation images.
        for i, (image, mask) in enumerate(self.dataset.shuffle(10).take(1)):
            # Predict the mask for the image (first image of the batch)
            predicted_mask = self.model.predict(image[0][np.newaxis, ...], verbose=0)[0]
            mask = tf.cast(mask, tf.float32)
            iou = iou_score(mask, predicted_mask) 
            # Reshaping to (1, H, W, C) since display function requires this shape.
            predicted_mask = predicted_mask[np.newaxis, ...]
            self.display([image, mask, predicted_mask], epoch=epoch, iou=iou, size=(6,2))


class ModelPipeline:

    def __init__(self, 
                 input_shape, 
                 backbone, 
                 activation, 
                 epochs,
                 loss, 
                 lr, 
                 metrics, 
                 train_ds_size,
                 val_ds_size, 
                 batch_size, 
                 optimizer,
                 encoder_freeze,
                 model_architecture="Unet"):
        
        self.input_shape = input_shape
        self.backbone = backbone
        self.activation = activation
        self.epochs = epochs
        self.loss = loss
        self.lr = lr
        self.metrics = metrics
        self.batch_size = batch_size
        self.steps_per_epoch = int(train_ds_size // batch_size)
        self.val_steps = int(val_ds_size // batch_size)
        self.optimizer = optimizer
        self.model_architecture = model_architecture
        self.encoder_freeze = encoder_freeze
        self.model_name = f"{model_architecture}_{input_shape}_{backbone}_{self.epochs}"
        
        
    def create_callbacks(self, val_dataset, metric_to_track="val_loss"):
        """
        Creates a list of callbacks to use during model training.
        """
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(filepath=f'model_checkpoints/{self.model_name}@epoch{self.epochs}_best.h5', 
                                               monitor=metric_to_track, save_best_only=True),
            tf.keras.callbacks.CSVLogger(f"saved_models2/{self.model_name}@epoch{self.epochs}.csv"),
            tf.keras.callbacks.ReduceLROnPlateau(monitor=metric_to_track, factor=0.4, patience=4, verbose=1, min_lr=1e-7),
            DisplaySegmentationResults(val_dataset)
        ]
        
        self.callbacks=callbacks
    
    def create_model(self):
        """
        Creates a model using the segmentation models library.
        """
        model_creators = {
            "Unet": sm.Unet,
            "FPN": sm.FPN,
            "Linknet": sm.Linknet,
            "PSPNet": sm.PSPNet
        }
        
        model_creator = model_creators.get(self.model_architecture)
        
        if model_creator is None:
            raise ValueError(f"Invalid model architecture: {self.model_architecture}")
        
        model = model_creator(self.backbone, classes=1, activation=self.activation, 
                             input_shape=self.input_shape, encoder_weights='imagenet',
                             encoder_freeze=self.encoder_freeze)
 
 
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

        return model
        
    def train_model(self, model, train_ds, val_ds):
        """
        Trains the model using the provided datasets.
        """
        results = model.fit(train_ds, epochs=self.epochs, verbose=2,
                            steps_per_epoch=self.steps_per_epoch,
                            validation_steps=self.val_steps,
                            validation_data=val_ds,
                            callbacks=self.callbacks)
        return results
    
    def save_model(self, model):
        model.save(f"saved_models/{self.model_name}.h5")
    
    @staticmethod
    def dice_coef(y_true, y_pred):
        smooth = 1
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
    def bce_dice_loss(self, y_true, y_pred):
        y_true = K.cast(y_true, dtype=y_pred.dtype)
        bce = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        dice = self.dice_coef(y_true, y_pred)
        return bce + 1 - dice
    
    def bce_lndice_loss(self, y_true, y_pred):
        y_true = K.cast(y_true, dtype=y_pred.dtype)
        bce = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
        dice = self.dice_coef(y_true, y_pred)
        return bce - K.log(dice)
    
    def dice_loss(self, y_true, y_pred):
        y_true = K.cast(y_true, dtype=y_pred.dtype)
        return 1 - self.dice_coef(y_true, y_pred)