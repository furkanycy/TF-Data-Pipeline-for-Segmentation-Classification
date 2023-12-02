import os
from glob import glob
import numpy as np
import tensorflow as tf
import albumentations as A
from functools import partial
import segmentation_models as sm

sm.set_framework('tf.keras')


class DatasetCreator:
    """
    A class used to create a dataset for image segmentation tasks.

    ...

    Attributes
    ----------
    working_dir : str
        The directory where the dataset is located.
    dataset_name : str
        The name of the dataset.
    image_folder : str
        The name of the folder containing the images.
    mask_folder : str
        The name of the folder containing the masks.
    train_split_percentage : float
        The percentage of the dataset to use for training.
    img_height : int
        The height of the images.
    img_width : int
        The width of the images.
    backbone : str
        The name of the backbone to use for the segmentation model.
    buffer_size : int, optional
        The number of elements to prefetch in the dataset (default is 1000).
    batch_size : int, optional
        The number of elements in each batch (default is 8).
    autotune : tf.data.experimental.AUTOTUNE, optional
        The number of elements to prefetch in the dataset (default is tf.data.experimental.AUTOTUNE).
    test : bool, optional
        Whether the dataset is a test dataset (default is False).
    label_type : str, optional
        The type of label to load from the image (default is 'artifact').
    shuffle : bool, optional
        Whether to shuffle the dataset (default is True).

    Methods
    -------
    get_path_array()
        Returns a numpy array containing the paths of the images and masks.

    datasets_from_path_array(path_arr)
        Returns a TensorFlow dataset created from the given path array.

    load_image_function()
        Returns a function that loads an image and its corresponding mask from their paths.

    filter_mask(image, mask)
        Returns a boolean indicating whether the given mask contains any non-zero values.
        This function is used for binary task, for example this filters out the tumors in the dataset
        if the task is for artifacts.

    preprocess(image, mask)
        Returns the preprocessed image and mask for the backbone if segmentation models library is used.

    augment_data(image, mask)
        Returns the augmented image and mask.

    create_dataset()
        Returns the created dataset and its size."""
        
    def __init__(self, 
                 working_dir, # bunlara :type ekle
                 dataset_name, 
                 image_folder, 
                 mask_folder, 
                 train_split_percentage, 
                 img_height, 
                 img_width, 
                 backbone,
                 buffer_size:int=1000, 
                 batch_size:int=8, 
                 autotune=tf.data.AUTOTUNE,
                 test:bool=False,
                 label_type:str='artifact',
                 shuffle:bool=True):
        
        self.working_dir = working_dir
        self.dataset_name = dataset_name
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.train_split_percentage = train_split_percentage
        self.img_height = img_height
        self.img_width = img_width
        self.backbone = backbone
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.autotune = autotune
        self.test = test
        self.label_type = label_type
        self.shuffle = shuffle

    def get_path_array(self):
        """
        Folders containing the dataset should follow this structure:
            /dataset_name
            /test_dataset_name
        """
        image_paths = []
        mask_paths = []
        
        if self.test:
            image_dir = os.path.join(self.working_dir, self.dataset_name, ("test_" + self.image_folder))
            mask_dir = os.path.join(self.working_dir, self.dataset_name, ("test_" + self.mask_folder))
        else:
            image_dir = os.path.join(self.working_dir, self.dataset_name, self.image_folder)
            mask_dir = os.path.join(self.working_dir, self.dataset_name, self.mask_folder)
        
        for file in glob(os.path.join(image_dir, '*.png')):
            image_paths.append(file)
        for file in glob(os.path.join(mask_dir, '*.png')):
            mask_paths.append(file)
            
        image_paths.sort()
        mask_paths.sort()
        path_arr = list(zip(image_paths, mask_paths))
        path_arr = np.array(path_arr)
        
        if self.shuffle:
            np.random.shuffle(path_arr)
        
        return path_arr

    def datasets_from_path_array(self, path_arr):
        """
        test folder contains nearly 15%, 
        train_split:0.82 makes the split 0.70 train 0.15 val 0.15 test 
        """
        
        if self.test:
            test_dataset = tf.data.Dataset.from_tensor_slices(path_arr)
            return test_dataset.map(lambda x: (x[0], x[1]))
        
        total_images = path_arr.shape[0]
        train_split = int(self.train_split_percentage * total_images)
        
        train_paths = path_arr[:train_split]
        val_paths = path_arr[train_split:]
        
        train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
        val_dataset = tf.data.Dataset.from_tensor_slices(val_paths)
        
        # Necessary for the tf data map functions to work
        train_dataset = train_dataset.map(lambda x: (x[0], x[1]))
        val_dataset = val_dataset.map(lambda x: (x[0], x[1]))
        
        return train_dataset, val_dataset


    def load_image_function(self): #############################################
        def load_image(image_path, label_path):
            image = tf.io.read_file(image_path)
            image = tf.image.decode_png(image, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            image = tf.image.resize(image, (self.img_height, self.img_width))

            label_raw = tf.io.read_file(label_path)
            label_raw = tf.image.decode_png(label_raw, channels=3)
            label_raw = tf.image.resize(label_raw, 
                                        (self.img_height, self.img_width), 
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            if self.label_type == 'artifact':
                label = label_raw[:,:,1] # Artifacts are labeled green
            elif self.label_type =='tumor':
                label = label_raw[:,:,0] # Tumors are labeled red
            elif self.label_type == None:
                label = label_raw
                label = tf.image.rgb_to_grayscale(label)
            else:
                raise ValueError(f"Failed to decode label image from '{self.label_type}'. \
                                 Please enter valid label_type")

            label = tf.where(label == 0, 0, 1)
            label = tf.image.convert_image_dtype(label, tf.int32)
            # Add an extra dimension to mask to return (h, w, 1)
            label = tf.expand_dims(label, axis=-1)
            return image, label

        return load_image

    @staticmethod
    def filter_mask(image, mask):
        if tf.reduce_sum(mask) == 0:
            return False
        else:
            return True

    def get_preprocess_function(self):
        preprocess_input = sm.get_preprocessing(self.backbone)

        @staticmethod
        def preprocess_function(image, mask):
            image = preprocess_input(image)
            return image, mask

        return preprocess_function

    @staticmethod
    def augment_func(image, mask):
        transforms = A.Compose([

            # Flips 50%
            A.OneOf([
                A.HorizontalFlip(p=0.4),
                A.NoOp(p=0.6)], p=1.0),
            
            # Apply elastic transformation 20%
            A.OneOf([
                A.ShiftScaleRotate(p=0.5),
                A.ElasticTransform(p=0.5)], p=0.2),
            
            # Apply brightness and gamma 20%
            A.OneOf([
                A.RandomBrightnessContrast(p=0.10), 
                A.RandomGamma(p=0.10),
                A.NoOp(p=0.80)], p=1.0),
            
    #         A.GridDistortion(p=0.1),
    #         A.Transpose(p=0.1),
    #         A.RandomSizedCrop(min_max_height=(, ), height=, width=, p=0.15)
        
        ], additional_targets={'mask': 'mask'})

        data = {"image":image, "mask":mask}
        aug_data = transforms(**data)

        aug_img = aug_data["image"]
        aug_img = tf.cast(aug_img, tf.float32)

        aug_mask = aug_data["mask"]
        # Making sure labels are binary (background or label)
        aug_mask = np.where(aug_mask == 0, 0, 1)
        aug_mask = tf.cast(aug_mask, tf.int32)

        return aug_img, aug_mask


    def augment_data(self, image, mask):
        # To fix "Tensor has no attribute .numpy() method" error tf.numpy_function() is used
        image, mask = tf.numpy_function(func=self.augment_func, inp=[image, mask], Tout=(tf.float32, tf.int32))
        """    
        Datasets loses its shape after applying a tf.numpy_function, so this is 
        necessary for the sequential model and when inheriting from the model class.
        """
        image.set_shape((self.img_height, self.img_width, 3))
        mask.set_shape((self.img_height, self.img_width, 1))

        return image, mask
    

    def create_dataset(self):     
        """
        Returns the datasets and its sizes using methods defined in the class
        """
        load_image = self.load_image_function()
        preprocess_function = self.get_preprocess_function()
        
        arr = self.get_path_array()
        
        if self.test:
            test_ds = self.datasets_from_path_array(arr)
            test_ds = test_ds.map(load_image)
            test_ds_size = len(test_ds) 
            
            return test_ds, test_ds_size
            
        else: 
            train_ds, val_ds = self.datasets_from_path_array(arr)
        
            train_ds = (
                train_ds
                .map(load_image)
                .filter(self.filter_mask)
                .map(preprocess_function)
            )

            # can not measure size of the dataset after filtering
            train_ds_size = train_ds.reduce(0, lambda x,_: x+1).numpy()
            train_ds = (
                train_ds
                .cache()
                # Only appllying augmentation on train dataset
                .map(partial(self.augment_data))
                .shuffle(buffer_size=self.buffer_size)
                .repeat()
                .batch(self.batch_size)
                .prefetch(buffer_size=self.autotune)
            )
            
            val_ds = (
                val_ds
                .map(load_image)
                .filter(self.filter_mask)
                .map(preprocess_function)
            )

            val_ds_size = val_ds.reduce(0, lambda x,_: x+1).numpy()

            val_ds = (
                val_ds
                .cache()
                .batch(self.batch_size)
                .prefetch(buffer_size=self.autotune)
            )
            
            return train_ds, train_ds_size, val_ds, val_ds_size