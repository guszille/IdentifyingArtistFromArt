import os.path
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import models, layers

def get_generators_from_dir(params, rescale=False, enable_augmentation=False):
    default_args = {'validation_split': 0.2}
    
    params['shuffle'] = True
    params['seed'] = 123
    
    if rescale:
        default_args['rescale'] = 1 / 255
    
    if enable_augmentation:
        default_args['rotation_range'] = 45
        default_args['width_shift_range'] = 0.2
        default_args['height_shift_range'] = 0.2
        default_args['shear_range'] = 0.2
        default_args['zoom_range'] = 0.2
        default_args['horizontal_flip'] = True
        default_args['vertical_flip'] = True
        
    img_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(**default_args)
    
    train_gen = img_data_gen.flow_from_directory(**params, subset='training')
    validation_gen = img_data_gen.flow_from_directory(**params, subset='validation')
    
    return train_gen, validation_gen

def get_dataset_from_dir(params):
    autotune = tf.data.AUTOTUNE
    
    params['validation_split'] = 0.2
    params['seed'] = 123

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(**params, subset='training')
    validation_ds = tf.keras.preprocessing.image_dataset_from_directory(**params, subset='validation')

    n_batches = tf.data.experimental.cardinality(validation_ds)

    test_ds = validation_ds.take(n_batches // 5) # Creating a test dataset from the validation dataset.
    validation_ds = validation_ds.skip(n_batches // 5)
    
    # Configure the dataset for performance.
    #
    # https://www.tensorflow.org/tutorials/load_data/images#configure_the_dataset_for_performance
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    validation_ds = validation_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.prefetch(buffer_size=autotune)

    return train_ds, validation_ds, test_ds

def show_gen_samples(gen, class_names):
    # Visualize the data: here are the first 9 images from the generator.

    plt.figure(figsize=(10, 10))
    
    mat_x, arr_y = gen.next()

    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)

        plt.imshow(mat_x[i])
        plt.title(class_names[int(arr_y[i])])
        plt.axis('off')
    
def show_dataset_samples(ds):
    # Visualize the data: here are the first 9 images from the dataset.

    plt.figure(figsize=(10, 10))

    class_names = ds.class_names

    for images, labels in ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)

            plt.imshow(images[i].numpy().astype('uint8'))
            plt.title(class_names[int(labels[i])])
            plt.axis('off')
    
def build_and_compile_binary_model(optimizer=None, enable_augmentation=False, rescale=False, input_shape=(256, 256, 3)):
    model = models.Sequential()

    if enable_augmentation:
        augmentation_layers = []

        augmentation_layers.append(layers.experimental.preprocessing.RandomFlip('horizontal', input_shape=input_shape))
        augmentation_layers.append(layers.experimental.preprocessing.RandomRotation(0.1))
        augmentation_layers.append(layers.experimental.preprocessing.RandomZoom(0.1))

        model.add(models.Sequential(augmentation_layers))

    if rescale:
        model.add(layers.experimental.preprocessing.Rescaling(1./255, input_shape=input_shape))

    model.add(layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(32, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, 3, padding='same', activation='relu'))
    model.add(layers.MaxPooling2D())

    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(  1, activation='sigmoid'))

    if not optimizer:
        optimizer = tf.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=optimizer, loss=tf.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

    return model

def run_model(model, train_data, validation_data, epochs):
    history = model.fit(train_data, validation_data=validation_data, epochs=epochs)
    
    return history

def show_model_process_results(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(10)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()
    
def evaluate_and_predict(test_data):
    image_batch, label_batch = test_data.next()
    predictions = simple_binary_model.predict_generator(image_batch)
    predictions = tf.where(predictions < 0.5, 0, 1)
    predictions = predictions.numpy().flatten()

    plt.figure(figsize=(10, 10))
    
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        
        plt.imshow(image_batch[i])
        plt.title("{} ({})".format(useful_artists[predictions[i]], predictions[i] == label_batch[i]))
        plt.axis("off")
