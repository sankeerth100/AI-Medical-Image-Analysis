import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def load_data():

    train_dir = "data/train"
    val_dir = "data/val"

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_data = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    return train_data, val_data