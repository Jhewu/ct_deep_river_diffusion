""" 
THIS DDIM MODEL
TRAINS OUR 100X300 IMAGES

ONLY TENSORFLOW 2.16.1 WILL 
WORK, PLEASE REFER TO THE
OFFICIAL KERAS WEBSITE
https://keras.io/getting_started/
"""

""" PARAMETERS TO CHANGE BELOW """
image_size = (96, 296)           # (height, width)
img_folder_name = "flow_300_100" # must be in cwd

seed = 42
validation_split = 0.2
batch_size = 12
dataset_repetitions = 5
kid_image_size = 128      
plot_diffusion_steps = 20 # 100

# for the time being I will add padding
# to the images, I will modify it in the 
# future
pad_to_aspect_ratio = False
crop_to_aspect_ratio = True     # just in case



""" ALL IMPORTS """
# import necessary libraries
import warnings
import os
import matplotlib.pyplot as plt 
import tensorflow as tf
import keras

# import from local scripts
from diffusion_model import DiffusionModel
from callbacks import * 





embedding_dims = 128


# The widths of each subsequent downsample blocks
#widths = [32, 64, 96, 128, 256] # there will be 5 downsample blocks (including the bottleneck) 
widths = [32, 64, 96, 128] 
#widths = [150, 300, 450, 600]
#widths = [128, 256, 384, 512]



# number of diffusion steps for calculating KID
kid_diffusion_steps = 5

# sampling
min_signal_rate = 0.1
max_signal_rate = 0.5

# exponential moving average weight used during evaluation
ema = 0.999
# optimization
learning_rate = 1e-3
weight_decay = 1e-4





""" HELPER FUNCTIONS """

def load_dataset(): 
    cwd = os.getcwd()
    img_dir = os.path.join(cwd, img_folder_name)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        img_dir, 
        validation_split = validation_split,
        subset="training", 
        seed = seed,
        image_size = (image_size[0], image_size[1]),  
        batch_size = None,
        shuffle = True,
        crop_to_aspect_ratio = crop_to_aspect_ratio,
        pad_to_aspect_ratio = pad_to_aspect_ratio,
        #labels = labels, 
        #label_mode = None
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        img_dir, 
        validation_split = validation_split,
        subset="validation", 
        seed = seed,
        image_size = (image_size[0], image_size[1]), 
        batch_size = None,
        shuffle = True,
        crop_to_aspect_ratio = crop_to_aspect_ratio,
        pad_to_aspect_ratio = pad_to_aspect_ratio,
        #labels = labels, 
        #label_mode = None
    )
    return train_ds, val_ds

def prepare_dataset(train_ds, val_ds): 
    train_ds = (train_ds
        .map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE) # each dataset has the structure
        .cache()                                                   # (image, labels) when inputting to 
        .repeat(dataset_repetitions)                               # map
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE))
    val_ds = (val_ds
        .map(normalize_image, num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .repeat(dataset_repetitions)
        .shuffle(10 * batch_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(buffer_size=tf.data.AUTOTUNE)) # THIS IS A PREFETCH DATASET
    return train_ds, val_ds

def normalize_image(images, _):    
    # clip pixel values to the range [0, 1]
    return tf.clip_by_value(images / 255, 0.0, 1.0)

def plot_images(dataset, num_images=5):
    # Create an iterator to get images from the dataset
    iterator = iter(dataset)
    
    # Get the first batch of images
    images = next(iterator)

    # Plot the images
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        axes[i].imshow(images[i])
        axes[i].axis("off")
    plt.show()

def plot_history(history, dict_key): 
    plt.plot(history.history[dict_key]) 
    plt.title(dict_key) 
    plt.ylabel(dict_key)
    plt.xlabel("epoch")
    plt.legend(["train"], loc="upper left")
    plt.show()

""" Main Runtime """

def TrainDiffusionModel():
    """
    THIS SCRIPT IS USED TO TRAIN THE DIFFUSION MODEL
    """

    # load and prepare the dataset
    train_dataset, val_dataset = load_dataset()
    train_dataset, val_dataset = prepare_dataset(train_dataset, val_dataset)

    # Plot a sample of images from the train dataset
    plot_images(train_dataset)

    # # create and compile the model
    # model = DiffusionModel(image_size, widths, block_depth)
    #     # below tensorflow 2.9:
    #     # pip install tensorflow_addons
    #     # import tensorflow_addons as tfa
    #     # optimizer=tfa.optimizers.AdamW

    # model.compile(
    #     optimizer=keras.optimizers.AdamW(
    #         learning_rate=learning_rate, weight_decay=weight_decay
    #     ),
    #     loss=keras.losses.mean_absolute_error,
    #     # optimizer: AdamW with a specified learning rate (learning_rate) and weight decay (weight_decay).
    #     # Loss function: Pixelwise mean absolute error (MAE).
    # )

    # # calculate mean and variance of training dataset for normalization
    # model.normalizer.adapt(train_dataset)
    #     # The adapt method is called on the normalizer using the training dataset.
    #     # This calculates the mean and variance of the training dataset for normalization.

    # """
    # Training: 
    # - train for at least 50 epochs for good results
    # - run training and plot generated images 

    # WHEN YOU WANT TO PERFORM INFERENCE BASED ON PREVIOUS
    # WEIGHTS, COMMENT THIS BLOCK OUT

    # """

    # num_epochs = 1
    # history = model.fit(
    #     train_dataset,
    #     epochs=num_epochs,
    #     validation_data=val_dataset,
    #     callbacks=[
    #         early_stop_callback,
    #         csv_callback, 
    #         plot_image_callback,
    #         checkpoint_callback, # checkpoint callback located here
    #     ],
    # )


def Load_Diffusion(): 
    """
    Loading the saved model
    """
    train_dataset, val_dataset = load_dataset()
    train_dataset, val_dataset = prepare_dataset(train_dataset, val_dataset)

    # Plot a sample of images from the train dataset
    plot_images(train_dataset)
    model = DiffusionModel(image_size, widths, block_depth)
    model.normalizer.adapt(train_dataset) # need this in order for it not to produce any error
    model.load_weights(checkpoint_path)
    model.plot_images()


    model.save(f'{folder_path}/my_model.keras')


    """
    SOMETIMES AFTER LOADING THE WEIGHTS 
    ONTO THE MODEL, SOMETIMES IT DOES NOT 
    GENERATE FLOWERS (PROBABLY OVERFITTING), 
    AND TRAINING IT FOR A COUPLE GENERATION 
    HELPS WITH RECOVERING BEHAVIOR

    num_epochs = 5
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=val_dataset,
        callbacks=[
            keras.callbacks.LambdaCallback(on_epoch_end=model.plot_images), # plot images as well
            checkpoint_callback, # checkpoint callback located here
        ],
    )

    """





    plt.figure(1)
    plot_history("i_loss")

    plt.figure(2)
    plot_history("n_loss")

    plt.figure(3)
    plot_history("val_i_loss")

    plt.figure(4)
    plot_history("val_kid")

    plt.figure(5)
    plot_history("val_n_loss")


if __name__ == "__main__":
    """ 
    ENSURE WE ARE USING THE CORRECT VERSION 
    ONLY KERAS 3.6 AND TENSORFLOW 2.16.1 HAS 
    BEEN PROVEN TO WORK WITH THIS SCRIPT
    """
    print(keras.__version__)
    print(tf.__version__)

    warnings.filterwarnings('ignore')
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # disable if JIT Compilation error
    # os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=0' 
    # tf.config.optimizer.set_jit(False)

    TrainDiffusionModel()



    # folder_path = "[INSERT NAME HERE]"
    # # create the folder if it does not exist
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    # print("Hello, World!")





