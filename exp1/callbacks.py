"""
THIS SCRIPT CONTAINS ALL THE CALLBACK FUNCTIONS 
FOR THE DIFFUSION MODEL
"""

""" ALL IMPORTS """
# import necessary libraries
import keras

#import form local script
# from ddim import folder_path

""" 
Create Custom Callback 
This callback only sample and plot
the images after each 10 epochs, 
saving computer resource
"""
class CustomCallback(keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs=None): 
        if (epoch + 1) % 10 == 0: 
            self.model.plot_images()

"""
Checkpoint Callback
Save the best performing models
only
"""
checkpoint_path = f"{folder_path}/checkpoints/diffusion_model.weights.h5"
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor="val_kid",
    mode="min",
    save_best_only=True,)

plot_image_callback = CustomCallback()

"""
Early Stopping Callback
Ensure we are not wasting resources
"""
early_stop_callback = keras.callbacks.EarlyStopping(
    monitor="val_kid", 
    min_delta=1e-3,
    patience=3,
    verbose=1,
    mode="min",
    restore_best_weights=True,
    start_from_epoch=50,
)

"""
CSV Logger Callback
Creates a csv file that logs 
epoch, acc, val_acc, val_loss
"""
csv_callback = keras.callbacks.CSVLogger(
    f"{folder_path}/model_history_1.csv", 
    separator=",", 
    append=False)