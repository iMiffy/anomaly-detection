from tensorflow.keras.layers import Conv2D, MaxPool2D, Conv2DTranspose
import tensorflow.keras as keras
import numpy as np
from PIL import Image
import pandas as pd
import argparse 

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image-path", required=True,
	help="path to input dataset")
ap.add_argument("-o", "--model-path", required=True,
	help="path to model")
ap.add_argument("-m", "--mode", required=True,
	help="either train or test")    
   
ap.add_argument("-e", "--epoch", required=False, type=int, default=10,
	help="number of epochs for training the model on the dataset")
ap.add_argument("-b", "--batch-size", required=False, type=int, default=16,
	help="batch size of images to be passed through network")
ap.add_argument("-s", "--valsplit-ratio", required=False, type=float, default=0.2,
	help="train-validation split ratio")
args = vars(ap.parse_args())


mode = args['mode']
if mode == 'train':

    conv_encoder = keras.models.Sequential([
        Conv2D(16, (9,9), padding='same', activation='selu', kernel_initializer='lecun_normal', input_shape=(512,512,3)),
        MaxPool2D(pool_size=2),
        Conv2D(32, (7,7), padding='same', activation='selu', kernel_initializer='lecun_normal'),
        MaxPool2D(pool_size=2),
        Conv2D(64, (7,7), padding='same', activation='selu', kernel_initializer='lecun_normal'),
        MaxPool2D(pool_size=2),
        Conv2D(128, (5,5), padding='same', activation='selu', kernel_initializer='lecun_normal'),
        MaxPool2D(pool_size=2),
        Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal'),
        MaxPool2D(pool_size=2)
    ])

    conv_decoder = keras.models.Sequential([
        Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', activation='selu', kernel_initializer='lecun_normal', 
                        input_shape=(16,16,256)),
        Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', activation='selu', kernel_initializer='lecun_normal'),
        Conv2DTranspose(64, kernel_size=5, strides=2, padding='same', activation='selu', kernel_initializer='lecun_normal'),
        Conv2DTranspose(32, kernel_size=7, strides=2, padding='same', activation='selu', kernel_initializer='lecun_normal'),
        Conv2DTranspose(16, kernel_size=7, strides=2, padding='same', activation='selu', kernel_initializer='lecun_normal'),
        Conv2DTranspose(3, kernel_size=3, strides=2, padding='same', activation='linear')
    ])

    conv_ae = keras.models.Sequential([conv_encoder, conv_decoder])
    conv_ae.compile(optimizer=keras.optimizers.RMSprop(lr=1e-4), loss="mean_squared_error")

    checkpoint_cb = keras.callbacks.ModelCheckpoint(args['model_path'], save_best_only=True)

    img_list = []
    for each_img in os.listdir(args['image_path']):
        pixel_vals = np.asarray(Image.open(each_img))
        img_list.append(pixel_vals)

    good_train = np.asarray(img_list, dtype='float32')
    good_train /= 255.0

    history = conv_ae.fit(good_train, good_train, epochs = args['epoch'], batch_size = args['batch_size'], 
                          validation_split=args['valsplit_ratio'], callbacks=[checkpoint_cb])


elif mode == 'test':
    model_ae = keras.models.load_model(args['model_path'])
    mse_list = []
    label_list = []
    
    
    for each_img in os.listdir(args['image_path']):
        pixel_vals = np.asarray(Image.open(each_img), dtype='float32')
        pixel_vals /= 255.0
        pixel_vals = pixel_vals.reshape((1, pixel_vals.shape[0], pixel_vals.shape[1], pixel_vals.shape[2]))
        output_image = model_ae.predict(pixel_vals)
        mse = np.mean(np.power(pixel_vals - output_image, 2))
        mse_list.append(mse)
        label_list.append(1)
    
    res_df = pd.DataFrame({'Reconstruction_error': mse_list, 'True_class': label_list})
    res_df.reset_index()
    threshold = 0.0035
    
    groups = res_df.groupby('True_class')
    fig, ax = plt.subplots()
    for name, group in groups:
        ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
               label='Defective' if name == 1 else "Good")
    
    ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
    ax.legend()
    plt.title("Reconstruction Error in Dataset")
    plt.ylabel("reconstruction error")
    plt.xlabel("data point index")
    plt.show();

