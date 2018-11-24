from keras import Sequential
from keras.preprocessing import image
from keras.layers import Dense, Conv2D, Flatten, Activation, Reshape, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.losses import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

def load_image(path):
    img = image.load_img(path)
    return image.img_to_array(img) / 255

def create_net(lr):
    kernel = (4, 4)
    filt = 64
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(256, )))
    model.add(Reshape((16, 16, 1)))
    model.add(Conv2D(filt, kernel, padding='same', activation='relu'))
    model.add(UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest'))
    model.add(Conv2D(filt, kernel, padding='same', activation='relu'))
    model.add(UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest'))
    model.add(Conv2D(filt, kernel, padding='same', activation='relu'))
    model.add(UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest'))
    model.add(Conv2D(filt, kernel, padding='same', activation='relu'))
    model.add(Conv2D(3, kernel, padding='same', activation='sigmoid'))

    model.compile(Adam(lr), loss=mean_squared_error, metrics=['accuracy'])
    model.summary()
    return model

def train(lr):
    test_count = 1000
    print('\n\n\n\nTRAINING NEW MODEL, LR: {}'.format(lr))
    img = load_image('./pika.jpeg')
    model = create_net(lr)

    X = np.random.rand(test_count, 256)
    y = np.array([img] * test_count)

    tensorboard = TensorBoard(log_dir='./Graph/{}x{}'.format(128, 128), histogram_freq=0, write_graph=True, write_images=True)

    hist = model.fit(X, y, batch_size=5, epochs=2, validation_split=0.1, verbose=1, callbacks=[tensorboard])

    # plt.imshow(img)
    # plt.show()

    prediction = model.predict(np.reshape(X[1], (1, 256)))
    plt.imshow(prediction[0])
    plt.savefig('output.png')

if __name__ == '__main__':
    train(0.001)
