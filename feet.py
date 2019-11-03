from flask import Flask, render_template
app = Flask(__name__)

@app.route('/')
def index():
  import keras
  from keras.models import Sequential
  from keras.layers import Dense, Dropout, Flatten
  from keras.layers import Conv2D, MaxPooling2D
  from keras.utils import to_categorical
  from keras.preprocessing import image
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  from sklearn.model_selection import train_test_split
  from keras.utils import to_categorical
  from tqdm import tqdm
  import glob
  import os
  import numpy as np
  import logging
  #for filename in glob.glob(os.path.join('C:\\Users\\gentrt\\Documents\\Personal\\untied\\data', "*.jpg")):
  train = pd.read_csv('C:\\Users\\gentrt\\Documents\\Personal\\untied\\data\\trainingset.csv')
  train_image = []
  for i in tqdm(range(train.shape[0])):
    img = image.load_img('C:\\Users\\gentrt\\Documents\\Personal\\untied\\data\\' + train['File Name'][i], target_size=(320,180), grayscale=False)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
    # plt.imshow(img)
  X = np.array(train_image)
  y=train['Classification'].values
  y = to_categorical(y)
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
  from keras.layers.normalization import BatchNormalization

  model = Sequential()
  model.add(Conv2D(16, kernel_size = (3, 3), activation='relu', input_shape=(320, 180, 3)))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(BatchNormalization())
  model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  # model.add(BatchNormalization())
  model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  # model.add(BatchNormalization())
  # model.add(Conv2D(96, kernel_size=(3,3), activation='relu'))
  # model.add(MaxPooling2D(pool_size=(2,2)))
  # model.add(BatchNormalization())
  # model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
  # model.add(MaxPooling2D(pool_size=(2,2)))
  # model.add(BatchNormalization())
  model.add(Dropout(0.3))
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.3))
  model.add(BatchNormalization())
  model.add(Dense(3, activation = 'softmax'))
  
  model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
  model.fit(X_train, y_train, epochs=15, validation_data=(X_test, y_test))
  print("done")
  return render_template('template.html')

@app.route('/my-link/')
def my_link():
  print('I got clicked!')

  return 'Click.'

if __name__ == '__main__':
  app.run(debug=True)
