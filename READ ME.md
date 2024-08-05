

Key Requirements:
Python 3.6.1
OpenCV 3.4.1
Keras 2.0.2
Tensorflow 1.2.1
Theano 0.9.0   


 
# Repo contents
- **trackgesture.py** : The main script launcher. This file contains all the code for UI options and OpenCV code to capture camera contents. This script internally calls interfaces to gestureCNN.py.
- **gestureCNN.py** : This script file holds all the CNN specific code to create CNN model, load the weight file (if model is pretrained), train the model using image samples present in **./imgfolder_b**, visualize the feature maps at different layers of NN (of pretrained model) for a given input image present in **./imgs** folder.
- **imgfolder_b** : This folder contains all the 4015 gesture images I took in order to train the model.
- **version1.5.hdf5** : This file is a pretrained model of training over 150 epochs.
- **result of version1.5** : This file contains the accuracy and loss of train set and validation set of training over 150 epochs.



# Usage
**On Mac**
```bash
eg: With Theano as backend
$ KERAS_BACKEND=tensorflow python trackgesture.py 
```
**On Windows**
```bash
eg: With Tensorflow as backend
> set "KERAS_BACKEND=tensorflow"
> python trackgesture.py 
```

l am setting KERAS_BACKEND to change backend to Theano, so in case you have already done it via Keras.json then no need to do that. But if you have Tensorflow set as default then this will be required.

# Features
This application comes with CNN model to recognize upto 5 pretrained gestures:
- OK
- PEACE
- STOP
- PUNCH
- NOTHING (ie when none of the above gestures are input)

This application provides following functionalities:
- Prediction : Which allows the app to guess the user's gesture against pretrained gestures. App can dump the prediction data to the console terminal or to a json file directly which can be used to plot real time prediction bar chart 
- New Training : Which allows the user to retrain the NN model. User can change the model architecture or add/remove new gestures. This app has inbuilt options to allow the user to create new image samples of user defined gestures if required.
- Visualization : Which allows the user to see feature maps of different NN layers for a given input gesture image. Interesting to see how NN works and learns things.



# Gesture Input
I am using OpenCV for capturing the user's hand gestures. In order to simply things I am doing post processing on the captured images to highlight the contours & edges. Like applying binary threshold, blurring, gray scaling.

I have provided two modes of capturing:
- Binary Mode : In here I first convert the image to grayscale, then apply a gaussian blur effect with adaptive threshold filter. This mode is useful when you have an empty background like a wall, whiteboard etc.
- SkinMask Mode : In this mode, I first convert the input image to HSV and put range on the H,S,V values based on skin color range. Then apply errosion followed by dilation. Then gaussian blur to smoothen out the noises. Using this output as a mask on original input to mask out everything other than skin colored things. Finally I have grayscaled it. This mode is useful when there is good amount of light and you dont have empty background.

**Binary Mode processing**
```python
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),2)   
th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
```



# CNN Model used
The CNN I have used for this project is pretty common CNN model which can be found across various tutorials on CNN. Mostly I have seen it being used for Digit/Number classfication based on MNIST database.

```python
model = Sequential()
model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                    padding='valid',
                    input_shape=(img_channels, img_rows, img_cols)))
convout1 = Activation('relu')
model.add(convout1)
model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
convout2 = Activation('relu')
model.add(convout2)
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
```

This model has following 12 layers -
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_1 (Conv2D)            (None, 32, 198, 198)      320       
_________________________________________________________________
activation_1 (Activation)    (None, 32, 198, 198)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 32, 196, 196)      9248      
_________________________________________________________________
activation_2 (Activation)    (None, 32, 196, 196)      0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2) (None, 32, 98, 98)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 32, 98, 98)        0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 307328)            0         
_________________________________________________________________
dense_1 (Dense)              (None, 128)               39338112  
_________________________________________________________________
activation_3 (Activation)    (None, 128)               0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 5)                 645       
_________________________________________________________________
activation_4 (Activation)    (None, 5)                 0         
=================================================================
```
Total params: 39,348,325.0
Trainable params: 39,348,325.0
















