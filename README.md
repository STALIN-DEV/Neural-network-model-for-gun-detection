# Neural network model for gun detection
 A trained neural network model designed to detect people with a gun on surveillance cameras

 ## Features

- The model is trained and ready to use in other projects
- The model was trained on custom dataset with 1871 photos
- The neural network was written on Python using Google Colab and Visual Studio
- Accuracy on the test dataset was 93.4%
- During training, maximal accuracy was 0.9247 / 92.4%
```sh
7/7 - 54s - loss: 0.2165 - accuracy: 0.9150 - val_loss: 0.2000 - val_accuracy: 0.9247 - 54s/epoch - 8s/step
```

## Tech

The project uses:

- tensorflow
- numpy
- matplotlib - plotting and drawing frames (disabled)

## Installing

To install you must:
```sh
git clone https://github.com/STALIN-DEV/Neural-network-model-for-gun-detection
```

## Using

If you want to use the model in your project, move it to the project repository and import it:
```python
from tensorflow.keras.models import load_model
model = load_model('gunmen.h5')
model.summary()
```

```sh
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 100, 100, 16)      1216      
                                                                 
 max_pooling2d (MaxPooling2  (None, 50, 50, 16)        0         
 D)                                                              
                                                                 
 conv2d_1 (Conv2D)           (None, 50, 50, 32)        12832     
                                                                 
 max_pooling2d_1 (MaxPoolin  (None, 25, 25, 32)        0         
 g2D)                                                            
                                                                 
 conv2d_2 (Conv2D)           (None, 25, 25, 64)        51264     
                                                                 
 max_pooling2d_2 (MaxPoolin  (None, 12, 12, 64)        0         
 g2D)                                                            
                                                                 
 conv2d_3 (Conv2D)           (None, 12, 12, 128)       204928    
                                                                 
 max_pooling2d_3 (MaxPoolin  (None, 6, 6, 128)         0         
 g2D)                                                            
                                                                 
 flatten (Flatten)           (None, 4608)              0         
                                                                 
 dense (Dense)               (None, 1024)              4719616   
                                                                 
 dropout (Dropout)           (None, 1024)              0         
                                                                 
 dense_1 (Dense)             (None, 256)               262400    
                                                                 
 dropout_1 (Dropout)         (None, 256)               0         
                                                                 
 dense_2 (Dense)             (None, 131)               33667     
                                                                 
=================================================================
Total params: 5285923 (20.16 MB)
Trainable params: 5285923 (20.16 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

