# Practical Computer Vision Topics

## Basic Concepts
- coming soon
## Pipeline
### Data input
- In my experience for good results in limited time you should have 3000+ images for each output class, multiply this number if class is complex.
#### ETL tools
##### Data Validation
##### Data versioning
##### Data Storage
- Images can be stored in lossy (like jpeg) or lossless (like png) formats
- lossy formats like jpeg introduce compression artefacts
- If you train with lossless images but expect that lossy images will be used for inference then including jpeg compression augmentation in pipeline can increase model's accuracy in real world situations
- Another common mistake is storing segmentation masks in jpeg images which may cause mask to change value during compression

### Preprocessing
#### Augmentations
- types
    - Geometric
    - Non-gemetric
- see imgaug python library to see list of transformations commonly used
- Goal is to make model training more general and also to augment when input data is of very low quantity
- Should be used in moderation
- GANs are also used for data augmentation because
    - They can produce realistic data
    - They are also used when privacy is required like in Medical Health Data. Most of the time you do not have permission to use patient data because of privacy laws. For that purpose GANs can be used to replace original data while maintaining the original data distribution and removing any identifying info.
#### Input Scaling
- scaled_input = (input-mean)/std
- mean and std may depend on data or can be set for given models
- e.g. some models perform well when input is between -1 and 1. To achieve that mean = 127.5 and std = 127.5
#### Input Quantization
- if your model is quantized then you need to quantize your inputs as well
- see [link](https://www.tensorflow.org/lite/convert/metadata#normalization_and_quantization_parameters) for further info
#### Resizing Input
- Most of the models are trained with resized input.
- Resizing with padding is usually a good approach to preserve image scaling and results in higher accuracy compared to simple resizing
- Resizing algo is also important. Commonly resizing is done via pixel value interpolation and methods are
    - Nearest (fastest but may cause a large amount of aliasing/artefacts)
    - Bilinear (fast and usually default method)
    - Cubic (slow and may cause some blurring but reduces aliasing artefacts)

### Training
#### Optimizer types
- Optimizers main goal is that global minimum loss is found. Optimizer can be thought of as a blind man finding floor of lowest valley. Optimizers usually use heuristic approaches for finding global minimum but they usually get stuck in local minimum. This means that distribution of your data is an important factor in deciding on an optimizer. Most likely you will need experimentation to decide on the best one but you can usually get along by using a simple optimiser like adam because simple means easier to debug.
- Types
    - Adam
    - RMSProp
    - Batch gradient decent (studies show that this simple algo **sometimes** works better than others, so one must not simply disregard this)
- see tensorflow / keras losses to see list of all optimizers
- see tensorflow addons to see some interesting optimizers
#### Losses types 
- See keras / tensorflow losses to find a comprehensive list
- see tensorflow addons library to see some interesting losses
#### Learning rate scheduling
- I have found that learning rate  scheduler to be the most important parameter to tune when convergence is an issue
- There are many different types of schedulers (see tensorflow-addons and automl/efficientdet )
- Some of the most important schedulers that I found are in automl/efficientdet repo. [link](https://github.com/google/automl/blob/master/efficientdet/tf2/train_lib.py)
- Some common characteristics of learning rate schedulers:
    - initial learning rate
    - warmup period and warmup function (to get maximum training during start)
    - learning rate decay function (to get bet convergence)
    - Scheduler may also be periodic (increasing and decreasing learning rate periodically) so that model does not get stuck in local minimum but it may also cause divergence during later stages. If you plant to use periodic scheduler make sure to use a decay function along as well.

#### Other training factors
- 300 to 600 epochs is usually a good number in my experience for training if data is complex. (depends largely on task and data)
- Stop early if chances of overfitting or introduce regularisation. Make sure you know what type of regularization you want (l1 or l2 or something else). Depends on what you want to achieve. Regularization is one of the most important factor in model generalization (after data distribution). Rule of thumb is that you want maximum depth model that you can afford with as much regularization as necessary.
- Sometime loss may not reduce for 40-60 epochs especially in later stage of training when reduction of loss stagnates. If you allow training to continue you may be rewarded with a good performance boost (depends on learning rate scheduler)
- If you can then it is best that you perform quantization during training for best accuracy
- Quantization can cause training divergence

### Model Evaluation
#### Kfold evaluation 
#### Metrics
#### Some notes
- One of good practice for evaulation is that you extract top 10-20 examples from test dataset on which model performed worse and try to understand. what went wrong
- Another one is model understanding. There are various techniques to learn why your model predicted the given output e.g. [link](https://www.tensorflow.org/tutorials/interpretability/integrated_gradients). This becomes necessary when the model is used in critical applications.

### Inference
####  Model Optimization
- There are libraries like TensorRT for automatic model optimisations.
- Operations used in models are first thing to optimise. E.g. softmax layer can reduce model execution speed by up to 30% depending on hardware so first step of optimisation is usually to substitute expensive operations with simpler ones.
- Models are quantized according to required hardware to achieve compatibility and less size in memory.
- Model optimization depends on what type of device you are deploying your model to and what hardware it will run on. e.g:
    - CPU: modern CPU can run almost any operation but since it has very little parallelism the execution is slow
    - GPU: GPU can run only matrix operations so this limits the type of models you can run on it. It is faster than cpu and benifits greatly from batch input. It has good throughput
    - TPU - Can only run matrix operations and it is more restricted than GPU. Very high throughput
    - Custom ASICs / FPGAs : For large scale industrial applications
- Some examples
    - iphone gpu is float16 so it runs float16 quantized models best. If you try to run e.g. 8 bit quantized model that in theory should run faster but it will be considerably slowed
    - pixel phones have tpu that can only run int8 models
    - Microcontrollers usually don't have floating point operations

#### Inference pipelines
- Like training pipelines there are several frameworks for inference that include
    - Data ETL
    - Inference
    - Results ETL
- See Tensorflow Extended as an example of production framework and pipelines used in model inference
- NVidia also features some very optimised production engines like DeepStream 



## Computer Vision Tasks
### Object Detection
- One of the best repo for tensorflow users to see implementation of object detectors is [automl/efficientdet](https://github.com/google/automl/tree/master/efficientdet). It contains implementation of almost all the important concepts in computer vision model training. If reader has time it is recommended to run this repo and try and understand the code. (best approach will be to train for your own custom task)
- Types of detectors (also read on pros and cons of each)
    - Single Shot (low accuracy but high inference speed)
        - YOLO/SSD
            - Anchorboxes
            - NMS algo
         - Centernet
             - Detection algo using center points
             - Can also detect keypoints
    - Multi Shot (high accuracy but low inference speed)
        - Usually involve region proposals

### Keypoints detection
- Keypoints direct prediction (like in centernet) It is gaining more popularity these days because of simplicity of use i.e. its output is of shape Nx3 where N is number of keypoints and last dimension containes x,y, and confidence of keypoint. These detectors are also very simple to deploy and use on mobile devices
- Keypoints detection using heatmaps
    - output heatmap types:
        - Gaussian heatmap: shape: HxWxN where N is number of keypoints. output[i,j,k] gives probability that pixel[i,j] represents keypoint k.
        - Vector heatmap: shape is (needs checking) HxWx2xN where N is number of classes and third dimension of size 2 represents a x,y vector pointing to the pixel where keypoint k (last dimension) is actually located
    - Single person - simple
    - Multi person detectors use single person detectors using heatmap with further postprocessing of heatmaps. It usually involves heuristic algos (low accuracy)


### Image Segmentation
- SOTA models usually follow a UNET structure
    - Feature extraction stage
    - Deconvolution stage (important)
    - Loss function is usually same as Image classification like binary cross entropy
    - In original UNET paper the loss is little different because author wanted to give importance to cell boundaries (interesting read)
    - Models have interesting invariance characteristics (some have scaling invariance) (advanced)
- Model output types:
    - output shape: HxWxN where N is the number of classes and output[i,j,k] gives probability of pixel (i,j) for class k
    - output shape : HxWx1 , output[i,j] gives the predicted class of pixel [i,j]
    - Sometimes input shape and output shape is not same due to type of convolution / deconvolution used
### Image Captioning
- coming soon

### Feature Extraction
- A task with many sub applications
    - Similarity detection
        - Model is trained to find similarity between multiple images. Face recognition is important application
        - See the losses used for face recognition
        - Model structure is usually like Image Classification model with outer layer removed
        - Also used in object trackers to find lost object again, see Deep SORT 
    - Image encoding
        - Reduced number of features to represent image data. Applications in data obfuscation, encryption, and image compression
        - Model structure is usually encoder decoder type where during training input and output is same
        - After training encoder is used to extract features and decoder is used to get original image back from features
        - If model structure includes only convolution layers then model is input size invariant. This means it can accept image of any input size

### Image style transfer / Super Resolution
 - Coming soon

