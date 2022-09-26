# Predictive-CT-Scan-Imaging-Research
Currently, patients undergo four different scans before a 3D image of the liver is obtained. The objective of the research is to use the data extracted from the available samples to develop a machine learning program which will predict 25%-50% of the CT volumes. If successful, this would allow for patients to be scanned fewer times resulting in reduced exposure to harmful radiations.​

## Tools  

### Visual Geometry Group Annotator 
VGG Image Annotator is a simple and standalone manual annotation software for image, audio and video. VIA runs in a web browser and does not require any installation or setup. The complete VIA software fits in a single self-contained HTML page of size less than 400 Kilobyte that runs as an offline application in most modern web browsers. 

### OpenCV 
OpenCV is the tool we used for image processing tasks. It is an open-source library that can be used to perform tasks like face detection, objection tracking, landmark detection, and much more. It supports multiple languages including python, java C++. 

### Monai (Preprocessing Package) 
We used Monai, an open-source framework based on PyTorch to perform the data preprocessing step. A regular Monai workflow can be seen in figure 1. For the purpose of this project, we focus mainly on adopting the transforms chain and the fashion of using Dataset/DataLoader.  

### U-Net Model 
The basic organization of the U-Net code into convolutional (conv.py), decoding (decoding.py), and encoding (encoding.py) components that we are adopting was drawn heavily from Fernando Perez-Garcia's excellent U-Net code (Repository at https://github.com/fepegar/unet/tree/master/unet). 

### PyTorch Lighting (Training Workflow) 
We used the PyTorch Lighting package to organize our training code for the following advantages: 
- Flexible (this is all pure PyTorch), but removes a ton of boilerplate 
- More readable by decoupling the research code from the engineering 
- Easier to reproduce 
- Less error-prone by automating most of the training loop and tricky engineering 
- Scalable to any hardware without changing your model 

### torchsummaryX (Visualization of the Model) 

We used the summary function from the package torchsummaryX, a visualization tool that aims to provide information complementary to, what is not provided by print(your_model) in PyTorch. This tool allowed us to visualize kernel size, output shape, # params, and Mult-Adds. 
