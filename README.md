# Detecting License Plates with YoloV5 and CNN´s
A fun way to practice some computer vision knowledge using CNN´s.  

Motivation and Results
The motivation behind this project was to create a useful application for the tools I have learned, particularly CNN´s for computer vision. I wanted to develop some parts from scratch (CNN),  but I also wanted to use transfer learning to facilitate the process by leveraging previous work done in the field.

 Cars167.png



In a very high level, the process to accomplish the goal is the following:
1.- Input image
2.- Licence plate location identification
3.- License plate cropping
4.- Image preprocessing on cropped image 
5.- Character detection
6.- Image to text with custom trained CNN.

The following is a more detailed explanation. 

### YoloV5 Licence plate detection 
For the first two points, I used the Crop_plate notebook where I took advantage of a [YoloV5 pretrained model](https://huggingface.co/keremberke/yolov5m-license-plate) specially trained to detect licence plates and their location. 

### Licence plate preprocessing 
To leverage the performance of computer vision it is useful to preprocess images. I preprocessed the cropped license plates to achieve the following:

- Resize the image
- Convert it into grayscale
- Inverted the colors
- Image Erotion
- Image Dilation 

With this preprocessing steps I intended to have the images in the same format as the images I used during training for my CNN and also achieve a better quality in the image and make it easier for future CNN to detect borders and patterns in the images.

Images of preprocessing 



### Character Detection
Finally I identify the individual characters of the licence plate sequence, as my CNN will predict individual characters and not the complete string. To do this, I identify the contours of the characters with the help of the CV2 library. Then I filter them based on size criteria to get only the important contours on the image and do not include noise like dashes or other symbols.

And finally this detected contours are cropped again into individual pictures which will be fed into a CNN. 

Aditional images




### Convolutional Neural Network
The CNN is the model that will "see" the images and predict which character they are. To train the CNN I used [this dataset](
https://drive.google.com/drive/folders/1dtFh-x_NX_PXqPS_k1SrTay7bdDxJGJG?usp=sharing) which I originally downloaded form a [github repo](https://github.com/SarthakV7/AI-based-indian-license-plate-detection/blob/master/data.zip) which contains 864 images for training and 216 images for validation. Each separated into 36 classes (0-9 and A-Z) of equal size. I downloaded them and added them into my personal google drive folder to be able to retrive them directly from Google Colab anytime. In my drive folder I also have the cars dataset which I use to predict. 

I created 5 different CNN models with different architectures and different parameters. I wanted to try different architectures and compare how well they generalized for the unseen data of real licence plates. In terms of training performance they were all pretty similar with  accuracy levels over 98% and validation over 96%. 
I decided to stick with the model that was having better accuracy at predicting real licence plates and not the one with the best training/validation accuracy. The model architecture that I chose at the end is very simple given that our images were also simple.


### Opportunities and Future Steps 
The downside of the dataset used to train the CNN is that characters have very particular shapes. In licence plates, the shapes are often not that sharp or have different styles and that is where our model has difficulties interpreting the characters. One of my errors was trying to predict licence plates from different countries and different styles with a dataset that is not well generalized. I noticed some confusion mainly between 0 and D and also between some H and M characters. 

As next steps, we can take advantage of the YoloV5 model, the multiple open source cars datasets and the preprocessing steps to enlarge the characters dataset into a wider and more diverse dataset and then come up with different CNN architectures and retrain on the richer data.
 
