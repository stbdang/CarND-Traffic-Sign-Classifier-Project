#**Traffic Sign Recognition** 

##Daniel Kim

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report/histogram.png "Visualization"
[image3]: ./report/fake_data.png "Fake data"
[image4]: ./test_images/50.png "Traffic Sign 1"
[image5]: ./test_images/bumpy_road.png "Traffic Sign 2"
[image6]: ./test_images/children_crossing.png "Traffic Sign 3"
[image7]: ./test_images/narrows_right.png "Traffic Sign 4"
[image8]: ./test_images/roundabout.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/stbdang/CarND-Traffic-Sign-Classifier-Project/)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples = 4410
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a histogram showing how the data is distributed across classes.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I do grayscaling on the images, then I normalize the data per image to zero mean and standard dev of 1 by subtracting by its average value the divide by its standard deviation.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I believe the data set was already divided into training and validation set. 

The code for this step is contained in the fifth code cell of the IPython notebook.

I created fake data by randomized scaling and rotating of the training data. For the final implementation, I've created 2 additional data per each trainging data. I decided to generate additional data because by looking at the distribution of classes, it seemed that there are some classes with small number of images I wanted to provide more training data for those classes. Also, adding randomized scale and rotation seemed to make the learning more robust.

Here is an example of an original image and an augmented image:

![alt text][image3]

My final training set had 104397 number of images. My validation set and test set had 4410 and 12630 number of images.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the sixth cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 28x28x10 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x10 				|
| RELU					|												|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16	 				|
| RELU					|												|
| Fully connected		| 600->70    									|
| Fully connected		| 70->43    									|
|						|												|
|						|												|
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

* Optimizer : Adam 
* Learning rate : 0.005
* Batch : 256
* Epoch : 10
* L2 regularization (optional) : 0.0001

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ? 0.998
* validation set accuracy of ? 0.947
* test set accuracy of ? 0.922

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I chose LeNet as the starting point since it operates on the similar imput dimension and it's known to provide good result.

* What were some problems with the initial architecture?
Because the traffic sign is more complicated that a digit and it had more classes, the maximum accuracy that it could reach was limited (~89%)

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I've added more depth (16 to 24) to the second convolution layer because I figured that the sign would have more varieties in terms of higher level features. I thought the first convolution layer doesn't need additional depth since the number of primitive feature categories which fits into the kernel (5x5x3) would be small.

* Which parameters were tuned? How were they adjusted and why?
Learning rate, sigma and mu for initial weight were adjusted to achieve better results.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
Convolution layer works well because the position/scale of the sign and features can vary.

If a well known architecture was chosen:
* What architecture was chosen?
LeNet
* Why did you believe it would be relevant to the traffic sign application?
It solves a similar problem (digit recognition) very well.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
Validation and test set performance is similar which is good. Would like to see training and validation accuracy to be closer but they are much better than before.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The third image is difficult to classify because the overall constast of the sign vs. background is small. It makes it even harder because of the reflections on the sign which creates artifacts.

The fourth is difficult to classify because there are other signs with similar features (triangle with vertical lines). This is shown by the fact the classifier guessed this to be "Traffic signals", then "Caution". 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        	|     Prediction	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Speed limit (50km/h)  	| Speed limit (50km/h)  						| 
| Bumpy road    			| Bumpy road 									|
| Children crossing			| Keep right									|
| Road narrows on the right	| General caution				 				|
| Roundabout mandatory		| Roundabout mandatory      					|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

It seems for all images, the classifier is very sure about in terms of probablity. It seems to be caused by relatively high logit values.

Top 5 =  TopKV2(values=array([[  9.99999881e-01,   1.09859322e-07,   5.32981517e-08,
          8.57544924e-10,   3.04272496e-10],
       [  1.00000000e+00,   1.15931926e-11,   2.60003370e-15,
          1.18172245e-15,   2.98809997e-17],
       [  9.89136457e-01,   1.01152947e-02,   3.32979311e-04,
          2.01514573e-04,   1.13836300e-04],
       [  9.99598205e-01,   3.95005773e-04,   6.81606480e-06,
          2.08666462e-09,   7.84008081e-10],
       [  9.99351203e-01,   6.48822286e-04,   5.44065404e-10,
          2.32380643e-10,   6.75871295e-12]], dtype=float32), indices=array([[ 2,  4,  1,  0, 39],
       [22, 25, 24, 26, 29],
       [38, 12, 36,  1, 14],
       [18, 26, 40, 11, 37],
       [40, 28, 12, 30,  9]], dtype=int32)) 