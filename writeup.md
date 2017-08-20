##Writeup Template

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a  Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[example_car]: ./output_images/example_car.png
[example_hog_features]: ./output_images/example_hog_features.png
[example_notcar]: ./output_images/example_notcar.png
[final_bboxes]: ./output_images/final_bboxes.png
[heatmap]: ./output_images/heatmap.png
[hog_sub_sampling_window_search]: ./output_images/hog_sub_sampling_window_search.png
[original_image]: ./output_images/original_image.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

All the code referenced in this writeup can be found in the [vehicle_detection.ipynb](./vehicle_detection.ipynb) notebook.

###Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images. 

I started by reading in all the `vehicle` and `non-vehicle` images (see the **Load data** section).  Here is an example image the `vehicle`  class:

![alt text][example_car]

And here an example of the  `non-vehicle` class:

![alt text][example_notcar]

I then reused the `get_hog_features` and `extract_features` from the **HOG Classify** lesson in the [Udacity Self Driving car nanodegree](https://www.udacity.com/)

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][example_hog_features]

####2. Explain how you settled on your final choice of HOG parameters.

I tried all the following combinations of parameters (see the **Choice of HOG parameters** section):
 
```python
color_spaces = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
orients = [6,9,12]
pix_per_cells = [4, 8,16]
cell_per_blocks = [2,4]
hog_channels = [0, 1, 2, "ALL"]
```
For each combination, I extracted the HOG features and build and SVM linear classifier using a subset of 500 images of each class.
The following 3 combinations produced the maximum accuracy:

* Using:  HSV  color space,  12  orientations,  4  pixels per cell,  2  cells per block and  ALL  hog channel
	* Feature vector length: 32400
	* Test Accuracy of SVC =  0.99
* Using:  YCrCb  color space,  6  orientations,  4  pixels per cell,  4  cells per block and  ALL  hog channel
	* Feature vector length: 48672
	* Test Accuracy of SVC =  0.99
* Using:  YCrCb  color space,  9  orientations,  4  pixels per cell,  2  cells per block and  ALL  hog channel
	* Feature vector length: 24300
	* Test Accuracy of SVC =  0.99

I dicided to continue with the third combination because it extracts less features than the two others.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the HOG features extracted as described above (see **Build Model** and **Train with full data** sections).
I used a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) to scale the features to to zero mean and unit variance before training the classifier

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to extract HOG features only once per image, and then for each window searched, extract the corresponding features, normalize them and predict its class using
the trained model (see **Hog Sub-sampling Window Search**). 
The scale of **1.5** and **2** cells per steps taken from the **Hog Sub-sampling Window Search** lesson in the  [Udacity Self Driving car nanodegree](https://www.udacity.com/)
produced acceptable results.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Having an image like this one:

![alt text][original_image]

I apply the following pipeline (see **Car detection pipeline** sectioin):

I first predict the boxes in the image which might contain a car using Hog Sub-sampling Window Search described above. The output looks like this:

![alt text][hog_sub_sampling_window_search]

I then use a heat map to eliminate false positives and keep a single prediction per car (see **Heat map for single detection per car** section)
Following are images of the heatmap coresspoing to the example above, and the resulting boxes

![alt text][heatmap]

![alt text][final_bboxes]

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_outpt.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video (see `Tracker()`).  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions (here I resused the code from the **Heat map for single detection per car** section).  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The performance of the pipleline is quite good on the project video, but following are few points that need to be improved:

* Despite the techniques used here te eliminate false positives, they are still apearing in few places in the output video
* I used only a subset of 5 000 examples of each class to train the classifier, and I din't took into account timeseries in the images of the GTI* folders.
* To make the classifier more robust, I shoul try to use the [Udacity data](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment my training data


