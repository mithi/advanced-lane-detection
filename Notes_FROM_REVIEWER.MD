Notes From The reviewer 

To see some ideas on using deep learning instead, you can read
- https://chatbotslife.com/small-u-net-for-vehicle-detection-9eec216f9fd6#.3dr04zwkm
- https://github.com/JamesLuoau/Self-Driving-Car-Vehicle-Detection
- http://www.learnopencv.com/histogram-of-oriented-gradients/

- You might be able to speed up the pipeline's speed a bit while maintaining good accuracy by using more pixels per cell. For example, I've seen these settings perform well...

` colorspace = 'YUV' `
` orient = 11 `
` pix_per_cell = 16 `
` cell_per_block = 2 `
` hog_channel = "ALL" `

- To optimize the model's accuracy, you could also try a grid search to tune the SVC's C parameter.
- http://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
- http://scikit-learn.org/stable/modules/grid_search.html

- It will probably help the output video results to use a different colorspace for the HOG extraction. Most students have success with YCrCb or YUV, and the best solutions seem to use one of those colorspaces with all color channels.

- http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
-  Tip: You might be able to use fewer window scales — many students do well using just 1-3 scales, with overlap in the 0.65 - 0.85 range.

### Tip: To help with reducing false positives, other ideas you could have tried include:
- Using the SVM's built-in decision_function method, which returns a confidence score based on how far a data point is from the decision boundary — higher values equate to higher confidence predictions that can be thresholded.
- Augment the training with hard negative mining.
- http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC.decision_function
- https://www.reddit.com/r/computervision/comments/2ggc5l/what_is_hard_negative_mining_and_how_is_it/

###  Tip: pipeline speed
Considering that a pipeline on a real car needs to be processed in real-time, it would be great to also address the pipeline's speed performance more explicitly (e.g., in frames per second).

For improved speed performance in extracting HOG features, you could try using cv2.HOGDescriptor (read more here)
And for additional ideas on speeding up vehicle detection, you can read about Haar Cascades.
- http://stackoverflow.com/questions/28390614/opencv-hogdescripter-python
- https://github.com/andrewssobral/vehicle_detection_haarcascades
