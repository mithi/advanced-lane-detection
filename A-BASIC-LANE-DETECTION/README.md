# Detected highway lane lines on a video stream. Used OpencV image analysis techniques to identify lines, including Hough Transforms and Canny edge detection.
---

### Requirements: 
- Jupyter notebook: BASIC_LANE_DETECTION.ipynb
- Writeup: WRITEUP.pdf 

### Possible improvements on my pipeline
- Parameters can be tweaked for even better performance
- If there was abrupt changes between two frames, we can reject the result of the second frame because it does not make sense.
- Considering the following two consecutive frames, this does not make sense because of the abrupt change so we should disregard the second frame and consider it as an error/anomaly/miscalculation.

### References
- http://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
- https://alyssaq.github.io/2014/understanding-hough-transform/
- https://medium.com/@vijay120/detecting-car-lane-lines-using-computer-vision-d23b2dafdf4c#.x8h9qq21q
- http://airccj.org/CSCP/vol5/csit53211.pdf
- http://stackoverflow.com/questions/36598897/python-and-opencv-improving-my-lane-detection-algorithm
