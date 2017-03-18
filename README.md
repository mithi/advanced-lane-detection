Built an advanced lane-finding algorithm using distortion correction, image rectification, color transforms, and gradient thresholding. Identified lane curvature and vehicle displacement. Overcame environmental challenges such as shadows and pavement changes

---

In this project, I have used computer vision techniques to identify lane boundaries and compute the estimate the radius of curvature given a frame of video of the road.  
To achieve this, the following steps are taken:
- Computed the camera calibration matrix and distortion coefficients of the camera lens used given a set of chessboard images taken by the same camera
- Used the aforementioned matrix and coefficient to correct the distortions given by the raw output from the camera
- Use color transforms, and sobel algorithm to create a thresholded binary image that has been filtered out of unnecessary information on the image 
- Apply perspective transform to see a “birds-eye view” of the image as if looking from the sky 
- Apply masking to get the region of interest, detect lane pixels, 
- Determine the best fit curve for each lane the curvature of the lanes
- Project the lane boundaries back onto the undistorted image of the original view 
- Output a visual display of the lane boundaries and other related information 

### Writeup
- https://github.com/mithi/self-driving-project-4/blob/master/WRITEUP.pdf

### Video Output
- https://github.com/mithi/self-driving-project-4/blob/master/project_video_output.mp4
- https://github.com/mithi/self-driving-project-4/blob/master/project_video_verbose_output.mp4

### Pipeline
- https://github.com/mithi/self-driving-project-4/blob/master/pipeline.ipynb
- https://github.com/mithi/self-driving-project-4/blob/master/pipeline_verbose.ipynb

### Classes
- ChessBoard - https://github.com/mithi/self-driving-project-4/blob/master/chessboard.py
- BirdsEye - https://github.com/mithi/self-driving-project-4/blob/master/birdseye.py
- LaneFilter - https://github.com/mithi/self-driving-project-4/blob/master/lanefilter.py
- Curves -https://github.com/mithi/self-driving-project-4/blob/master/curves.py
