# Improve both phases using features adaptive weighting for street view image classification
# Introduction
  Features adaptive weighting is used in both phases to improve the overall performance of street view image classification. Firstly, visual features adaptive weighting is 
introduced in the detection phase to improve the recognition accuracy of building objects. Then, a semantic matrix is constructed by using the centroid sorting algorithm to 
encode the detection results. Finally, in the text classification phase, the semantic feature adaptive weighting is used to enhance the connection between the building objects, 
and then predict a land category label for the street view image. Experimental results show that the proposed method can effectively improve the performance of [“detection-
encoding-classification”](https://github.com/nuotian1096/Context-Encoding-of-Detected-Buildings) framework in both phases.

# Overall process 
![总体流程](https://github.com/nuotian1096/Street-view-image-classification/blob/master/1.png)

# Results
| Detector | AP(IoU=0.5:.05:.95) | AP(IoU=0.75) | AP(IoU=0.50) |
|:--------:|:--:|:--:|:--:|
|    Ca-50   |    48.72 | 53.24 | 70.21 |
| Ca-50 + CBAM  | 50.21 |  55.38 |  72.82 |
| Ca-50 + S-FAW  | 50.94 |  55.97 |  73.32 |
|    Ca-101  |    48.92 | 53.88 | 70.13 |
| Ca-101 + CBAM | 51.14 |  56.28 |  73.17 |
| Ca-101 + S-FAW | 51.83 |  56.91 |  73.78 |

Local self-correlation guided feature adaptive weighting is introduced in the "bottom-up" phase.

| Models | M-P | M-R | M-F1 | ACC |
|:--------:|:--:|:--:|:--:|:--:|
| MLP | 80.93 | 79.89 | 80.41 | 84.94 |
| RNN | 81.47 | 80.53 | 81.00 | 85.17 |
| GRU | 80.43 | 79.22 | 79.82 | 84.57 |
| LSTM | 80.58 | 79.50 | 80.17 | 84.73 |
| C-FAW | 82.69 | 81.44 | 82.06 | 86.13 |

Local cross-correlation guided feature adaptive weighting is used in the "top-down" phase.

| Models | M-P | M-R | M-F1 | ACC |
|:--------:|:--:|:--:|:--:|:--:|
| no sorting | 82.69 | 81.44 | 82.06 | 86.13 |
| Layout+RNN | 81.47 | 80.53 | 81.00 | 85.17 |
| Layout+C-FAW | 83.04 | 81.10 | 82.06 | 86.21 |
| Centroid Sorting+ C-FAW | 83.73 | 81.22 | 82.64 |

Centroid Sorting algorithm is used in bounding boxes sorting.

# Experiments and Requirement
GPU: GeForce GTX 1080 X 2; OS: Ubuntu 18.04.3 LTS; CUDA Version: 1.8.0 for cu102; TorchVision Version: 0.9.0 for cu102. 

# Citing:

# Get strating
Project code display, readme.txt is the details.






