# Improve both phases using features adaptive weighting for street view image classification
# Introduction
    features adaptive weighting is used in both phases to improve the overall performance of street view image classification. Firstly, visual features adaptive weighting is introduced in the detection phase to improve the recognition accuracy of building objects. Then, a semantic matrix is constructed by using the centroid sorting algorithm to encode the detection results. Finally, in the text classification phase, the semantic feature adaptive weighting is used to enhance the connection between the building objects, and then predict a land category label for the street view image. Experimental results show that the proposed method can effectively improve the performance of “detection-encoding-classification” framework in both phases.

# Overall process 
![总体流程图](https://github.com/nuotian1096/Street-view-image-classification/edit/master/1.png)

# Results
| Detector | AP(IoU=0.5:.05:.95) | AP(IoU=0.75) | AP(IoU=0.50) |
|--------|--|--|--|
| Ca-50 | 48.72 | 53.24 | 70.21 |
| Ca-101 | 48.92 | 53.88 | 70.13 |
| Ca-50 + VFAW | 50.1 | 55.2 | 72.4 |
| Ca-101 + VFAW | 50.8 | 55.9 | 72.8 |

visual features adaptive weighting is introduced in the detection phase.

| Models | M-P | M-R | M-F1 |
|--------|--|--|--|
| Layout+simple-RNN | 81.81 | 80.94 | 81.37 |
| Layout+SFAW | 82.04 | 80.10 | 81.06 |
| Centroid Sorting+ SFAW | 83.32 | 79.42 | 81.31 |

Semantic features adaptive weighting is used in the context classification phase.


Project code display, readme.txt is the details 






