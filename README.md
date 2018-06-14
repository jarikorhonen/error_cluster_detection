# error_cluster_detection

## Packet loss visibility estimation based on error clusters

![Error clusters](https://github.com/jarikorhonen/error_cluster_detection/blob/master/tob_2018.png)

In this work, we performed a subjective study on a tapscreen to detect visible packet loss artifacts in a video sequence by tapping a touchscreen. We defined a new concept of "error cluster" to describe packet loss impairments located in a spatiotemporally limited region. On this repository, we share the Matlab code for deterimining the error clusters by comparing the non-impaired decoded reference sequence and the test sequence impaired by packet losses. The code can also read the results of a subjective tapscreen study to assign taps to error clusters, and extract features characterizing error clusters. There is also a Python script for predicting the subjective visibility of error clusters from the features.

Note that some parts of the code is specific to the our subjective study. Matlab script *error_cluster_analysis.m* contains the main functionality. Matlab script *error_cluster_analysis_example.m* shows how the script can be used; however, it cannot be directly used without the test video sequences and the subjective results. You can, however, use it as a starting point for your own study. It writes the results in a comma separated values (csv) file, that can be then analyzed afterwords. Python script *analysis.py* shows an example; however, it contains some hardcoded values specific to our study, so you cannot use it directly in your own work without modifications.

More information about the work is available in the following publications:

J. Korhonen, and C. Mantel, "Assessing Visibility of Individual Transmission Errors in Networked Video," Proc. of IS&T International Symposium on Electronic Imaging, Human Vision and Electronic Imaging (HVEI'16), San Francisco, USA, January 2016. [DOI: 10.2352/ISSN.2470-1173.2016.16.HVEI-106](http://www.ingentaconnect.com/contentone/ist/ei/2016/00002016/00000016/art00030)

J. Korhonen, "Study of the Subjective Visibility of Packet Loss Artifacts in Decoded Video Sequences," IEEE Transactions on Broadcasting, vol. 64, no. 2, pp. 354-366, June 2018. [DOI: 10.1109/TBC.2018.2832465](https://ieeexplore.ieee.org/abstract/document/8361765/ "IEEE Xplore")

If you use the code in your research, please cite our work.
