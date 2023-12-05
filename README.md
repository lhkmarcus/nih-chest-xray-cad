# NIH Chest X-ray Computer-aided Diagnostics (CAD) with Deep Learning


### General Information and Problem Statement

- Chest X-ray (CXR) imaging is a commonly implemented technique for disease detection and diagnosis. Diseases include pneumonia, tuberculosis, COVID-19, malignancy, among others. It is relatively low cost, and highly accessible. 
- Radiologists can extract important information pertaining to the health of a patient. However, interpretations can sometimes be difficult, and it can be a lengthy and complex procedure. In remote areas, expert radiologists may be hard to come by. 
- Fortunately, research has been focused on machine learning and deep learning approaches for automated diagnostics. This involves the integration of GPUs, medical image processing techniques, and deep learning modelling, allowing the automation of disease detection through CAD systems, which may significantly aid medical professionals in decision-making. 
- With new and improving techniques, CAD systems have improved efficiency by automating menial tasks and priotising cases which are time-sensitive. Furthermore, these systems circumvent the issue where radiologists are not present in remote areas and underdeveloped countries. 


### Project

This project aims to develop a CAD solution to detect a variety of diseases with NIH CXR images. The data is composed of 112,120 CXR images with 15 classes (including "No findings") taken from 30,805 unique patients. The labelling process is detailed in [1].

The classes included (excluding "No findings") are as follows:
- Atelectasis
- Consolidation
- Infiltration
- Pneumothorax
- Edema
- Emphysema
- Fibrosis
- Effusion
- Pneumonia
- Pleural_thickening
- Cardiomegaly
- Nodule Mass
- Hernia


### References
[1] Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. (2017).   ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. *Computer Vision and Pattern Recognition*. https://doi.org/10.48550/arXiv.1705.02315


### Acknowledgements
This work was supported by the Intramural Research Program of the NClinical Center (clinicalcenter.nih.gov) and National Library of Medicine (www.nlm.nih.gov).