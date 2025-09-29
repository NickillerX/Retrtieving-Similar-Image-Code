# Content-Based Image Retrieval for Computer Science Dissertation
**Author:** Nicholas Jaune
**Institution:** University Of Mauritius

---

## Abstract

This repository contains the source code and documentation for the dissertation titled: "*Retrieving Similar Image Using Machine Learnig & Computer Vison*". 

The project implements a content-based image retrieval (CBIR) system using a custom Convolutional Neural Network (CNN) and transfer learning techniques. The system can take a query image and retrieve the top-k visually similar images from a large dataset.


## Code and Usage

The primary scripts for this project are located in the `/src` folder. 

### Dependencies
To run this code, you will need the following Python libraries:
* Python 3.10 or lower
* pytocrh 2.x
* NumPy
* Pillow
* scikit-learn

You can install them using:
`pip install tensorflow numpy pillow scikit-learn`

### How to Run
1.  **Prepare the dataset:** Place your image dataset in the `/data` folder and upload to gcs
2.  **Train the model:** see code: "code_for_GCLab"
3.  **Perform a search:** See code: "gui.py"


