# Turtle Recall: Wildlife Conservation with Machine Learning
Final Project for 'Implementing ANNs with TensorFlow'  &nbsp; <kbd>Implementing ANNs with TensorFlow</kbd> &nbsp;
WS 21/22, Osnabrueck University  
Dennis Hesenkamp, Lennart Zastrow, Madhuri Ramesh  

This repository contains all files associated with our final project for the abovementioned course. Idea and data come from Zindi's [Turtle Recall Conversation Challenge](https://zindi.africa/competitions/turtle-recall-conservation-challenge/data). Implementations are our own if not noted otherwise.

## Virtual environment

To continue, make sure that you have a local distribution of [Conda] installed (e.g. Anaconda, miniconda, miniforge). The code in this repo has been written and tested on an Apple Silicon device with Apple's `tensorflow-macos` distribution.

__environment file will be added in due time__

- `data`: Contains dataframes as .csv files with image IDs and associated turtle IDs  
- `documentation`: Written documentation of the project, including LaTeX source files  
- `functions`: helper functions  
- `models`: code to create the models we used  
- `turtleRecall.ipynb`: main notebook  
- `turtleRecallColab.ipynb`: main notebook adapted for use with Google Colab  
- `utils.py`: utilities  

## Overall Pipeline

The overall pipeline can be executed following the `turtleRecall.ipynb` notebook. The notebook contains useful notes and explanations and guides through the entire process of data acquisition, preprocessing, data augmentation, model creation, and training. It can be seen as a helpful complement to the documentation ([`documentation/documentation.pdf`](https://github.com/dhesenkamp/turtleRecall/blob/main/documentation/documentation.pdf)), which describes many of the steps in greater detail and with scientific background.