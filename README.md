# Turtle Recognition Using Deep Convolutional Neural Networks
Final Project for **Implementing ANNs with TensorFlow**  
WS 21/22, Osnabrueck University  
Dennis Hesenkamp, Lennart Zastrow, Madhuri Ramesh  

This repository contains all files associated with our final project for the abovementioned course. Idea and data come from Zindi's [Turtle Recall: Conversation Challenge](https://zindi.africa/competitions/turtle-recall-conservation-challenge). Implementations are our own if not noted otherwise.

## Virtual environment

To continue, make sure that you have a local distribution of Conda installed (e.g. Anaconda, miniconda, miniforge). The code in this repo has been written and tested on an Apple Silicon device. Create a local environment with the `environment.yml` file to be able to execute code from the `turtleRecall.ipynb` notebook.

## Repository Structure

- `data`: Contains dataframes as .csv files with image IDs and associated turtle IDs
- `documentation`: Written documentation of the project, including LaTeX source files
- `functions`: Helper functions used in the main notebook
- `models`: Code to create the models we used
- `environment.yml`: Environment in which this project was built
- `turtleRecall.ipynb`: Main notebook with pipeline for the entire project
- `turtleRecallColab.ipynb`: Main notebook adapted for use with Google Colab. Works as standalone, all helper functions, models, and utility variables are included
- `utils.py`: Utilities

## Overall Pipeline

The overall pipeline can be executed following the `turtleRecall.ipynb` notebook. The notebook contains useful notes and explanations and guides through the entire process of data acquisition, preprocessing, data augmentation, model creation, and training. It can be seen as a helpful complement to the documentation ([`documentation/documentation.pdf`](https://github.com/dhesenkamp/turtleRecall/blob/main/documentation/documentation.pdf)), which describes many of the steps in greater detail and with scientific background.

Use the `turtleRecallColab.ipynb` notebook for execution in [Colab](https://colab.research.google.com/) as opposed to local execution. The image files are stored online in a GBucket, which makes working with them in Colab easy, no local download required.