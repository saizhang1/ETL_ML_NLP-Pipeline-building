# ETL_ML_NLP-Pipeline-building
Build ETL and ML pipeline for a real case(Disaster Response) by applying data engineering knowledge, build NLP pipeline for text mining.

## Project Overview
In original data file, there is data set containing real messages that were sent during disaster events. I need to create a 
machine learning pipeline to categorize these events so that I can send the messages to an appropriate disaster relief agency.

This project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will show off your software skills, including your ability to create basic data pipelines and write clean, organized code!

Below are a few screenshots of the web app.
![Capture](https://user-images.githubusercontent.com/36822899/56208621-dbeef980-6051-11e9-8950-85d6c0e80ccc.PNG)

![Capture1](https://user-images.githubusercontent.com/36822899/56208669-f4f7aa80-6051-11e9-82ca-071821a10f3d.PNG)

## Project Components
There are three components you'll need to complete for this project.

#### 1. Original Data

In this file, you will see the original data set of the disaster message.

#### 2. Pipelines

In this file, you will see the pipeline I have built for both ETL and ML.

> ETL pipeline:

I write a data cleaning pipeline that: 

* Loads the messages and categories datasets

* Merges the two datasets

* Cleans the data

* Stores it in a SQLite database

> ML Pipeline:

I write a machine learning pipeline that:

* Loads data from the SQLite database

* Splits the dataset into training and test sets

* Builds a text processing and machine learning pipeline

* Trains and tunes a model using GridSearchCV

* Outputs results on the test set

* Exports the final model as a pickle file

#### 3. Models

In this file, I put in 2 python script files:

> process_data

Is the file contains the functions of clean data based on the ETL pipeline I build.

> train_classifier

Is the file contains the functions of machine learning training models based on the ML pipeline I build.

#### 4. App

In this file you will see the codes of some javascript, html and css to be able to run a web app and visualize the data.