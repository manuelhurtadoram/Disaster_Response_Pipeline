# Disaster Response Pipeline
This project builds a Multi-Output Random Forest Classifier to classify disaster response messages into one of 36 relevant categories such as food, weather, etc and therefore help
disaster relief organizations better respond to emergencies. 

The trained model has been embedded into a web application, where the user can input a message and obtain the message's 
classification in return. 

***Disaster Response Pipeline***

1. **Installations**

    - Uses Python 3.0
    
    - Other libraries used:
    
        - **Pandas**: https://pandas.pydata.org/pandas-docs/stable/)
        - **Numpy**: https://docs.scipy.org/doc/)
        - **Sci-Kit Learn**: https://scikit-learn.org/stable/documentation.html)
        - **NLTK**: https://www.nltk.org
        - **Flask**: http://flask.pocoo.org/docs/1.0/
        - **Joblib**: https://joblib.readthedocs.io/en/latest/
        - **Plotly**: https://plot.ly/python/

2. **Project Motivation**

This project was completed as part of the Udacity Data Scientist Nanodegree program requirements.

Like all the projects related to machine learning that I have undertaken in the past, this model has important
real-life applications, since it can help disaster relief organizations filter through the deluge
of information they may receive from volunteers and news reports to focus their limited attention towards the 
content that is relevant to their operations. 

Aggregating the increase in efficiency of all the different organizations that would benefit from such a model,
disaster relief could be streamlined considerably in order to more effectively help stricken populations. 

3. **File Descriptions**

	1. **Data**:
	
		1. *process_data.py*: Python script for an ETL pipeline that stores the cleaned data in a local SQLite database.
    		2. *DisasterResponseData.db*: SQLite database containing clean data.
   		3. *disaster_categories.csv*: Comma-separated file with information on the labels associated with each message.
    		4. *disaster_messages.csv*: Comma-separated file containing the raw messages.
		
	2. **Models**: 
	
		1. *train_classifier.py*: Python script for an ML pipeline that loads the data from the SQL database, extracts the features of the data, and trains a classifier on it. The script then saves the model as a pickle file to access it from the web app. 
		2. *trained_model.pkl*: File containing the data for the trained classifier.
		
	3. **App**:
	
		1. *run.py*: Python script for launching the Flask web app containing the message classifier and other data visualizations. 
		2. *templates*:
			1. *master.html*: HTML markup for the landing page of the web app.
			2. *go.html*: HTML markup containing div to show message classification results. 


4. **Instructions**

To interact with the project, clone the Repo and execute
```python
python3 run.py
```
from within the 'app' folder in your terminal. The link to the web app will then appear. 


5. **Acknowledgements**

This project was made possible with the knowledge I obtained from the Udacity Machine Learning Engineer Nanodegree and the Udacity Data Scientist Nanodegree. 

The disaster_categories.csv and disaster_messages.csv data were made available by FigureEight as part of a collaboration with Udacity.

The web app template and code were provided by Udacity, and modified by myself to include the relevant model data as well as 
personalized visualizations. 
