# Project Outline: Identifying and Understanding Posts in Mental Health Subreddits using Natural Language Processing

** Research Question** How can natural language processing and machine learning techniques be utilised to classify and analyse posts in mental health on Reddit? What insights can be extracted from social media to contribute to understanding mental health discussions and potentially aiding clinical research on user well-being?

## Table of Contents
- [Data Gathering and Pre-processing](#data-gathering-and-pre-processing)
  - [1.1 Data Scraping](#1.1-data-scraping)

- [Database Initialisation](#database-initialisation)
  - [2.1 Gather Connection Details](#2.1-gather-connection-details)
  - [2.2 Establish a Connection](#2.2-establish-a-connection)

- [Develop Ground Truth](#develop-ground-truth)
  - [3.1 Develop Annotation Guide](#3.1-develop-annotation-guide)
  - [3.2 Develop Annotation Widget](#3.2-develop-annotation-widget)
  - [3.3 Seed Set Annotation](#3.3-seed-set-annotation)

- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [4.1 General Analysis](#4.1-general-analysis)

- [Data Cleaning](#data-cleaning)
  - [5.1 Develop Slang Dictionary](#5.1-develop-slang-dictionary)
  - [5.2 Data Cleaning](#5.2-data-cleaning)

- [Natural Language Processing](#natural-language-processing)
  - [6.1 BERT Pretraining](#6.1-bert-pretraining)
  - [6.2 Fine-Tuning BERT](#6.2-fine-tuning-bert)
  - [6.3 Model Evaluation and Optimisation](#6.3-model-evaluation-and-optimisation)
  - [6.4 Interpretability](#6.4-interpretability)
  - [6.5 Evaluation](#6.5-evaluation)

- [Sentiment Analysis](#sentiment-analysis)
  - [7.1 Sentiment Scoring](#7.1-sentiment-scoring)
  - [7.2 Sentiment Visualisation](#7.2-sentiment-visualisation)

- [Active Learning and Evaluation](#active-learning-and-evaluation)
  - [8.1 Active Learning and Manual Review](#8.1-active-learning-and-manual-review)
  - [8.2 Community Cross-over Evaluation](#8.2-community-cross-over-evaluation)
  - [8.3 Assess Contribution to Clinical Research](#8.3-assess-contribution-to-clinical-research)

- [Ethics, Conclusion, and Further Work](#ethics-conclusion-and-further-work)
  - [9.1 Ethics Consideration](#9.1-ethics-consideration)
  - [9.2 Conclusion](#9.2-conclusion)
  - [9.3 Further Work](#9.3-further-work)


## Data Gathering and Pre-processing

### 1.1 Data Scraping
- Utilise Python's PRAW (Python Reddit API Wrapper) to scrape data from chosen mental health subreddits, including depression, bipolar, schizophrenia, suicide_watch, and others.
- Collect posts and associated metadata from the year 2022.
- Data extraction will be performed for the months of January, April, July, and October to capture diverse user mental states influenced by different seasons.


## Database Initialisation

### 2.1 Gather Connection Details:
- Obtain the necessary connection details for the PostgreSQL database:
- Host: The IP address or domain name where the PostgreSQL server is hosted.
- Port: The network port number on which the PostgreSQL server is listening.
- Database Name: The name of the PostgreSQL database.
- User: The username used to authenticate and connect to the database.
- Password: The password associated with the specified user.

### 2.2 Establish a Connection
- Set up a new connection to the PostgreSQL database using the gathered connection details.

## Develop Ground Truth

### 3.1 Develop annotation guide based on best practices from DSM-5, CAMH, and other sources.
- Divide data into 5 categories: immediate crisis, mental distress, challenges/struggles, recovery/management, and advice/support.
- Create guideline for cases where posts are ambiguous, or could fit into more than one category

### 3.2 Develop annotation widget
- Use ipywidget to create an annotation tool with the 5 class options:immediate crisis, mental distress, challenges/struggles, recovery/management, and advice/support.
- Set up widget to save directly to CSV.

### 3.3 Seed Set Annotation
- Manually review a small subset (500 rows) of the scraped data and annotate posts as "suicidal" or "non-suicidal" based on their content.
- Use the annotated seed set as the initial labeled dataset.
- Aiming to manually review 5000 per subreddit community (start with 4 communities - allow for 1-2 days for this)
- Make note of any interesting observations that occur when reading through the annotations.

## Exploratory Data Analysis (EDA)

### 4.1 General Analysis
- Conduct an exploratory analysis of the data, including basic statistics on the number of posts, number of users, most active users, and other relevant features.

## Data Cleaning
### 5.1 Develop Slang Dictionary
- Create a dictionary of clinical and diagnostic acronyms, mental health slang, and shortforms. This is so the model can fully understand what is being said as language models can miss nuance if they do not understand a slang meaning (i.e. kms may look like kilometres to BERT, however, in the mental health subreddit it is referring to kill myself.)

### 5.2 Data Cleaning
- Run spell check on the post data (include the slang dictionary to learned terms)
 - Clean the extracted data by removing redundant or unnecessary information, handling missing values, and normalising text (e.g., converting to lowercase, removing punctuation and special characters).

## Develop BERT Model

### 6.1 BERT Pretraining
- Leverage a pretrained BERT model (e.g., BERT base or BERT large) as a starting point for the fine-tuning process.
- Utilise BERT's knowledge of language patterns and semantics.

### 6.2 Fine-Tuning BERT
- Fine-tune the pretrained BERT model on the task of identifying suicidal posts.
- Add a classification layer on top of BERT and train the model using the labeled dataset.
- Use techniques like cross-entropy loss and gradient descent optimisation to update the model's weights.

### 6.3 Model Evaluation and Optimisation
- Evaluate the performance of the fine-tuned BERT model on the validation set, calculating metrics such as accuracy, precision, recall, and F1-score.
- Utilise Optuna Bayesian hyperparameter tuning to search for the optimal set of hyperparameters that maximises the model's performance.

### 6.4 Interpretability
- Explore techniques such as attention visualisation or gradient-based methods (e.g., Integrated Gradients) to gain insights into the model's decision-making process.

### 6.5 Evaluation
- Evaluate the performance of the Multinomial Naive Bayes classifier using appropriate metrics such as accuracy, precision, recall, and F1-score. These metrics provide an assessment of the classifier's effectiveness in correctly predicting suicidal posts and non-suicidal posts.
- Calculate accuracy, which measures the overall correctness of the classifier's predictions.
- Compute precision, which quantifies the proportion of suicidal posts correctly identified out of all posts predicted as suicidal.
- Calculate recall, also known as sensitivity or true positive rate, which measures the proportion of actual suicidal posts correctly identified by the classifier.
- Compute the F1-score, which combines precision and recall into a single metric, providing a balanced evaluation of the classifier's performance.##

## Sentiment Analysis

### 7.1 Sentiment Scoring
- Perform sentiment analysis on the posts using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool.
- Assign sentiment scores to each post, capturing the overall emotional tone expressed.

### 7.2 Sentiment Visualisation
- Visualise sentiment trends and sentiments associated with specific topics or clusters.
- Generate word clouds to highlight key words based on their frequency and importance within the dataset.
- Analyse sentiment patterns to gain insights into the emotional context of posts.


## Active Learning and Evaluation

### 8.1 Active Learning and Manual Review
- Implement an active learning approach to improve the accuracy of the model.
- Manually review and label a small set of posts flagged as potentially suicidal to iteratively train and refine the model.
- Continuously review and iterate the system based on the uncertain predictions within the dataset.

### 8.2 Community Cross-over Evaluation
- Evaluate the degree of crossover between users in different mental health communities, particularly those identified as posting potentially suicidal content.
- Gain insights into correlations or patterns among users engaging in multiple subreddits.

### 8.3 Assess Contribution to Clinical Research
- Analyse the insights extracted from the identification of potential suicidal ideation and the overall analysis of user well-being.
- Investigate how these insights can contribute to clinical research, informing the development of interventions and support systems for individuals experiencing mental health distress.
- Assess the potential impact of the project's findings on clinical practices and strategies for user well-being.

## Ethics, Conclusion, and Further Work

### 9.1 Ethics Consideration
- Address ethical concerns surrounding privacy and the potential misuse of data in the study. Ensure the project contributes positively to mental health awareness and support.
- Discuss the responsible handling of sensitive data and strike a balance between privacy and the need for research.
- Evaluate the potential for bias in the models and the implications of potential false positives or negatives in the identification of suicidal posts.

### 9.2 Conclusion
- Summarise the findings of the study, focusing on the primary outcomes and insights regarding the identification and analysis of suicidal posts in mental health subreddits.
- Discuss the significance of these findings in the context of mental health support on platforms like Reddit.
- Reflect on the limitations of the study, highlighting the challenges encountered during data collection, model training, and performance evaluation.

### 9.3 Further Work
- Propose potential future studies and enhancements to the methodology, including exploring other social media platforms, incorporating user interactions or temporal dynamics, or integrating external data sources for a more accurate identification of suicidal posts.
- Discuss how the current work could be expanded to include additional mental health topics or to incorporate other linguistic or socio-demographic factors.
- Suggest potential applications of the study's findings in designing interventions, developing policy, or informing public awareness campaigns.
