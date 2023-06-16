# Project Outline: Identifying and Understanding Posts in Mental Health Subreddits using Natural Language Processing

** Research Question** How can natural language processing and machine learning techniques be utilised to classify and analyse posts in mental health on Reddit? What insights can be extracted from social media to contribute to understanding mental health discussions and potentially aiding clinical research on user well-being?

## Introduction
The exponential growth of online platforms has triggered a parallel increase in the volume of user-generated content, unveiling unique challenges and opportunities for understanding human behavior. This is particularly the case for mental health communities, where the anonymity of the internet allows individuals to share their deepest fears and darkest thoughts. Such candid insights into mental health challenges, while providing invaluable support networks for the individuals involved, also possess the potential to serve as an untapped resource for clinical research and therapeutic intervention.

Therefore, the objective of this project is to conduct an semi-supervised study on Reddit data from various mental health subreddits. The primary aim is to identify potential user-wellbeing based on posts data using natural language processing techniques, with a specific focus on incorporating BERT (Bidirectional Encoder Representations from Transformers). Through an in-depth examination of posts, assessing their semantics, context, and underlying sentiment, a BERT model will be tested to see if it can help recognise linguistic patterns that indicate whether an individual is in crisis, struggling, working on recovery or management, or seeking support or advice. 

Given that individuals may not always be forthcoming with a doctor or a therapist due to fears of being sectioned, medicated, or for child and youth reported to a parent or guardian. Therefore, this research could yield meaningful implications for clinical research by identifying linguistic patterns or undertones within text that may also be observed in clinical practice and clinical session notes. By identifying trends and patterns related to user-wellbeing in social media discussions, they is further opportunity to help inform more responsive and tailored intervention strategies, bolstering our capacity to support individuals facing mental health challenges in the digital age.

## Table of Contents
- [Data Gathering and Pre-processing](#data-gathering-and-pre-processing)
  - [1.1 Reddit API connnection - PRAW] 
  - [2.1 Data Scraping](#11-data-scraping)

- [Database Initialisation](#database-initialisation)
  - [2.1 Gather Connection Details](#21-gather-connection-details)
  - [2.2 Establish a Connection](#22-establish-a-connection)

- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [3.1 General Analysis](#31-general-analysis)
  - [3.2 Cross-community Analysis](#32-cross-community-analysis) *********

- [Sentiment Analysis](#sentiment-analysis)
  - [6.1 Sentiment Scoring](#61-sentiment-scoring)
  - [6.2 Sentiment Visualisation](#62-sentiment-visualisation)

- [Data Cleaning](#data-cleaning)
  - [5.1 Spellcheck and Slang Dictionary Creation]
  - [5.2 Post Normalisation]
  - [5.3 Tokenisation]

- [Natural Language Processing](#natural-language-processing)
  - [4.1 Seed Set Annotation](#41-seed-set-annotation)
  - [4.2 BERT Pretraining](#42-bert-pretraining)
  - [4.3 Fine-Tuning BERT](#43-fine-tuning-bert)
  - [4.4 Input Encoding](#44-input-encoding)
  - [4.5 Model Evaluation and Optimisation](#45-model-evaluation-and-optimisation)
  - [4.6 Interpretability](#46-interpretability)

- [Evaluation](#evaluation)
  - [8.1 Model Performance Evaluation](#81-model-performance-evaluation)
  - [8.2 Active Learning and Manual Review](#82-active-learning-and-manual-review)
  - [8.3 Community Cross-over Evaluation](#83-community-cross-over-evaluation)
  - [8.4 Assess Contribution to Clinical Research](#84-assess-contribution-to-clinical-research)

- [Ethics, Conclusion, and Further Work](#ethics-conclusion-and-further-work)
  - [9.1 Ethics Consideration](#91-ethics-consideration)
  - [9.2 Conclusion](#92-conclusion)
  - [9.3 Further Work](#93-further-work)

## Data Gathering and Pre-processing

### 1.1 Data Scraping
- Utilise Python's PRAW (Python Reddit API Wrapper) to scrape data from chosen mental health subreddits, including depression, bipolar, schizophrenia, suicide_watch, and others.
- Collect posts and associated metadata from the year 2022.
- Data extraction will be performed for the months of January, April, July, and October to capture diverse user mental states influenced by different seasons.

### 1.2 Data Cleaning
- Clean the extracted data by removing redundant or unnecessary information, handling missing values, and normalising text (e.g., converting to lowercase, removing punctuation and special characters).

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

## Exploratory Data Analysis (EDA)

### 3.1 General Analysis
- Conduct an exploratory analysis of the data, including basic statistics on the number of posts, number of users, most active users, and other relevant features.

### 3.2 Cross-community Analysis
- Identify users who post in multiple subreddits to assess the degree of crossover in the mental health communities.

## Natural Language Processing

### 4.1 Seed Set Annotation
- Manually review a small subset of the scraped data and annotate posts as "suicidal" or "non-suicidal" based on their content.
- Use the annotated seed set as the initial labeled dataset.
- Aiming to manually review 1,000 per subreddit community (start with 4 communities - allow for 1-2 days for this)

### 4.2 BERT Pretraining
- Leverage a pretrained BERT model (e.g., BERT base or BERT large) as a starting point for the fine-tuning process.
- Utilise BERT's knowledge of language patterns and semantics.

### 4.3 Fine-Tuning BERT
- Fine-tune the pretrained BERT model on the task of identifying suicidal posts.
- Add a classification layer on top of BERT and train the model using the labeled dataset.
- Use techniques like cross-entropy loss and gradient descent optimisation to update the model's weights.

### 4.4 Input Encoding
- Tokenise the text data using the WordPiece tokeniser or a similar approach.
- Convert the tokenised input into BERT's input format, including token IDs, attention masks, and segment IDs.

### 4.5 Model Evaluation and Optimisation
- Evaluate the performance of the fine-tuned BERT model on the validation set, calculating metrics such as accuracy, precision, recall, and F1-score.
- Utilise Optuna Bayesian hyperparameter tuning to search for the optimal set of hyperparameters that maximises the model's performance.

### 4.6 Interpretability
- Explore techniques such as attention visualisation or gradient-based methods (e.g., Integrated Gradients) to gain insights into the model's decision-making process.

## Multinomial Naive Bayes

### 5.1 Feature Extraction
- Extract relevant features from the pre-processed text data, such as bag-of-words representations or TF-IDF features.
    - Train Word2Vec models on the pre-processed text data to obtain word embeddings.
    - Map words or phrases from the vocabulary onto vectors of real numbers, enabling numerical representation of the linguistic data.

### 5.2 TF-IDF Weighted Word2Vec
- Apply TF-IDF (Term Frequency-Inverse Document Frequency) approach coupled with Word2Vec to establish word weights in the document.
- Compute TF-IDF weighted Word2Vec representations for each post, capturing both word importance and contextual relevance.

### 5.3 Training and Classification
- Train a Multinomial Naive Bayes classifier on the extracted features.
- Use Optuna Bayesian hyperparameter tuning to search for the optimal set of hyperparameters that maximises the classifier's performance. This involves exploring different hyperparameter configurations and evaluating their impact on the classifier's metrics.

### 5.4 Evaluation
- Evaluate the performance of the Multinomial Naive Bayes classifier using appropriate metrics such as accuracy, precision, recall, and F1-score. These metrics provide an assessment of the classifier's effectiveness in correctly predicting suicidal posts and non-suicidal posts.
- Calculate accuracy, which measures the overall correctness of the classifier's predictions.
- Compute precision, which quantifies the proportion of suicidal posts correctly identified out of all posts predicted as suicidal.
- Calculate recall, also known as sensitivity or true positive rate, which measures the proportion of actual suicidal posts correctly identified by the classifier.
- Compute the F1-score, which combines precision and recall into a single metric, providing a balanced evaluation of the classifier's performance.##

## Sentiment Analysis

### 6.1 Sentiment Scoring
- Perform sentiment analysis on the posts using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool.
- Assign sentiment scores to each post, capturing the overall emotional tone expressed.

### 6.2 Sentiment Visualisation
- Visualise sentiment trends and sentiments associated with specific topics or clusters.
- Generate word clouds to highlight key words based on their frequency and importance within the dataset.
- Analyse sentiment patterns to gain insights into the emotional context of posts.

## Suicidal Post Detection

### 7.1 Suicidal Ideation Identification
- Integrate the findings from Word Embeddings, TF-IDF Weighted Word2Vec, BERT, Multinomial Naive Bayes, and sentiment analysis to develop a comprehensive system that identifies potential suicidal ideation in posts.
- Combine insights from these techniques to create a scoring or probability system that quantifies the likelihood of suicidal ideation for each post.
- Higher scores or probabilities suggest higher-risk posts requiring further examination.

## Evaluation

### 8.1 Model Performance Evaluation
- Evaluate the performance of the developed system for identifying suicidal posts using appropriate metrics such as accuracy, precision, recall, and F1-score.
- Use a validation or test set to assess the effectiveness of the system.

### 8.2 Active Learning and Manual Review
- Implement an active learning approach to improve the accuracy of the model.
- Manually review and label a small set of posts flagged as potentially suicidal to iteratively train and refine the model.
- Continuously review and iterate the system based on the uncertain predictions within the dataset.

### 8.3 Community Cross-over Evaluation
- Evaluate the degree of crossover between users in different mental health communities, particularly those identified as posting potentially suicidal content.
- Gain insights into correlations or patterns among users engaging in multiple subreddits.

### 8.4 Assess Contribution to Clinical Research
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
