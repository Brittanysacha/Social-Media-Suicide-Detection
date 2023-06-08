# Project Outline: Identifying and Understanding Suicidal Posts in Mental Health Subreddits using Natural Language Processing

1. Research Question: How can natural language processing and machine learning enhance the identification of potential suicidal posts in mental health subreddits on Reddit, and what insights can be gained from their analysis?

## Introduction
The objective of this project is to conduct an unsupervised study on Reddit data from various mental health subreddits. The primary aim is to identify potential suicidal posts using natural language processing techniques, with a specific focus on incorporating BERT (Bidirectional Encoder Representations from Transformers) while also leveraging Word Embeddings, TF-IDF Weighted Word2Vec, and Multinomial Naive Bayes. The models will be further optimized using Optuna Bayesian hyperparameter tuning.

## Data Gathering and Pre-processing

### 1.1 Data Scraping
- Utilise Python's PRAW (Python Reddit API Wrapper) to scrape data from chosen mental health subreddits, including depression, bipolar, schizophrenia, suicide_watch, and others.
- Collect posts and associated metadata from the year 2022.
- Data extraction will be performed for the months of January, April, July, and October to capture diverse user mental states influenced by different seasons.

### 1.2 Data Cleaning
- Clean the extracted data by removing redundant or unnecessary information, handling missing values, and normalising text (e.g., converting to lowercase, removing punctuation and special characters).

## Exploratory Data Analysis (EDA)

### 2.1 General Analysis
- Conduct an exploratory analysis of the data, including basic statistics on the number of posts, number of users, most active users, and other relevant features.

### 2.2 Cross-community Analysis
- Identify users who post in multiple subreddits to assess the degree of crossover in the mental health communities.

## Natural Language Processing

### 3.1 Seed Set Annotation
- Manually review a small subset of the scraped data and annotate posts as "suicidal" or "non-suicidal" based on their content.
- Use the annotated seed set as the initial labeled dataset.
- Aiming to manually review 1,000 per subreddit community (start with 4 communities - allow for 1-2 days for this)

### 3.2 BERT Pretraining
- Leverage a pretrained BERT model (e.g., BERT base or BERT large) as a starting point for the fine-tuning process.
- Utilise BERT's knowledge of language patterns and semantics.

### 3.3 Fine-Tuning BERT
- Fine-tune the pretrained BERT model on the task of identifying suicidal posts.
- Add a classification layer on top of BERT and train the model using the labeled dataset.
- Use techniques like cross-entropy loss and gradient descent optimization to update the model's weights.

### 3.4 Input Encoding
- Tokenise the text data using the WordPiece tokenizer or a similar approach.
- Convert the tokenised input into BERT's input format, including token IDs, attention masks, and segment IDs.

### 3.5 Model Evaluation and Optimization
- Evaluate the performance of the fine-tuned BERT model on the validation set, calculating metrics such as accuracy, precision, recall, and F1-score.
- Utilise Optuna Bayesian hyperparameter tuning to search for the optimal set of hyperparameters that maximises the model's performance.

### 3.6 Interpretability
- Explore techniques such as attention visualisation or gradient-based methods (e.g., Integrated Gradients) to gain insights into the model's decision-making process.

## Multinomial Naive Bayes

### 4.1 Feature Extraction
- Extract relevant features from the pre-processed text data, such as bag-of-words representations or TF-IDF features.
    - Train Word2Vec models on the pre-processed text data to obtain word embeddings.
    - Map words or phrases from the vocabulary onto vectors of real numbers, enabling numerical representation of the linguistic data.

### 3.3 TF-IDF Weighted Word2Vec
- Apply TF-IDF (Term Frequency-Inverse Document Frequency) approach coupled with Word2Vec to establish word weights in the document.
- Compute TF-IDF weighted Word2Vec representations for each post, capturing both word importance and contextual relevance.

### 4.2 Training and Classification
- Train a Multinomial Naive Bayes classifier on the extracted features.
- Use Optuna Bayesian hyperparameter tuning to search for the optimal set of hyperparameters that maximizes the classifier's performance. This involves exploring different hyperparameter configurations and evaluating their impact on the classifier's metrics.

### 4.3 Evaluation
- Evaluate the performance of the Multinomial Naive Bayes classifier using appropriate metrics such as accuracy, precision, recall, and F1-score. These metrics provide an assessment of the classifier's effectiveness in correctly predicting suicidal posts and non-suicidal posts.
- Calculate accuracy, which measures the overall correctness of the classifier's predictions.
- Compute precision, which quantifies the proportion of suicidal posts correctly identified out of all posts predicted as suicidal.
- Calculate recall, also known as sensitivity or true positive rate, which measures the proportion of actual suicidal posts correctly identified by the classifier.
- Compute the F1-score, which combines precision and recall into a single metric, providing a balanced evaluation of the classifier's performance.##

## Sentiment Analysis

### 5.1 Sentiment Scoring
- Perform sentiment analysis on the posts using the VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool.
- Assign sentiment scores to each post, capturing the overall emotional tone expressed.

### 5.2 Sentiment Visualization
- Visualise sentiment trends and sentiments associated with specific topics or clusters.
- Generate word clouds to highlight key words based on their frequency and importance within the dataset.
- Analyse sentiment patterns to gain insights into the emotional context of posts.

## Suicidal Post Detection

### 6.1 Suicidal Ideation Identification
- Integrate the findings from Word Embeddings, TF-IDF Weighted Word2Vec, BERT, Multinomial Naive Bayes, and sentiment analysis to develop a comprehensive system that identifies potential suicidal ideation in posts.
- Combine insights from these techniques to create a scoring or probability system that quantifies the likelihood of suicidal ideation for each post.
- Higher scores or probabilities suggest higher-risk posts requiring further examination.

## Evaluation

### 7.1 Model Performance Evaluation
- Evaluate the performance of the developed system for identifying suicidal posts using appropriate metrics such as accuracy, precision, recall, and F1-score.
- Use a validation or test set to assess the effectiveness of the system.

### 7.2 Active Learning and Manual Review
- Implement an active learning approach to improve the accuracy of the model.
- Manually review and label a small set of posts flagged as potentially suicidal to iteratively train and refine the model.
- Continuously review and iterate the system based on the uncertain predictions within the dataset.

### 7.3 Community Cross-over Evaluation
- Evaluate the degree of crossover between users in different mental health communities, particularly those identified as posting potentially suicidal content.
- Gain insights into correlations or patterns among users engaging in multiple subreddits.
