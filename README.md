# Social-Media-Suicide-Detection
Leveraging NLP to identify suicidal ideation on social media, extracting insights into user well-being for application in clinical research.

## Research Question
How can natural language processing and machine learning techniques be utilised to classify and analyse posts in mental health on Reddit? What insights can be extracted from social media to contribute to understanding mental health discussions and potentially aiding clinical research on user well-being?

## Overview
This project aims to employ natural language processing to scrutinise social media posts concerning mental health to ascertain whether a user is suicidal and in immediate crisis, undergoing mental distress (considering suicide or self harm), experiencing a challenge or struggle (and having difficulties with coping or managing their mental load), working towards recovery or management of their mental health, or seeking advice and support from others. 

## Subreddit Selection
The chosen subreddits cover a wide array of mental health conditions, from anxiety and depression to more specific conditions like bipolar disorder, schizophrenia, and eating disorders. This broad selection enables a comprehensive examination of language patterns associated with diverse mental health experiences. Moreover, these subreddits are renowned for their active communities and open discussions about personal struggles, making them valuable sources of authentic, self-reported data on mental health, including instances of suicidal ideation.

## Classification Categories
To analyse mental health discussions on Reddit, a machine learning model will be built to classify posts into five primary categories. These categories aim to encapsulate the variety of experiences and emotions expressed by users in these communities, from times of severe crisis to the stages of recovery and management, as well as the offering of advice or support. Here's a rundown of these categories:

- **1. Immediate Crisis:** This category covers Reddit posts where the poster is experiencing an immediate crisis situation, suggesting a need for immediate intervention. 
- **2. Mental Distress:** This category comprises Reddit posts that indicate severe, ongoing mental distress, such as feelings of depression, anxiety, or non-immediate suicidal thoughts. 
- **3. Struggle/Challenge:** This category should capture Reddit posts in which the poster discusses ongoing struggles or challenges causing distress. 
- **4. Advice/Support:** This category pertains to Reddit posts where the poster is actively seeking advice, support, or guidance from the community. 
- **5. Recovery/Management:** This category includes Reddit posts where the poster is actively engaged in recovery or managing their situation. 

## Selection of the BERT Model
For this project, BERT has been chosen over other models because of its ability to analyse language bidirectionally. This means it can understand both the preceding and following context of a word, allowing it to grasp the complexities and subtleties within mental health discussions on Reddit. BERT's transformer architecture, which employs self-attention mechanisms, surpasses conventional machine learning models like logistic regression or random forest. This is because transformers outperform traditional models by being able to process large amounts of data in parallel, efficiently handle long-range dependencies in text, and capture the relationships between all words in a sentence simultaneously. This architecture enables BERT to extract deeper and more meaningful insights from mental health discussions on Reddit. Furthermore, BERT's high level of understanding of text semantics, especially in user-generated content that often includes slang, acronyms, and colloquial language, significantly aids the task of classifying and analysing mental health posts for this research.

## Model Development
Initial steps in the Model Development Stage involved pulling 3000 rows from four Mental Health Subreddits using PRAW and exporting them to PgAdmin for future use. An annotation guide was created to classify Reddit posts into the aforementioned categories and an ipywidget was developed to manually assign ground truth labels.

## Data Cleaning and Enhancement
Data cleaning involved the creation of a

 slang dictionary to capture different terms and acronyms frequently used by users in the mental health community. This helps to ensure that language models don't mislabel posts due to a lack of understanding of these terms. 

## Model Performance
BERT's performance improved with additional data, indicating a need for more data in the future. Based on current forecasting, it's estimated that approximately 23,300 rows would be needed to achieve 75% accuracy. However, a significant statistic is the confidence predictability report, which states that among the 55% overall accuracy the model is identifying 35% of cases with 95% certainty where individuals are in either immediate crisis or distress, and in need of additional help.

## Future Work
Future work requires more data and rigorous validation including dual independent screening, stakeholder engagement, and continuous feedback mechanism, with Quality Control checks to ensure ongoing accuracy. This project opens avenues for enhancing patient risk assessment and mental health care by using online discourse in clinical practice to identify individuals in urgent distress who may currently be falling through the cracks.
