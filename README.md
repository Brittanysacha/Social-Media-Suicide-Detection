# Social-Media-Suicide-Detection
Leveraging NLP to identify suicidal ideation on social media, extracting insights into user well-being for application in clinical research.

### Subreddit Selection
The selected subreddits encompass a wide range of mental health conditions, from anxiety and depression to more specific conditions like bipolar disorder, schizophrenia, and eating disorders. This broad selection allows for a comprehensive exploration of language patterns associated with diverse mental health experiences. Moreover, these subreddits are known for their active communities and open discussions about personal struggles, which makes them valuable sources of authentic, self-reported data on mental health, including instances of suicidal ideation.

### Classifacation Catergories
In the endeavor to analyse mental health discussions on Reddit, a machine learning model will be developed to classify posts into five primary categories. These categories aim to capture the range of experiences and emotions expressed by users in these communities, from times of severe crisis to the stages of recovery and management, as well as the offering of advice or support. Here's a breakdown of these categories:

- **1. Immediate Crisis:** This category covers Reddit posts where the poster is experiencing an immediate crisis situation, implying a need for immediate intervention. 
- **2. Mental Distress:** This category comprises Reddit posts that indicate severe, ongoing mental distress, such as feelings of depression, anxiety, or non-immediate suicidal thoughts. 
- **3. Struggle/Challenge:** This category should capture Reddit posts in which the poster discusses ongoing struggles or challenges causing distress. 
- **4. Advice/Support:** This category pertains to Reddit posts where the poster is actively seeking advice, support, or guidance from the community. 
-- **5. Recovery/Management:** This category includes Reddit posts where the poster is actively engaged in recovery or managing their situation. 

### Selection of the BERT Model

For this project, BERT is selected over other models because of its ability to analyse language bidirectionally. This means it can capture both the preceding and following context of a word, allowing it to understand the complexities and subtleties within mental health discussions on Reddit. BERT's transformer architecture, which employs self-attention mechanisms, surpasses conventional machine learning models like logistic regression or random forest. This is because transformers outperform traditional models by being able to process large amounts of data in parallel, efficiently handle long-range dependencies in text, and capture the relationships between all words in a sentence simultaneously. This architecture enables BERT to extract deeper and more meaningful insights from mental health discussions on Reddit. Additionally, BERT's high level of understanding of text semantics, especially in user-generated content that often includes slang, acronyms, and colloquial language, significantly aids the task of classifying and analysing mental health posts for this research.