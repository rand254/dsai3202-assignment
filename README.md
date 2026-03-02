In this lab, we moved from raw review text to machine learning features using Azure Machine Learning pipelines and the Feature Store. Even though the dataset was already curated (Gold layer – features_v1), the review text cannot be used directly by models. Text must first be converted into numerical features that capture information like length, sentiment, and semantic meaning.

First, I opened Azure Databricks and loaded the curated Gold dataset. I verified the number of rows and columns and checked the schema to make sure everything had the correct data type. The reviewText column was stored as string, the overall rating column was numeric, and identifier columns like asin and reviewerID were present. I also checked for missing or empty values in important columns like reviewText and overall to make sure the dataset was clean enough for feature engineering.

I created two main visualizations. The first one was the rating distribution. This helps us understand how balanced the ratings are (for example, if most reviews are 5 stars). This matters because imbalance can affect model training. The second visualization was the review length distribution (number of words or characters). This is important because very short reviews may not contain enough useful information, and extremely long reviews may affect computational performance. These plots helped validate that the data is suitable for feature extraction.

Since the original dataset contains over 20 million reviews, I created a sampled version to make the pipeline computationally feasible. I sampled around 300,000 reviews using a fixed seed for reproducibility. To reduce drift risk (because language evolves over time), a better approach is to ensure the sample represents different time periods instead of taking only the earliest records. This helps keep the dataset more robust against temporal language drift. After sampling, I compared the rating distribution between the original and sampled datasets to make sure they were roughly similar. Then I wrote the sampled dataset back to the Gold layer as features_v1_sampled without modifying the original dataset.

Next, I moved to Azure ML. I created a new GitHub branch called lab4_feature_engineering and configured Azure ML CLI. I created a compute cluster to run the pipeline. Then I configured Azure ML access to the curated data lake container by registering a datastore using the storage account key.

After that, I registered the sampled dataset as an Azure ML Data Asset. This allows the pipeline to access the data in a structured and versioned way.

I then created a Feature Store entity called AmazonReview. The entity uses asin and reviewerID as index columns, which uniquely identify each review. These keys are used later to join all generated features.

Feature engineering was implemented using Azure ML command components. Each component performs exactly one clear task and runs on Azure ML compute.

The first component splits the dataset into train (70%), validation (15%), and test (15%). This step is very important to prevent data leakage. The training split is used to fit feature transformers like TF-IDF, while validation and test splits remain unseen.

The second component normalizes the text. It lowercases all text, removes punctuation, replaces URLs and numbers using regex, trims whitespace, and removes very short reviews. This ensures consistency and prevents noisy text from affecting feature quality.

The third component creates review length features. It calculates the number of words and number of characters in each review. These features capture structural information about the text and can sometimes correlate with rating or sentiment strength.

The fourth component extracts sentiment features using VADER or TextBlob. It produces positive, negative, neutral, and compound sentiment scores. These features capture emotional tone and polarity of the review, which is highly useful in review-based prediction tasks.

The fifth component generates TF-IDF features using scikit-learn’s TfidfVectorizer. It uses stop word removal and supports n-grams. The vectorizer is fitted only on the training data and then applied to validation and test splits. This avoids leakage and ensures fair evaluation. TF-IDF represents word importance in a high-dimensional feature space.

The sixth component extracts semantic embeddings using transformer-based models such as SBERT. Unlike TF-IDF, embeddings capture contextual meaning and relationships between words. This allows the model to understand semantic similarity rather than just word frequency.

After all features were generated, I created a merge component. This component joins all feature outputs (length, sentiment, TF-IDF, embeddings) using asin and reviewerID as keys. The final output is a single feature-enriched dataset stored as Parquet.

Then I created an Azure ML pipeline that connects all components together. The pipeline reads the sampled dataset, splits it, normalizes text, generates all features, merges them, and produces the final dataset. The pipeline runs entirely on Azure ML compute. After submission, I monitored the job in Azure ML Studio and checked logs to ensure successful execution.

Finally, I registered the merged dataset as a Feature Set in the Azure ML Feature Store. The FeatureSetSpec file defines the schema, source location, index columns, and feature data types. The feature set is versioned, which ensures reproducibility and consistency for future modeling labs.

All components, pipeline definitions, and feature store assets were committed to GitHub in the lab4_feature_engineering branch.

Overall, this lab demonstrates how to build a reproducible and scalable text feature engineering workflow using Azure ML. It ensures no data leakage, supports versioning, and prepares features that can be reused in downstream machine learning tasks.
