# Assignment 1: 
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
During pipeline execution, the merge step initially failed due to memory limitations (out-of-memory error) caused by the high dimensionality of TF-IDF features and SBERT embeddings. To resolve this while maintaining the lab structure, I reduced the TF-IDF feature size using max_features and limited the embedding dimensions. Additionally, I applied sampling during the split step to reduce dataset size. These adjustments allowed the pipeline to run successfully without changing the overall design or logic of the lab.

---

# Assignment 2: Model Training & Automation with Azure
This project focuses on building a complete machine learning workflow using Azure Machine Learning, starting from feature-engineered data in Lab 4 all the way to training, tuning, deploying, and testing a model in a realistic production-like environment.

Before starting the modeling phase, I first made sure that the dataset was fully consistent and usable. I verified that all dataset splits (train, validation, and test) were passed through the exact same feature engineering pipeline, so the model would see the same structure during training and evaluation. I also ensured that the overall rating column was preserved, since it is used to generate the target label.

After that, I updated the dataset splitting strategy based on the new assignment requirements. Instead of the traditional three splits, I created four: train (60%), validation (15%), test (15%), and a deployment split (10%). The deployment split was especially important because it represents the most recent data, allowing me to simulate real-world conditions where the model is exposed to new and slightly different data. This helps evaluate how the model behaves under data drift.

To keep everything organized, I worked in a separate branch and structured the repository into clear folders. The src folder contains all scripts such as training and inference, jobs includes Azure ML job configurations, and env defines the required environment. This structure made it easier to connect everything with Azure ML.

Next, I registered all four datasets (train, validation, test, and deployment) as Azure ML Data Assets. Instead of referencing individual files, I pointed to the full folders containing the data.parquet files. This ensures reproducibility and allows Azure ML to track exactly which data was used for each training run.

The main part of the assignment was implementing the training script. In train.py, I converted the overall rating into a binary label, where ratings of 4 and above are considered positive. For feature construction, I reused the engineered features from Lab 4 and selected only numeric columns, while dropping the label and filling missing values with zeros. No new feature engineering was done here, as required.

For the model, I chose Logistic Regression. It is simple, fast, and works well with high-dimensional text features such as TF-IDF. Since the workflow is automated through pipelines, having a fast model was important to reduce training time and keep the process efficient.

During training, I evaluated the model on train, validation, and test sets, and logged all important metrics using MLflow. These included accuracy, precision, recall, F1-score, AUC, and training runtime. This allowed me to track performance across different runs directly in Azure ML.

Once the script was ready, I created a training job configuration file (train_job.yml) to run the model on Azure ML compute. I first tested this manually using the CLI to make sure everything worked correctly, including metric logging and model artifact generation.

After confirming that the training job worked, I automated the process using Azure DevOps. I created a pipeline that triggers a training job every time I push code to the repository. This turned the workflow into a fully automated system where training, logging, and model creation happen without manual intervention.

To improve performance, I implemented a hyperparameter sweep job. Instead of manually trying different values, Azure ML automatically tested multiple combinations of parameters such as C and max_iter. After analyzing the results, I selected the best configuration (C = 0.557 and max_iter = 300) and retrained the final model using these values.

The optimized model achieved a test accuracy of around 82.2%. The recall was significantly higher than precision, which indicates that the model is very good at identifying positive reviews, although it may classify some negative reviews as positive. Overall, the model performed well and trained very quickly, making it suitable for automated pipelines.

After training, I registered the model in the Azure ML Model Registry. This allows version tracking and ensures that the model can be reused or deployed later while maintaining full traceability of how it was created.

The next step was deployment. I created a scoring script (score.py) that loads the model and handles prediction requests, along with an inference environment file defining all required dependencies. Then, I deployed the model as a managed online endpoint in Azure ML, which exposes it as a REST API.

To test the deployed model, I used the deployment dataset. I created a script (invoke_endpoint.py) that loads the dataset, builds the same feature matrix used during training, and sends it to the endpoint using an HTTP request. The endpoint returns predictions, which I compared with the true labels to calculate accuracy.

The final deployment accuracy was approximately 83.85%, which is slightly higher than the test accuracy. This indicates that the model generalizes well to unseen data and performs reliably in a production-like scenario.

Finally, after completing the testing, I deleted the endpoint to avoid unnecessary resource usage and compute costs, as recommended in the assignment.

Overall, this project successfully demonstrates a complete MLOps workflow, including training, experimentation, automation, deployment, and evaluation. The Logistic Regression model proved to be efficient, stable, and well-suited for this pipeline.
