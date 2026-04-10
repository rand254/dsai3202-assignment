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


To understand how different feature combinations affect model performance, I tested multiple configurations using the available features in the dataset.

Instead of using SBERT embeddings, I focused on the features generated in Lab 4, which include sentiment scores, TF-IDF vectors, and additional metadata.

The experiments were:

Run 1 – Sentiment features only
Run 2 – Sentiment + TF-IDF
Run 3 – All features (Sentiment + TF-IDF + metadata)

The best performance was achieved using Run 3 (All Features), with a test accuracy of 82.22%.

This configuration performed better because it combined multiple types of information:

Sentiment captures emotional tone
TF-IDF captures important keywords
Metadata (such as length and price) adds structural context

By combining these features, the model was able to make more accurate and balanced predictions compared to using a single feature type.

The next step was deployment. I created a scoring script (score.py) that loads the model and handles prediction requests, along with an inference environment file defining all required dependencies. Then, I deployed the model as a managed online endpoint in Azure ML, which exposes it as a REST API.

To test the deployed model, I used the deployment dataset. I created a script (invoke_endpoint.py) that loads the dataset, builds the same feature matrix used during training, and sends it to the endpoint using an HTTP request. The endpoint returns predictions, which I compared with the true labels to calculate accuracy.

The final deployment accuracy was approximately 83.85%, which is slightly higher than the test accuracy. This indicates that the model generalizes well to unseen data and performs reliably in a production-like scenario.

Finally, after completing the testing, I deleted the endpoint to avoid unnecessary resource usage and compute costs, as recommended in the assignment.

Overall, this project successfully demonstrates a complete MLOps workflow, including training, experimentation, automation, deployment, and evaluation. The Logistic Regression model proved to be efficient, stable, and well-suited for this pipeline.
