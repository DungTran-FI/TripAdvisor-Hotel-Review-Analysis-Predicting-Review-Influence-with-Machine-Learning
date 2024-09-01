# TripAdvisor Hotel Review Analysis: Predicting Reviewer Inlfuence with Machine LearningAnalysis

### The complete analysis report is available in [Review Author Influence Report](https://github.com/DungTran-FI/TripAdvisor-Hotel-Review-Analysis-Predicting-Review-Influence-with-Machine-Learning/blob/main/Review%20Author%20Influence%20-%20Report.pdf)

## Project Overview

This project focuses on predicting the influence of review authors in both real-life and online environments using Weka machine learning toolkit. The goal is to identify influential reviewers who can significantly impact business outcomes, enabling targeted marketing, customer engagement strategies, and improved business decision-making.

## Project Summary

### 1. Predictive Model for Review Author Influence: Real Life
#### 1.1 Data Processing
The data for predicting real-life influence was processed using Python and Weka. Key steps included converting numeric variables to nominal and handling class imbalance through the SMOTE technique.

#### 1.2 Model Training & Testing
The models chosen to be trained are OneR, Decision Tree J48, Random Forest, Cost Sensitive Classification (J48/Random Forest/OneR as the base classifier) using the original imbalanced data and rebalanced data using SMOTE methods.
Model training is conducted through 3 rounds:

- **Round 1**: Initial models were trained on imbalanced data without SMOTE, including all variables.
- **Round 2**: SMOTE was applied to balance the dataset, and all variables were included.
- **Round 3**: A more refined model was developed by applying SMOTE and excluding the `Author_num_reviews` attribute through attribute selection.

#### 2.3 Conclusion
The best-performing model for predicting real-life influence was identified based on its ability to accurately predict the minority class (highly influential authors) while minimizing misclassification costs. Achieved 85.90% accuracy.

### 3. Predictive Model for Review Author Influence: Online Environment
#### 3.1 Data Processing
Similar to the real-life model, the data was preprocessed by converting variables to nominal and balancing the dataset with SMOTE to ensure accurate predictions of online influence.

#### 3.2 Model Training & Testing
The models chosen to be trained are Cost Sensitive Classifier using different classifiers as the base classifiers, such as OneR, Decision Tree J48, Random Forest, SMO, Simple Logistics and Bagging. 
Model training is conudcted through 2 rounds:
- **Round 1**: All variables were included, and the dataset was balanced with SMOTE.
- **Round 2**: The model was further refined by selecting the most significant attributes and applying SMOTE.

#### 3.3 Conclusion
The best model for predicting online influence was selected based on its precision, recall, and cost-effectiveness in identifying authors with significant online influence. Acheived 97.92% accuracy.

### 4. Business Suggestions
The project concludes with strategic business recommendations on how to utilize the identified influential reviewers. Suggestions include creating targeted marketing campaigns, enhancing customer engagement through personalized experiences, and forming partnerships with key influencers to boost brand visibility and drive revenue growth.

## Skills Utilized

### 1. **Data Processing**
   - **Data Transformation**: Utilized Weka to preprocess data, converting numeric values into nominal variables and creating new binary features using expressions. This step was crucial for preparing the dataset for effective model training.
   - **Imbalanced Data Handling**: Applied SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance, ensuring that the models could better predict minority classes, such as highly influential authors.

### 2. **Machine Learning Modeling**
   - **Model Selection and Training**: Trained various classifiers including OneR, J48 Decision Tree, Random Forest, and Cost Sensitive Classifiers using Weka. Models were chosen based on their suitability for handling imbalanced datasets and their ability to prioritize correct identification of influential reviewers.
   - **Cost Sensitive Learning**: Incorporated cost matrices to penalize misclassifications, particularly focusing on reducing false negatives, which are critical in business contexts where missing an influential reviewer could lead to lost opportunities.
   - **Model Evaluation**: Used cross-validation, accuracy metrics, precision, recall, F-measure, ROC Area, and total cost to evaluate model performance, ensuring the best model was selected for each scenario.

### 3. **Business Strategy Development**
   - **Insight Generation**: The project generated actionable insights for different business teams (Marketing, Customer Service, Operations, Revenue Management) to enhance their strategies by focusing on influential reviewers.
   - **Strategy Formulation**: Developed targeted business strategies such as engagement and recognition programs, influencer collaborations, and loyalty and referral programs based on the identified influential reviewers.

## Why This Project is Helpful for Business

1. **Targeted Marketing**: By identifying influential reviewers, the marketing team can tailor campaigns to leverage these individuals' reach, optimizing advertising spend and improving ROI.
   
2. **Customer Engagement**: Prioritizing interactions with highly influential reviewers allows businesses to enhance personalized communication and deliver exceptional experiences, increasing customer satisfaction and loyalty.

3. **Operational Efficiency**: Understanding the preferences of influential reviewers enables operations teams to adjust amenities and services to better meet customer expectations, improving overall guest satisfaction.

4. **Revenue Growth**: The predictive models help in guiding dynamic pricing, package offerings, and inventory allocation, which can lead to optimized revenue management strategies and increased profitability.

## Getting Started

### Prerequisites
- **Weka**: The project uses Weka for data preprocessing, model training, and evaluation.
  
- **Dataset**: The dataset concerns TripAdvisor hotel reviews
    - Data structure: The dataset contains 3,118 rows of instances and 22 attributes.
    - Source of data: The dataset is a small sample extracted from a large open TripAdvisor Review dataset from: [Hotel Review](http://www.cs.cmu.edu/~jiweil/html/hotel-review.html)
 
### Dataset Overview

| Feature                                  | Description                                                                                                                                           |
|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Hotel_id**                              | The ID of the hotel.                                                                                                                                   |
| **num_helpful_vote**                      | The number of helpfulness votes that a review has received. This measures how influential a review is.                                                 |
| **id**                                    | The ID of instances.                                                                                                                                   |
| **via_mobile**                            | Indicates whether a review was compiled on a mobile device (TRUE) or not (FALSE).                                                                      |
| **Revisit**                               | Indicates whether the customer revisited the hotel after writing this review. Values: "Trigger_revisit"; "No_revisit".                                  |
| **Rating_overall**                        | Customer's overall rating of the hotel in the review.                                                                                                  |
| **Rating_service**                        | Customer's rating of the hotel's service in the review.                                                                                                |
| **Rating_cleanliness**                    | Customer's rating of the hotel's cleanliness in the review.                                                                                            |
| **Rating_value**                          | Customer's rating of the hotel's value in the review.                                                                                                  |
| **Rating_location**                       | Customer's rating of the hotel's location in the review.                                                                                               |
| **Rating_sleep_quality**                  | Customer's rating of the hotel's sleep quality in the review.                                                                                          |
| **Rating_rooms**                          | Customer's rating of the hotel's rooms in the review.                                                                                                  |
| **Rating_check_in_front_desk**            | Customer's rating of the hotel's check-in/front desk service in the review.                                                                            |
| **Rating_business_service_(e_g_internet_access)** | Customer's rating of the hotel's business services (e.g., internet access) in the review.                                                            |
| **author_id**                             | ID of the review author/customer.                                                                                                                      |
| **Author_username**                       | Username of the review author.                                                                                                                         |
| **Author_num_cities**                     | The number of cities the review author has visited before. A higher number indicates greater influence of the author in real life.                      |
| **Author_num_helpful_votes**              | The number of helpfulness votes the review author has received across all their reviews. A higher number indicates greater influence in online environments. |
| **Author_num_reviews**                    | The number of reviews that the author has produced on TripAdvisor.                                                                                     |
| **Author_location**                       | The location where the review author is living.                                                                                                        |
| **title**                                 | The title of the review.                                                                                                                               |
| **text**                                  | The text content of the review.                                                                                                                        |



