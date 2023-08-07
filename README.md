# Machine Learning Pipeline for Mortgage Backed Securities Prepayment Risk

## Description
This project aims to predict loan pre-payment likelihood based on various factors such as Equated Monthly Installment (EMI), total payment, interest amount, monthly income, current principal, and whether the person has ever been delinquent in their loan payments. The goal is to provide insights into the likelihood of pre-payment and assist in making informed decisions related to loan repayment strategies.

## Dataset
The data dictionary provides an overview of the dataset used in this analysis. It describes the 28 columns present in the dataset, including details such as column names, data types, and a brief explanation of each column's meaning. Understanding the data dictionary is crucial for interpreting and analyzing the dataset accurately.

The dataset used for this analysis contains information about borrowers and their loan details, including EMI, total payment, interest amount, monthly income, current principal, and whether they have been delinquent in their loan payments. The dataset is in a tabular format and is provided as a CSV file called LoanExport.csv.
The data is obtained from Freddie Mac's official portal for home loans. The size of the home loan data is (291452 x 28). It contains 291452 data points and 28 columns or parameters denoting different data features. Some of the noteworthy features of the dataset are:

0 CreditScore : Credit score of the client

1 FirstPaymentDate : First payment date of the customer

2 FirstTimeHomebuyer : If the customer is first time home buyer

3 MaturityDate : Maturity date of the customer

4 MSA : Mortgage security amount

5 MIP : Mortgage insurance percentage

6 Units : Number of units

7 Occupancy : Occupancy status at the time the loan

8 OCLTV : Original Combined Loan-to-Value

9 DTI : Debt to income ratio

10 OrigUPB : Original unpaid principal balance

11 LTV : Loan-to-Value

12 OrigInterestRate : Original interest rate

13 Channel : The origination channel used by the party

14 PPM : Prepayment penalty mortgage

15 ProductType : Type of product

16 PropertyState : State in which the property is located

17 PropertyType : Property type

18 PostalCode : Postal code of the property

19 LoanSeqNum : Loan number

20 LoanPurpose : Purpose of the loan

21 OrigLoanTerm : Original term of the loan

22 NumBorrowers : Number of borrowers

23 SellerName : Name of seller

24 ServicerName : Name of the service used

25 EverDelinquent : If the loan was ever delinquent

26 MonthsDelinquent : Months of delinquent

27 MonthsInRepayment : Months in repayment


![Picture1](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/fcb6999a-4cd6-4c04-9058-a8b4a0f7a0e2)

## Importing libraries 

![Screenshot 2023-07-19 172016](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/2f3b35e1-b501-4419-8e50-550db0e48148)

## Pre-processing Steps
The following steps were taken to ensure the data's integrity and quality:
- Removal of duplicated rows: Any duplicated rows were removed to eliminate any redundant data that could potentially skew the analysis.

  ![drop (1)](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/349f07a9-d907-4e4b-a250-5f828dc16454)

- Handling missing values: Missing values were handled by removing rows that contained missing values in the SellerName column.
- Unique value analysis: The unique values present in each column were analyzed to identify any unusual or unexpected values.
- Handling missing and erroneous values: Any rows that contained the value X in the NumBorrowers, PropertyType, MSA, or PPM columns were removed, as these values were likely placeholders for missing or invalid data.


## Feature Engineering
The following steps were taken to enhance the analysis by creating additional columns:
- CreditScoreRange: A new column, CreditScoreRange, was created by categorizing the CreditScore column into four bands based on credit score ranges.
- RepaymentRange: Another new column, RepaymentRange, was created by dividing the MonthsInRepayment column into five bands to represent different repayment periods.

![rtrrt](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/83ce6fbe-670e-4718-a0e6-9d44cf00dca4)

- Label encoding: Categorical columns were encoded using label encoding or ordinal encoding to convert categorical values into numerical representations that can be processed by machine learning models.

  ![twtet](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/13ff6b99-9705-480c-9979-d05215b598a5)

- Ordinal encoding was used for columns with ordinal data, specifically 'OrigInterestRateRange' and 'RepaymentRange'. The encoding was performed using the OrdinalEncoder from scikit-learn, with specified category orders for each column.
- 	Feature Selection: Feature selection is crucial to identify the most relevant features for the model. In this project, we employed the SelectKBest algorithm with the chi-square (chi2) scoring function. The algorithm was used to rank the features based on their correlation with the target variable 'EverDelinquent'. The top 14 features were selected based on their scores, and a dataframe, featureScores, was created to provide insights into the selected features and their corresponding scores.
	
![aefe](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/9fefab1b-34ca-471e-a4d3-760981b5ca68)

![Screenshot 2023-07-19 164939](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/6d4745ea-7c5b-4f8b-a307-e9abd1fc1deb)


## SMOTE
To address the class imbalance problem, the Synthetic Minority Over-sampling Technique (SMOTE) is employed. SMOTE is a widely used technique in machine learning that generates synthetic samples for the minority class to balance the class distribution. By creating synthetic samples, SMOTE helps improve model performance and prevent biased predictions.

![Screenshot 2023-07-17 165531](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/3cc38495-0ad1-4bb4-a83a-738239c3fa68)

It creates synthetic samples by selecting nearest neighbors from the minority class and generating new instances along the line connecting them. This helps balance the class distribution and improve model performance. SMOTE was applied to predict Mortgage Backed Securities prepayment risk, ensuring more accurate predictions.

## Modelling
The model for predicting Mortgage Backed Securities prepayment risk utilizes various machine learning algorithms, including Logistic Regression, Random Forest Classifier, XGBoost Classifier, and K-Nearest Neighbors (KNN). These algorithms classify whether a loan is likely to become delinquent (EverDelinquent = 1) or not (EverDelinquent = 0).

- Logistic Regression: A binary classification algorithm that estimates coefficients for each feature to determine their impact on the target variable. It predicts the probability of delinquency based on these coefficients and classifies the loan accordingly.
- 
![Screenshot 2023-07-17 170558](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/e0a3de8f-5abe-4ac0-8ae4-4152ea832409)
  
- Random Forest Classifier: An ensemble learning method that combines multiple decision trees to make predictions. It creates collections of decision trees by training on random subsets of data and features, improving accuracy and reducing overfitting.
  
  ![Screenshot 2023-07-17 170629](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/74be7537-22a6-43e5-859c-24cb209affba)

- XGBoost Classifier: An ensemble technique based on gradient boosting, which sequentially adds decision trees to the ensemble. It handles missing values, captures feature interactions, and applies regularization to prevent overfitting, making it effective for classification tasks.
  
  ![Screenshot 2023-07-17 170658](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/a9fd4734-e22a-4949-ad27-b952660e6284)

- K-Nearest Neighbors (KNN): A simple yet powerful classification algorithm that assigns a data point to a class based on its k-nearest neighbors in the feature space. KNN does not assume any underlying data distribution and is non-parametric model evaluation.
  
![Screenshot 2023-07-17 170722](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/9bb57abf-1268-4dce-9e11-8288debe2ddd)

## Model Evaluation
The models are evaluated using various performance metrics, including accuracy, cross-validation scores, and ROC AUC. The performance evaluation provides insights into the accuracy and generalization capabilities of each model.

Based on the performance metrics, the XGBoost Classifier is identified as the top-performing model for predicting MBS prepayment risk due to its high accuracy, excellent generalization, and advanced ensemble technique.

## Model Performance
The performance metrics of different models as follows:

![Screenshot 2023-07-17 171157](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/1fa74f94-fd95-40c0-a545-25201bc13133)

All models perform well in predicting Ever Delinquent status of Mortgage Backed Securities. Random forest model show higher cross-validation and accuracy scores, indicating superior performance compared to Logistic Regression and other models . However, the absence of ROC AUC scores makes it challenging to assess the models' discrimination ability fully. Considering multiple performance metrics is crucial for a comprehensive evaluation of classification models.

## Data Preparation for pipelining
To build the pre-payment predictor, the following steps were performed on the dataset:
- Calculate EMI (Equated Monthly Installment): EMI for short, is the amount payable every month to the bank or any other financial institution until the loan amount is fully paid off. It consists of the interest on the loan as well as part of the principal amount to be repaid.
  
  ![Picture2](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/f24ce197-125d-492d-baec-385b631235e8)

- Calculation of Total Payment and Interest Amount: The total payment was calculated by multiplying the EMI with the loan tenure. The interest amount was derived by subtracting the principal amount from the total payment.
- - Calculation of Monthly Income: The monthly income of borrowers was estimated by dividing the DTI (debt-to-income ratio) with the monthly debt (EMI). This provides an approximation of the borrowers' monthly income based on their loan obligations.
- Calculation of Current Principal: The remaining principal amount was calculated based on the number of months in repayment, the monthly interest rate, the original principal amount, and the EMI. This reflects the outstanding loan amount after deducting the principal paid during the repayment period.
  
  ![Screenshot 2023-07-15 221050](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/d89a19c7-867e-4c95-bd26-ae6eeb45536e)
  
## Pre-payment Calculation
To simulate pre-payment, the following assumptions were made:
Savings Calculation: Borrowers with a DTI below 40% are assumed to save 50% of their income, inclusive of the EMI, for a period of two years. Borrowers with a DTI above 40% are assumed to save 75% of their income.
Pre-payment Amount: The pre-payment amount is calculated by subtracting the total savings from the outstanding loan amount after two years. This represents the amount borrowers are expected to pre-pay at the end of the designated period.

![Screenshot 2023-07-15 221203](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/829b7016-3aa6-4fec-9481-15874a671527)

## Data splitting and  model implementation

After pre-processing the data and separating the features and targets, we divided our data into training and testing sets using the train_test_split function from the scikit-learn library. The input features are denoted as "X" and the corresponding target variables or labels are denoted as "y". We allocated 20% of the data for testing and used the remaining 80% for training. 

![Picture3](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/c30e1e57-c9d7-4dbc-9f80-a220833928c2)

To evaluate the performance of the Regression model, we calculated several evaluation metrics. The Mean Squared Error (MSE) measures the average squared difference between the actual target values (y_test) and the predicted values (y_pred). The Mean Absolute Error (MAE) measures the average absolute difference between the actual target values and the predicted values. The R-squared (R2) score indicates the proportion of the variance in the target variable that is explained by the linear regression model. 
Finally, we printed the coefficients, intercept, and evaluation metrics of the Linear Regression model. The coefficients represent the magnitude and direction of the impact of each input feature on the target variable, while the intercept represents the model's predicted target value when all input features are zero. This information provides insights into the model's performance and the relationship between the features and the target variable.

![Screenshot 2023-07-17 170558](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/4d54a071-91d1-4b75-91a9-150e46c40971)


## Linear Regression
To implement Linear Regression, we followed these steps: 
First, we created a Linear Regression model by importing the LinearRegression class from the scikit-learn library. This model represents a linear regression model. Next, we fitted the model using the training data (X_train and y_train) by calling the fit() method. This step allowed the model to learn the coefficients and intercept, which define the linear relationship between the input features and the target variable. 
After training the model, we made predictions on the test data (X_test) by calling the predict() method. The predicted values were assigned to the variable "y_pred". 

![Picture4](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/5271818a-82d8-4e50-8a99-57d4607aa3bc)

## Ridge Regression
Firstly, we created a Ridge Regression model by importing the Ridge class from the scikit-learn library. We also specified a specific alpha value, which controls the strength of L2 regularization. This regularization helps balance between fitting the training data and preventing overfitting.
 Next, we fitted the Ridge regression model using the training data (X_train and y_train) by calling the fit() method. This step optimized the coefficients of the model by minimizing the sum of squared residuals between the predicted values and the actual target values, taking into account the additional penalty term introduced by the regularization. Once the model was trained, we made predictions on the test data (X_test) by calling the predict() method. The predicted values were stored in the variable "y_pred". 

## Random Forest Classifier
We also implemented a Random Forest Classifier using the following approach:
- Firstly, we imported the necessary modules from scikit-learn and imbalanced-learn to enable Random Forest classification.
- Next, we split the data into training and testing sets using the train_test_split function. This step allowed us to have separate datasets to train and evaluate our Random Forest Classifier.
- 
  ![Picture5](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/46845b4c-4be4-41e9-a006-243aac712fca)

- After that, we created a pipeline using ImbPipeline from the imbalanced-learn library. The pipeline consisted of a pre-processing step, which could include transformations such as scaling or encoding, and a Random Forest Classifier as the model. The pipeline helps to streamline the workflow and ensures consistent application of pre-processing steps to the data.
- After creating the pipeline, we fitted it with the training data. This step involved training the Random Forest Classifier model on the training set using the fit() method. The pipeline automatically applied the pre-processing steps before fitting the model.

## Results
The accuracy we obtained for each model was as follows:
- Linear Regression: 74%
- Ridge Regression: 74% (negligible difference from Linear Regression)
- Random Forest Classifier: 87%

## Creating the pipeline models

We did preprocessing part for the pipelinemodel so that it only allows the required columns to be used in the pipeline, that is only they are allowed to passthrough the pipeline using the column transform method.

![Screenshot 2023-07-15 213851](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/9a644487-35dd-4eaf-842f-b828eb80accc)

We will then do the preprocessing of the data for pipeline of random forest and xg boost model pipeline for checking everdelinquency prediction. we will then save the pipeline for the future, in deployment usage.

![Screenshot 2023-07-15 213814](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/33baac41-bce6-4689-8bab-85cf603214cd)

We made pipeline models for xg boost model, random forest model amd the logistic regression models.

![Screenshot 2023-07-19 201712](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/26e0ba8e-627e-43e9-8a8f-056bbd9235ab)

![Screenshot 2023-07-19 201652](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/2e76717c-825e-4427-8eab-ce8a0b297253)

![Screenshot 2023-07-19 201652](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/0ecf8f16-55e0-4220-a57f-390fa91c69cf)

## Creating Pickles for the pipeline models
We created Pipeline models for the Random Forest model of the EverDelinquent part and the linear regression pipeline for the prepayment risk predictor. We used the pickle library to pickle them and store them for future use in deployment.

![Screenshot 2023-07-15 210829](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/c27a230a-5dbe-4d1e-b6f8-fcf8eb302451)

We can load the file whenever needed

![Screenshot 2023-07-15 210927](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/20432845-1b84-426d-b815-c2875dee4d02)


## Deployment of the model
This code implements a FastAPI web application for predicting loan eligibility based on various input parameters. The application uses a machine learning model trained on loan data. Here is an overview of the code:
1. Importing Libraries: The necessary libraries and modules are imported, including uvicorn for running the FastAPI application, pandas for data manipulation, numpy for array operations, and FastAPI and Jinja2Templates for building the web interface.
2. Loading the Model: The pre-trained machine learning model is loaded using the pickle module.
3. Defining Endpoints: Two endpoints are defined. The first endpoint ("/") renders an HTML form using a Jinja2 template to collect user input. The second endpoint ("/predict") handles the form submission and predicts loan eligibility based on the provided input.
4. Handling Form Submission: The "/predict" endpoint retrieves the form data submitted by the user and converts it into a pandas DataFrame. Some data preprocessing steps are performed, such as label encoding categorical variables and creating derived features.
5. Predicting Loan Eligibility: The loaded machine learning model is used to predict loan eligibility based on the preprocessed input data. If the loan is predicted as delinquent, a message indicating ineligibility is displayed. Otherwise, if the input data is valid and the model predicts eligibility, an appropriate message is displayed along with the predicted loan outcome.
6. Running the Application: The FastAPI application is run using the uvicorn server on port 8000.
 
![Picture6](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/a14b9a66-8346-4bf3-b057-cc46c72f17a6)

Additionally, the code includes the use of ngrok to create a tunnel and expose the application to the internet for testing purposes.

It's worth noting that some parts of the code, such as the file paths for loading the model and templates, should be adjusted based on the actual directory structure of the project.

Overall, this code provides a functional implementation of a FastAPI web application for loan eligibility prediction, allowing users to input loan details and obtain predictions in real-time.

![Screenshot 2023-07-19 171641](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/1cacdbfb-e189-46fb-8e8e-9bb5c9e73609)

![Screenshot 2023-07-15 222819](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/c3be46e4-6c09-403c-8485-f8e0a3a05def)

![Screenshot 2023-07-15 222556](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/7e9930d1-54ab-4c4f-a875-d01c6631b43f)

![Screenshot 2023-07-15 223047](https://github.com/Technocolabs100/Machine-Learning-Pipeline-for-Mortgage-Backed-Securities-Prepayment-Risk/assets/112609213/9b27bdb3-f03f-4a4c-93d5-d5e9e801fb51)


## Conclusion
The loan pre-payment predictor offers valuable insights into borrowers' pre-payment behavior, allowing financial institutions and lenders to optimize loan repayment strategies. By considering factors such as EMI, loan tenure, DTI, and delinquency history, lenders can identify borrowers who are likely to pre-pay their loans and tailor their services accordingly.
