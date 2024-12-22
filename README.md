## AI-POWERED CROP YIELD PREDICTION USING MACHINE LEARNING – BY DECISION TREE ALGORITHM
This study proposes an innovative approach for crop yield prediction leveraging the power of artificial intelligence (AI) and machine learning (ML) techniques. By harnessing advanced algorithms and vast datasets, our model aims to accurately forecast crop yields, aiding farmers, policymakers, and stakeholders in making informed decisions. 

## About
Agriculture plays a pivotal role in the sustenance and economic development of many nations, making the optimization of crop yields a critical endeavor. Traditional farming methods, while effective to a degree, often rely heavily on the experience of farmers and can be subject to unpredictable environmental variables. With the advent of advanced technologies, there is a growing potential to transform agricultural practices through data-driven approaches. Among these, machine learning stands out as a powerful tool for predicting crop yields with greater accuracy and reliability.
Machine learning, a subset of artificial intelligence, involves the use of algorithms and statistical models to analyze and interpret complex data sets. In the context of agriculture, it enables the processing of vast amounts of data related to weather conditions, soil health, crop management practices, and historical yield patterns. By leveraging these data sources, machine learning models can identify patterns and correlations that might be imperceptible to human analysts.
The application of machine learning in crop yield prediction offers several advantages. It allows for the integration of real-time data, facilitating dynamic adjustments to farming practices. Additionally, these predictive models can provide insights into the optimal use of resources such as water, fertilizers, and pesticides, thereby enhancing both productivity and sustainability. Moreover, accurate yield predictions can assist farmers in making informed decisions about crop planning, marketing strategies, and risk management.
Despite its promising potential, the implementation of machine learning in crop yield prediction is not without challenges. The accuracy of predictions depends on the quality  and quantity of data available, which can vary significantly across different regions and crops. Furthermore, the complexity of agricultural ecosystems, influenced by a myriad of biotic and abiotic factors, necessitates sophisticated modeling approaches and continuous refinement of algorithms.
In this context, this project aims to develop a robust machine learning model for predicting crop yields, focusing on integrating diverse data sources and leveraging advanced algorithmic techniques. By addressing the challenges and harnessing the capabilities of machine learning, the project seeks to contribute to the optimization of agricultural practices, ultimately supporting food security and sustainable development.


## Features
<!--List the features of the project as shown below-->
- Implements advance neural network method.
- A framework based application for deployment purpose.
- High scalability.
- Less time complexity.
- A specific scope of Chatbot response model, using json data format.

## Requirements
<!--List the requirements of the project as shown below-->
* Operating System: Requires a 64-bit OS (Windows 10 or Ubuntu) for compatibility with deep learning frameworks.
* Development Environment: Python 3.6 or later is necessary for coding the sign language detection system.
* Deep Learning Frameworks: TensorFlow for model training, MediaPipe for hand gesture recognition.
* Image Processing Libraries: OpenCV is essential for efficient image processing and real-time hand gesture recognition.
* Version Control: Implementation of Git for collaborative development and effective code management.
* IDE: Use of VSCode as the Integrated Development Environment for coding, debugging, and version control integration.
* Additional Dependencies: Includes scikit-learn, TensorFlow (versions 2.4.1), TensorFlow GPU, OpenCV, and Mediapipe for deep learning tasks.

## System Architecture:
![image](https://github.com/user-attachments/assets/72a998ca-85c6-47ba-b9e0-7cbbc5f19ba0)

## Program:
```py
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# 1. Load Data
df_yield = pd.read_csv('input/yield.csv')
df_rain = pd.read_csv('input/rainfall.csv')
df_pes = pd.read_csv('input/pesticides.csv')
df_temp = pd.read_csv('input/temp.csv')

# 2. Clean and Prepare Data

# Strip leading and trailing spaces from all column names
df_yield.columns = df_yield.columns.str.strip()
df_rain.columns = df_rain.columns.str.strip()
df_pes.columns = df_pes.columns.str.strip()
df_temp.columns = df_temp.columns.str.strip()

# Rename and drop unnecessary columns in each dataset
df_yield = df_yield.rename(columns={"Value": "Yield"}).drop(
    ['Year Code', 'Element Code', 'Element', 'Area Code', 
     'Domain Code', 'Domain', 'Unit', 'Item Code'], axis=1
)

# Ensure rainfall data uses numeric values
df_rain['average_rain_fall_mm_per_year'] = pd.to_numeric(
    df_rain['average_rain_fall_mm_per_year'], errors='coerce'
)

# Rename pesticide data columns and drop unnecessary ones
df_pes = df_pes.rename(columns={"Value": "pesticides_tonnes"}).drop(
    ['Element', 'Domain', 'Unit', 'Item'], axis=1)

# Rename columns in temperature data for consistency
df_temp = df_temp.rename(columns={"year": "Year", "country": 'Area'})

# 3. Merge DataFrames by 'Year' and 'Area'
try:
    yield_df = pd.merge(df_yield, df_rain, on=['Year', 'Area'], how='inner')
    yield_df = pd.merge(yield_df, df_pes, on=['Year', 'Area'], how='inner')
    yield_df = pd.merge(yield_df, df_temp, on=['Year', 'Area'], how='inner')
except KeyError as e:
    print(f"Merge Error: {e}")
    raise
# 4. One-Hot Encode Categorical Variables (Area and Item)
yield_df_onehot = pd.get_dummies(yield_df, columns=['Area', 'Item'])

# 5. Separate Features and Labels
features = yield_df_onehot.drop(['Year', 'Yield'], axis=1)
labels = yield_df['Yield']

# 6. Normalize Features
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.3, random_state=42)

# 8. Train the Model
clf = DecisionTreeRegressor()
clf.fit(X_train, y_train)

# 9. Save the Trained Model
joblib.dump(clf, 'backend/model.pkl')
print("Model saved as model.pkl")

```
## Output:

#### Output1 - Final Dataset
![image](https://github.com/user-attachments/assets/3a730a9c-e9bd-4257-b1d0-3523323fc197)

#### Output2 - R^2 Scores
![image](https://github.com/user-attachments/assets/952f492c-300a-4191-b2f5-41d390792fd4)

#### Output3 - Actual vs Predicted
![image](https://github.com/user-attachments/assets/b883a3f6-b0ed-4a9f-8f52-8e1a59769d26)



## Results and Impact
The AI-powered crop yield prediction project successfully leverages machine learning techniques to provide accurate yield estimates based on various agricultural parameters. By analyzing data like rainfall, pesticides, and temperature, the model offers predictions that can help farmers make informed decisions, leading to better planning and resource management. This approach has the potential to optimize agricultural production, reduce waste, and ensure food security.
The project's implementation highlights the value of technology in agriculture, especially in terms of improving yield forecasting accuracy compared to traditional methods. It also demonstrates how data-driven insights can empower farmers and agricultural stakeholders, improving efficiency and productivity. Through continuous model evaluation and fine-tuning, we have achieved reliable results that are beneficial for both small-scale and large-scale farming operations.

## Articles published / References
1. Sharma, R., & Singh, A. (2020). Machine Learning Techniques for Crop Yield Prediction: A Review. Journal of Cleaner Production, 260, 121110.
2. Kumar, A., & Gupta, R. (2021). A Comparative Study of Machine Learning Algorithms for Crop Yield Prediction. Computers and Electronics in Agriculture, 182, 106024.
3. Ma, L., & Chen, Y. (2019). Predicting Crop Yield Using Machine Learning Techniques: A Review. Agricultural Systems, 177, 102697.



