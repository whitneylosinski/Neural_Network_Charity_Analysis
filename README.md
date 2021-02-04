# Neural Network Charity Analysis

The purpose of this analysis was to use machine learning and neural networks to create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.  The three steps of the analysis were 

1. Preprocessing the data
2. Compiling, Training and Evaluating the Model
3. Optimizing the Model

## Resources
Data: charity_data.csv </br>
Software: Python 3.7, Pandas 1.1.3, Scikit-learn 0.23.2, TensorFlow 2.4.1, Jupyter 1.0.0 

## Results

### Preprocessing the Data

The first step of the analysis was to preprocess the data.  First, the 'EIN' and 'NAME' columns were dropped from the data because it is safe to assume that they would not have an effect on the success of applicants.  

![Drop_columns](Results/Drop_columns.png)

Next, all object type columns were reviewed to determine if the number of unique values should be reduced by binning less common values into an "other" group. This was completed for both the Application Type and Classification columns.  

![Binning1](Results/Binning1.png)
![Binning2](Results/Binning2.png)

Then, because the model is unable to handle object type data, each of the object type columns was encoded using OneHotEncoder and the original object type columns were dropped.

![Encoding](Results/Encoding.png)
![Merge](Results/Merge.png)

After converting all of the data to the proper format, the target variable (y) for the analysis was set as the 'IS_SUCCESSFUL' column and the features variable (X) was set as all of the columns associated with 'APPLICATION TYPE', 'AFFILIATION', 'CLASSIFICATION', 'USE_CASE', 'ORGANIZATION', 'STATUS', 'INCOME_AMT', 'SPECIAL_CONSIDERATIONS' and 'ASK_AMT'.  The data was then split into training and testing datasets and the features datasets (X_train & X_test) were scaled using StandardScaler() to normalize the data in hopes to create a more accurate model.

![Scale](Results/Scale.png)

### Compiling, Training, and Evaluating the Model

For the first pass at running the model, the number of activation functions was set equal to the number of columns in the features variable (X), which was 43.  Two layers were used initially to see if a basic nueral network would be a good fit for the data.  The first layer was set to use 80 neurons, which is about double the input features, and the second layer was set to use 30 neurons.  The two layers were assigned the "relu" activation and the output later was assigned the "sigmoid" activation.  This original model acheived an accuracy of 72.40%.


How many neurons, layers, and activation functions did you select for your neural network model, and why?
Were you able to achieve the target model performance?
What steps did you take to try and increase model performance?


### Summary
Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.
