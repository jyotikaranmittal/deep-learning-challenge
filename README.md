# deep-learning-challenge

Before I Begin
I Create a new repository for this project called deep-learning-challenge
Clone the new repository to your computer.


Preprocess the Data
Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, where you'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.

From the provided cloud URL, read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:

What variable(s) are the target(s) for your model?
What variable(s) are the feature(s) for your model?
Drop the EIN and NAME columns.

Determine the number of unique values for each column.

For columns that have more than 10 unique values, determine the number of data points for each unique value.

Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, Other, and then check if the replacement was successful.

Use pd.get_dummies() to encode categorical variables.

Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.

Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

Step 2: Compile, Train, and Evaluate the Model
Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.

Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

Create the first hidden layer and choose an appropriate activation function.

If necessary, add a second hidden layer with an appropriate activation function.

Create an output layer with an appropriate activation function.

Check the structure of the model.

Compile and train the model.

Create a callback that saves the model's weights every five epochs.

Evaluate the model using the test data to determine the loss and accuracy.

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

Step 3: Optimize the Model
Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.

Use any or all of the following methods to optimize your model:

Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
Dropping more or fewer columns.
Creating more bins for rare occurrences in columns.
Increasing or decreasing the number of values for each bin.
Add more neurons to a hidden layer.
Add more hidden layers.
Use different activation functions for the hidden layers.
Add or reduce the number of epochs to the training regimen.


Note: If you make at least three attempts at optimizing your model, you will not lose points if your model does not achieve target performance.

Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.

Import your dependencies and read in the charity_data.csv to a Pandas DataFrame from the provided cloud URL.

Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.

Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.

Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.

Step 4: Write a Report on the Neural Network Model
For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.

The report should contain the following:

Overview of the analysis: Explain the purpose of this analysis. for AlphabetSoupCharoty.h5
The purpose of this analysis was to create and optimize a deep learning model using TensorFlow and Keras to predict whether applicants to Alphabet Soup's funding program will be successful. The goal was to achieve a predictive model accuracy higher than 75% by preprocessing data, designing neural networks, and evaluating multiple model architectures.

Results: Using bulleted lists and images to support your answers, address the following questions:

Data Preprocessing

What variable(s) are the target(s) for your model?
Target Variable:

IS_SUCCESSFUL — this binary column indicates whether a charity was successful in receiving funding.


What variable(s) are the features for your model?
Feature Variables:

All other variables (except identifiers and names), including:

Application Type

Classification

Affiliation

Income Amount

Special Conditions

etc. (after encoding)



What variable(s) should be removed from the input data because they are neither targets nor features?

Removed Variables:

EIN — the employer identification number; a unique identifier but not useful for training.

NAME — names of organizations, which do not offer predictive value and could introduce noise.


Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?
Model Configuration (Best Attempt):

Input Layer: Number of neurons = number of features after encoding and scaling.

Hidden Layers:

Dense Layer with 128 neurons, activation:relu

Dense Layer with 64 neurons, activation: relu

Output Layer:

Dense Layer with 1 neuron, activation: sigmoid

Loss Function: binary_crossentropy

Optimizer: adam

Epochs: 100


Were you able to achieve the target model performance?

Final Performance:

Loss: 0.5650

Accuracy: 72.46%
❌ No, the model did not achieve the target of >75% accuracy.

What steps did you take in your attempts to increase model performance?
Removed non-predictive features (EIN, NAME).

Encoded categorical variables using one-hot encoding.

Scaled numerical features using StandardScaler.

Increased the number of neurons in hidden layers.

Experimented with multiple activation functions (relu, tanh).

Increased the number of epochs from 50 to 100.


Summary: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
The optimized neural network achieved 72.46% accuracy, which is close to the target but not quite above the desired 75% threshold. While performance improved from baseline through tuning, the neural network struggled to generalize well to unseen data.

Recommendations:
Try Ensemble Models:

Algorithms like Random Forests or Gradient Boosting (e.g., XGBoost) could be more suitable due to their ability to handle imbalanced or categorical-heavy data and are known to perform well in classification tasks.

Feature Engineering:

Perform correlation analysis to drop redundant features.

Create grouped or binned categories for rare values in categorical features.

Hyperparameter Tuning:

Use tools like Keras Tuner or GridSearchCV (for traditional models) to optimize layer configurations and batch sizes.

Regularization Techniques:

Add dropout layers or L2 regularization to prevent overfitting and improve generalization.


Step 5: Copy Files Into Your Repository
Now that you're finished with your analysis in Google Colab, you need to get your files into your repository for final submission.

Download your Colab notebooks to your computer.

Move them into your Deep Learning Challenge directory in your local repository.

Push the added files to GitHub.# deep-learning-challenge