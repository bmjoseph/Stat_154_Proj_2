# Stat 154 Project 2 -- Cloud Classification  
## Bailey Joseph and Deborah Chang  

To reproduce our report, follow these steps:  

1) Clone or download this repository so that you have access to the scripts and data.  
2) Make sure you have R and RStudio installed on your computer. We used R version 3.3.3 and RStudio version 1.1.463.  
3) Open the file `TODO` in RStudio.  
    - If you just run this file, you'll be able to see all of our results and reproduce all of our figures.  
    - However, if you want to extend our analyses, continue to step 4.  
4) If you want to try using a different classification algorithm or data splitting method, you can still make use of the CVgeneric function in our helpers.R file. Precise documentation for these methods is available in the file, but essentially CVgeneric uses a functional paradigm to allow for flexibility. To change the algorithm or splitting method, you'll just need to define a new function. 

A new model fitting method must take as parameters:  
    - `training_data`: A dataframe of features to use for training the model.  
    - `training_labels`: The expert label responses corresponding to `training_data`.  
    - `new_data`: The new unlabeled test features to use as inputs for prediction.  
The model fitting method should return a vector of predicted classes, but the internals of the method are completely up to the user. Feel free to try other classification methods such as SVM or Random Forests.  

A new data splitting method must take as parameters:  
    -`training_data`: A dataframe of all training data. There are no restrictions on what you can put here, any changes will only change how you write the inside of your functiion.  
    -`num_folds`: An integer, the number of folds you want to return.  

The data splitting methods should return a list of length `num_folds`, where each element is a vector of `training_data` indices to use as a holdout set for each fold in CV.  

You can use CVgeneric as is, provided you define your new splitting and or fitting function correctly.  