# =========================================================
# Project 2 Helper Functions
# Authors: Bailey Joseph and Deborah Chang
# Due Date: May 1, 2019
# Contents (you can search these to find them)
# 1) Data Splitting Functions
#    - folds_from_xy_16
#    - folds_from_separate_imgs (TODO)
# 2) Classifier Functions
#    - log_reg_classifier
#    - qda_classifier
#    - lda_reg_classifier
#    - knn7_classifier
# 3) Accuracy Functions
#    - compute_standard_acc
# 4) CVgeneric Function
#    - CVgeneric
# =========================================================

library(caret) # For generating folds
library(MASS) # For LDA and QDA functions
library(dplyr) # For data cleaning
library(FNN) # For KNN function
# This overwrite is never desired
select <- dplyr::select


# =========================================================
# Data Splitting Functions for CV
# =========================================================
# Explanation: All splitting functions behave the same way
# INPUTS: 
#   training_data: dataframe of ONLY features we want to predict with
#   num_folds: the new (test) data to generate predictions for
# OUTPUTS:
#   a list of length num_folds, where each element
#         is a vector specifiying hold out indices for
#         training data
#   
# =========================================================

folds_from_xy_16 <- function(training_data, num_folds) {
  # Assumed that the training data has a column called
  #         `image` and a column called `group_lab`
  # Returns a list of length num_folds, where each element
  #         is a vector specifiying hold out indices for
  #         training data if training data is used in num_folds K fold CV.
  
  training_data <- training_data %>%
    mutate(real_group_id = 16*image + as.numeric(group_lab),
           real_ind = row_number())
  
  all_real_ids <- unique(training_data$real_group_id)
  
  fold_group_inds <- createFolds(all_real_ids, num_folds)
  
  final_folds <- list()
  
  for (i in 1:num_folds) {
    curr_exclude_groups <- all_real_ids[fold_group_inds[[i]]]
    curr_exclude_inds <- training_data %>%
      filter(real_group_id %in% curr_exclude_groups) %>%
      pull(real_ind)
    
    final_folds[[i]] <- curr_exclude_inds
  }
  return(final_folds)
}


folds_from_separate_imgs <- function(training_data, num_folds) {
  # training data is assumed to have x_coordinate and y_coordinate
  clusts <- kmeans(training_data %>%
                     select(x_coordinate, y_coordinate),
                   num_folds)$cluster
  
  final_folds <- list()
  
  for (i in 1:num_folds) {
    curr_exclude_inds <- which(clusts == i)
    final_folds[[i]] <- curr_exclude_inds
  }
  return(final_folds)
}

# =========================================================
# Classifier Functions
# =========================================================
# Explanation: All classifier functions behave the same way
# INPUTS: 
#   training_data: dataframe of ONLY features we want to predict with
#   training_labels: vector of 1s and 0s corresponding to training classes
#   new_data: the new (test) data to generate predictions for
# OUTPUTS:
#   preds: a vector of predicted classes for new_data
# =========================================================

# Logistic Regression
log_reg_classifier <- function(training_data, training_labels, new_data) {
  appended_data <- training_data %>%
    mutate(answers = training_labels)
  logreg_model <- glm(answers ~ .,
                      data = appended_data,
                      family = binomial)
  logreg_preds <- predict(logreg_model, new_data, type = "response") > .5
  return(logreg_preds)
}

# QDA
qda_classifier <- function(training_data, training_labels, new_data) {
  appended_data <- training_data %>%
    mutate(answers = training_labels)
  
  qda_model <- qda(answers ~ .,
                   data = appended_data)
  
  qda_preds <- predict(qda_model, new_data)$class
  
  return(qda_preds)
}

qda_classifier_with_prob <- function(training_data, training_labels, new_data, cutoff_prob = .3) {
  appended_data <- training_data %>%
    mutate(answers = training_labels)
  
  qda_model <- qda(answers ~ .,
                   data = appended_data)
  
  qda_preds <- predict(qda_model, new_data)$posterior[,2]
  
  return(qda_preds > cutoff_prob)
}

# LDA
lda_classifier <- function(training_data, training_labels, new_data) {
  appended_data <- training_data %>%
    mutate(answers = training_labels)
  
  lda_model <- lda(answers ~ .,
                   data = appended_data)
  
  lda_preds <- predict(lda_model, new_data)$class
  
  return(lda_preds)
}

# KNN with K = 7
knn7_classifier <- function(training_data, training_labels, new_data) {
  this_fit <- knn(training_data, new_data, training_labels, k = 7)
  return(this_fit)
}

# =========================================================
# Accuracy Functions
# =========================================================

compute_standard_acc <- function(preds, answers) {
  return(mean(preds == answers))
}

# =========================================================
# CVgeneric Function
# =========================================================

CVgeneric <- function(classifier, training_features, training_labels, num_folds,
                      loss_function, k_fold_function, random_seed = NA) {
  # Inputs:
  # classifier: a function that takes in training data, and new data and outputs
  #             class predictions for each of the data points in new data
  # training_features: dataframe with ALL features we need to train the model
  #     Warning: Do not include extra features or the classifier will use them
  #     You MAY have four additional columns called x_coordinate, y_coordinate, image, or group_lab
  #     if those are needed for splitting the data. They will be dropped after generating the folds
  # training_labels: vector of the target labels
  #     WARNING: Make sure these are preprocessed to make sense for your task. 
  #              So if you want to use logistic regression, labels should be 0/1
  # num_folds: K for K fold CV.
  # loss_function: A function that takes in predictions and answers and outputs the loss
  #     default: accuracy (average of prediction = answer)
  # k_fold_function: A function that takes in the data and a value of k and returns
  #     a list of vectors of hold out indices
  # random_seed: An optional integer seed to ensure that splitting the folds is reproducible
  
  # Outputs:
  # A vector of length num_folds, the error for each fold
  
  if(!is.na(random_seed)) {
    set.seed(random_seed)
  }
  
  # Get the folds
  folds <- k_fold_function(training_features, num_folds)
  # Drop the (potentially) harmful columns
  training_features$x_coordinate <- NULL
  training_features$y_coordinate <- NULL
  training_features$image <- NULL
  training_features$group_lab <- NULL
  
  # Calculate accuracy of each fold
  folds_losses <- numeric(num_folds)
  for(i in 1:num_folds) {
    curr_training_feats <- training_features[-folds[[i]], ]
    curr_training_labels <- training_labels[-folds[[i]]]
    curr_testing_feats <- training_features[folds[[i]], ]
    curr_testing_labels <- training_labels[folds[[i]]]
    
    test_pred_labels <- classifier(curr_training_feats,
                                   curr_training_labels,
                                   curr_testing_feats)
    folds_losses[i] <- loss_function(test_pred_labels, curr_testing_labels)
  }
  return(folds_losses)
}

# Assorted Helpers

theme_clean <- function() {
  theme_bw(base_size = 11) +
    theme(panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          panel.grid.minor.y = element_blank(),
          panel.background = element_rect(color = "#E1E1E1"),
          panel.border = element_blank())
}

knn_factory <- function(k_choice) {
  # Return a knn classifier for the input k
  knn_classifier <- function(training_data, training_labels, new_data, this_k = k_choice) {
    this_fit <- knn(training_data, new_data, training_labels, k = this_k)
    return(this_fit)
  }
  return(knn_classifier)
}


# Do not edit below this line
print("Sourced helpers successfully.")

