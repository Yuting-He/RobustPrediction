% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tuneandtrainInt.R
\name{tuneandtrainInt}
\alias{tuneandtrainInt}
\title{Tune and Train by tuning method Int}
\usage{
tuneandtrainInt(data, classifier, ...)
}
\arguments{
\item{data}{A data frame containing the training data. The first column should be the response variable (factor), 
and the remaining columns should be the predictor variables.}

\item{classifier}{A character string specifying the classifier to use. 
Must be one of 'boosting', 'rf', 'lasso', 'ridge', 'svm'.}

\item{...}{Additional arguments to pass to the specific classifier function.}
}
\value{
A list containing the results from the specific classifier's tuning and training process. 
  The list typically includes:
  \itemize{
    \item \code{best_hyperparams}: The best hyperparameters selected by cross-validation.
    \item \code{best_model}: The final trained model using the selected hyperparameters.
    \item \code{final_auc}: Cross-validation results (AUC).
  }
}
\description{
This function tunes and trains a specified classifier using internal cross-validation. The classifier is specified 
by the `classifier` argument, and the function delegates to the appropriate tuning and training function based 
on this choice.
}
\examples{
# Load sample data
data(sample_data_train)

# Example usage with Lasso
result_lasso <- tuneandtrainInt(sample_data_train, classifier = "lasso",
  maxit = 120000, nlambda = 100)
result_lasso$best_lambda
result_lasso$best_model
result_lasso$final_auc
result_lasso$active_set_Train

# Example usage with Ridge
result_ridge <- tuneandtrainInt(sample_data_train, classifier = "ridge", 
  maxit = 120000, nlambda = 100)
result_ridge$best_lambda
result_ridge$best_model
result_ridge$final_auc
}
