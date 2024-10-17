#' Tune and Train by tuning method Int
#'
#' This function tunes and trains a specified classifier using internal cross-validation. The classifier is specified 
#' by the `classifier` argument, and the function delegates to the appropriate tuning and training function based 
#' on this choice.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), 
#'   and the remaining columns should be the predictor variables.
#' @param classifier A character string specifying the classifier to use. 
#'   Must be one of 'boosting', 'rf', 'lasso', 'ridge', 'svm'.
#' @param ... Additional arguments to pass to the specific classifier function.
#'
#' @return A list containing the results from the specific classifier's tuning and training process. 
#'   The list typically includes:
#'   \itemize{
#'     \item \code{best_hyperparams}: The best hyperparameters selected by cross-validation.
#'     \item \code{best_model}: The final trained model using the selected hyperparameters.
#'     \item \code{final_auc}: Cross-validation results (AUC).
#'   }
#' @export
#'
#' @examples
#' \dontrun{
#' # Load sample data
#' data(sample_data_train)
#'
#' # Example usage: Tuning and training a Random Forest classifier
#' result_rf <- tuneandtrainInt(sample_data_train, classifier = "rf", 
#'   num.trees = 500, nfolds = 5, seed = 123)
#' result_rf$best_min.node.size
#' result_rf$best_model
#' result_rf$final_auc
#'
#' # Example usage: Tuning and training a SVM classifier
#' result_svm <- tuneandtrainInt(sample_data_train, classifier = "svm", 
#'   kernel = "linear", cost_seq = 2^(-5:5), 
#'   scale = FALSE, nfolds = 5, seed = 123)
#' result_svm$best_cost
#' result_svm$best_model
#' result_svm$final_auc
#' }
tuneandtrainInt <- function(data, classifier, ...) {
  
  # arguments
  #args <- list(...)
  
  # run function by classifer
  if (classifier == "boosting") {
    res <- tuneandtrainIntBoost(data = data,...)
  } else if (classifier == "rf") {
    res <- tuneandtrainIntRF(data = data, ...)
  } else if (classifier == "lasso") {
    res <- tuneandtrainIntLasso(data = data, ...)
  } else if (classifier == "ridge") {
    res <- tuneandtrainIntRidge(data = data, ...)
  } else if (classifier == "svm") {
    res <- tuneandtrainIntSVM(data = data, ...)
  } else {
    stop("Unsupported classifier type. Choose from 'boosting', 'rf', 'lasso', 'ridge', 'svm'.")
  }
  
  return(res)
}