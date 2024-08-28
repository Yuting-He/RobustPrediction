#' Tune and Train by Tuning Method Ext
#'
#' This function tunes and trains a specified classifier using an external validation dataset. The function selects 
#' the appropriate tuning and training function based on the specified classifier.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response variable (factor), and the remaining columns should be the predictor variables.
#' @param classifier A character string specifying the classifier to use. Must be one of 'boosting', 'rf', 'lasso', 'ridge', or 'svm'.
#' @param ... Additional arguments to pass to the specific classifier function.
#'
#' @return A list containing the results from the specific classifier tuning and training function.
#' @export
#'
#' @examples
#' \dontrun{
#' # Load sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage with Random Forest
#' result_rf <- tuneandtrainExt(sample_data_train, sample_data_extern, classifier = "rf", num.trees = 500)
#' result_rf$best_min.node.size
#' result_rf$best_model
#'
#' # Example usage with SVM
#' result_svm <- tuneandtrainExt(sample_data_train, sample_data_extern, classifier = "svm", kernel = "linear", cost_seq = 2^(-15:15))
#' result_svm$best_cost
#' result_svm$best_model
#' }
tuneandtrainExt <- function(data, dataext, classifier, ...) {
  
  # arguments
  #args <- list(...)
  
  # run function by classifer
  if (classifier == "boosting") {
    res <- tuneandtrainExtBoost(data = data, dataext = dataext, ...)
  } else if (classifier == "rf") {
    res <- tuneandtrainExtRF(data = data, dataext = dataext, ...)
  } else if (classifier == "lasso") {
    res <- tuneandtrainExtLasso(data = data, dataext = dataext, ...)
  } else if (classifier == "ridge") {
    res <- tuneandtrainExtRidge(data = data, dataext = dataext, ...)
  } else if (classifier == "svm") {
    res <- tuneandtrainExtSVM(data = data, dataext = dataext, ...)
  } else {
    stop("Unsupported classifier type. Choose from 'boosting', 'rf', 'lasso', 'ridge', 'svm'.")
  }
  
  return(res)
}