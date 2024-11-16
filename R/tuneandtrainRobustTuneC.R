#' Tune and Train Classifier by Tuning Method RobustTuneC
#'
#' This function tunes and trains a specified classifier using the "RobustTuneC" method and the provided data.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), 
#'   and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the 
#'   response variable (factor), and the remaining columns should be the predictor variables.
#' @param classifier A character string specifying the classifier to use. Must be one of the following:
#'   \itemize{
#'     \item "boosting" for Boosting classifiers.
#'     \item "rf" for Random Forest.
#'     \item "lasso" for Lasso regression.
#'     \item "ridge" for Ridge regression.
#'     \item "svm" for Support Vector Machines.
#'   }
#' @param ... Additional arguments to pass to the specific classifier function.
#'
#' @return A list containing the results from the specific classifier's tuning and training process, 
#'   the returned object typically includes:
#'   \itemize{
#'     \item \code{best_hyperparams}: The best hyperparameters selected through the RobustTuneC method.
#'     \item \code{best_model}: The final trained model based on the best hyperparameters.
#'     \item \code{final_auc}: Performance metrics (AUC) of the final model.
#'   }
#' @export
#'
#' @examples
#' # Load sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage with Lasso
#' result_lasso <- tuneandtrainRobustTuneC(sample_data_train, sample_data_extern, classifier = "lasso",
#'   maxit = 120000, nlambda = 100)
#' result_lasso$best_lambda
#' result_lasso$best_model
#' result_lasso$final_auc
#' result_lasso$active_set_Train
#'
#' # Example usage with Ridge
#' result_ridge <- tuneandtrainRobustTuneC(sample_data_train, sample_data_extern, 
#'   classifier = "ridge", maxit = 120000, nlambda = 100)
#' result_ridge$best_lambda
#' result_ridge$best_model
#' result_ridge$final_auc
tuneandtrainRobustTuneC <- function(data, dataext, classifier, ...) {
  
  # run function by classifier
  if (classifier == "boosting") {
    res <- tuneandtrainRobustTuneCBoost(data = data, dataext = dataext, ...)
  } else if (classifier == "rf") {
    res <- tuneandtrainRobustTuneCRF(data = data, dataext = dataext, ...)
  } else if (classifier == "lasso") {
    res <- tuneandtrainRobustTuneCLasso(data = data, dataext = dataext, ...)
  } else if (classifier == "ridge") {
    res <- tuneandtrainRobustTuneCRidge(data = data, dataext = dataext, ...)
  } else if (classifier == "svm") {
    res <- tuneandtrainRobustTuneCSVM(data = data, dataext = dataext, ...)
  } else {
    stop("Unsupported classifier type. Choose from 'boosting', 'rf', 'lasso', 'ridge', 'svm'.")
  }
  
  return(res)
}