#' Tune and Train Classifier by Tuning Method Ext
#'
#' This function tunes and trains a classifier using an external validation dataset. Based on the specified classifier, 
#'   the function selects and runs the appropriate tuning and training process. The external validation data is used 
#'   to optimize the model's hyperparameters and improve generalization performance across datasets.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), 
#'   and the remaining columns should be the predictor variables. Ensure that the data is properly formatted, 
#'   with no missing values.
#' 
#' @param dataext A data frame containing the external validation data. The first column should be the response variable (factor), 
#'   and the remaining columns should be the predictor variables. The external data is used for tuning hyperparameters to 
#'   avoid overfitting on the training data.
#'
#' @param classifier A character string specifying the classifier to use. Must be one of the following:
#'   \itemize{
#'     \item "boosting" for gradient boosting models.
#'     \item "rf" for Random Forest.
#'     \item "lasso" for Lasso regression (for feature selection and regularization).
#'     \item "ridge" for Ridge regression (for regularization).
#'     \item "svm" for Support Vector Machines (SVM).
#'   }
#'
#' @param ... Additional arguments to pass to the specific classifier function. These may include hyperparameters 
#' such as the number of trees for Random Forest, regularization parameters for Lasso/Ridge, or kernel settings for SVM.
#'
#' @return A list containing the results from the classifier's tuning and training process. The returned object typically includes:
#'   \itemize{
#'     \item \code{best_model}: The final trained model using the best hyperparameters.
#'     \item \code{best_hyperparams}: The optimal hyperparameters found during the tuning process.
#'     \item \code{final_auc}: Performance metrics (AUC) of the final model.
#'   }
#' @export
#'
#' @examples
#' \dontrun{
#' # Load sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage with Random Forest
#' result_rf <- tuneandtrainExt(sample_data_train, sample_data_extern, 
#'   classifier = "rf", num.trees = 500)
#' result_rf$best_min.node.size
#' result_rf$best_model
#' result_rf$final_auc
#'
#' # Example usage with SVM
#' result_svm <- tuneandtrainExt(sample_data_train, sample_data_extern, 
#'   classifier = "svm", kernel = "linear", 
#'   cost_seq = 2^(-15:15))
#' result_svm$best_cost
#' result_svm$best_model
#' result_svm$final_auc
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