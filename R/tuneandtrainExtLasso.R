#' Tune and Train External Lasso
#'
#' This function tunes and trains a Lasso classifier using the \code{glmnet} package. The function 
#' evaluates a sequence of lambda (regularization) values on an external validation dataset and selects 
#' the best model based on the Area Under the Curve (AUC).
#' 
#' This function trains a logistic Lasso model using the training dataset and validates it on the 
#' external validation dataset. 
#' The lambda value that gives the highest AUC on the external validation dataset is selected as the best model.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), 
#'   and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response variable 
#'   (factor), and the remaining columns should be the predictor variables.
#' @param maxit An integer specifying the maximum number of iterations. Default is 120000.
#' @param nlambda An integer specifying the number of lambda values to use in the Lasso model. Default is 100.
#'
#' @return A list containing the best lambda value (`best_lambda`), the final trained model (`best_model`), 
#'   the AUC value of the final model (`final_auc`), The number of active coefficients (`active_set_Train`).
#' @import glmnet
#' @import pROC
#' @importFrom stats predict
#' @importFrom stats coef
#' @export
#'
#' @examples
#' \dontrun{
#' # Load sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage
#' result <- tuneandtrainExtLasso(sample_data_train, sample_data_extern, maxit = 120000, nlambda = 100)
#' result$best_lambda
#' result$best_model
#' result$final_auc
#' result$active_set_Train
#' }

tuneandtrainExtLasso <- function(data, dataext, maxit = 120000, nlambda = 100) {
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  dataext <- as.data.frame(dataext)
  
  Train <- data
  Extern <- dataext
  
  # Fit Lasso Model using glmnet package
  fit_Lasso <- glmnet::glmnet(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]), 
                              family = "binomial", maxit = maxit, nlambda = nlambda, standardize = TRUE)
  
  # External Validation
  pred_Lasso <- stats::predict(fit_Lasso, newx = as.matrix(Extern[, -1]), s = fit_Lasso$lambda, type = "response")
  
  # Determine AUC to choose 'best' model using pROC package
  AUC <- numeric(ncol(pred_Lasso))
  
  for (i in seq_along(AUC)) {
    AUC[i] <- pROC::auc(response = as.factor(Extern[, 1]), predictor = pred_Lasso[, i])
  }
  
  chosen_model <- which.max(AUC)
  chosen_lambda <- fit_Lasso$lambda[chosen_model]
  coef_active_a <- coef(fit_Lasso, s = chosen_lambda)
  active_set_a <- length(coef_active_a@x)
  
  # Determine AUC of the chosen model on the Training dataset
  pred_Lasso_Train <- stats::predict(fit_Lasso, newx = as.matrix(Train[, -1]), s = chosen_lambda, type = "response")
  AUC_Train <- pROC::auc(response = as.factor(Train[, 1]), predictor = as.numeric(pred_Lasso_Train))
  
  # Train the final model with the chosen lambda
  final_model <- glmnet::glmnet(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]), 
                                family = "binomial", maxit = maxit, lambda = chosen_lambda, standardize = TRUE)
  
  # Return the result
  res <- list(
    best_lambda = chosen_lambda,
    best_model = final_model,
    final_auc = AUC_Train,
    active_set_Train = active_set_a
  )
  
  # Set class
  class(res) <- "ExtLasso"
  return(res)
}