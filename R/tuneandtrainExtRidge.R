#' Tune and Train External Ridge
#'
#' This function tunes and trains a Ridge classifier using an external validation dataset. The function 
#' selects the best model based on AUC (Area Under the Curve) and provides additional metrics.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), 
#'   and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response 
#'   variable (factor), and the remaining columns should be the predictor variables.
#' @param maxit An integer specifying the maximum number of iterations. Default is 120000.
#' @param nlambda An integer specifying the number of lambda values to use in the Ridge model. Default is 100.
#'
#' @return A list containing the best lambda value (`best_lambda`), the final trained model (`best_model`), 
#'   the AUC of the final model on the training data (`final_auc`).
#' @import glmnet
#' @import pROC
#' @importFrom stats predict
#' @export
#'
#' @examples
#' \dontrun{
#' # Load sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage
#' result <- tuneandtrainExtRidge(sample_data_train, sample_data_extern, maxit = 120000, nlambda = 100)
#' result$best_lambda
#' result$best_model
#' result$final_auc
#' }
tuneandtrainExtRidge <- function(data, dataext, maxit = 120000, nlambda = 100) {
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  dataext <- as.data.frame(dataext)
  
  Train <- data
  Extern <- dataext
  
  # Fit Ridge Regression Model using glmnet package
  fit_Ridge <- glmnet::glmnet(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]), 
                              family = "binomial", maxit = maxit, nlambda = nlambda, alpha = 0, standardize = TRUE)
  
  # External Validation
  pred_Ridge <- stats::predict(fit_Ridge, newx = as.matrix(Extern[, -1]), type = "response")
  
  # Determine AUC to choose 'best' model using pROC package
  AUC <- numeric(ncol(pred_Ridge))
  
  for (i in seq_along(AUC)) {
    AUC[i] <- pROC::auc(response = as.factor(Extern[, 1]), predictor = as.numeric(pred_Ridge[, i]))
  }
  
  chosen_model <- which.max(AUC)
  chosen_lambda <- fit_Ridge$lambda[chosen_model]
  
  # Determine AUC of the chosen model on the Training dataset
  pred_Ridge_Train <- stats::predict(fit_Ridge, newx = as.matrix(Train[, -1]), s = chosen_lambda, type = "response")
  AUC_Train <- pROC::auc(response = as.factor(Train[, 1]), predictor = as.numeric(pred_Ridge_Train))
  
  # Train the final model with the chosen lambda
  final_model <- glmnet::glmnet(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]), 
                                family = "binomial", maxit = maxit, lambda = chosen_lambda, alpha = 0, standardize = TRUE)
  
  # Calculate AUC on the external validation set with the final model
  pred_final <- stats::predict(final_model, newx = as.matrix(Extern[, -1]), type = "response")
  extern_auc <- pROC::auc(response = as.factor(Extern[, 1]), predictor = as.numeric(pred_final))
  
  # Return the result
  res <- list(
    best_lambda = chosen_lambda,
    best_model = final_model,
    final_auc = AUC_Train
  )
  
  # Set class
  class(res) <- "ExtRidge"
  return(res)
}
