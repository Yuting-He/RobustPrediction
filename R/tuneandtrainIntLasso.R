#' Tune and Train Internal Lasso
#'
#' This function tunes and trains a Lasso classifier using the \code{glmnet} package. The function 
#' performs internal cross-validation to evaluate a sequence of lambda (regularization) values and 
#' selects the best model based on the Area Under the Curve (AUC).
#'
#' This function trains a logistic Lasso model on the training dataset using cross-validation. 
#' The lambda value that results in the highest AUC during cross-validation is chosen as the best model, 
#' and the final model is trained on the full training dataset with this optimal lambda value.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), 
#'   and the remaining columns should be the predictor variables.
#' @param maxit An integer specifying the maximum number of iterations. Default is 120000.
#' @param nlambda An integer specifying the number of lambda values to use in the Lasso model. Default is 200.
#' @param nfolds An integer specifying the number of folds for cross-validation. Default is 5.
#'
#' @return A list containing the best lambda value (`best_lambda`), the final trained model (`best_model`), 
#'   and the number of active coefficients (`active_set_Train`).
#' @export
#'
#' @examples
#' # Load sample data
#' data(sample_data_train)
#'
#' # Example usage
#' result <- tuneandtrainIntLasso(sample_data_train, maxit = 120000, nlambda = 200, nfolds = 5)
#' result$best_lambda
#' result$best_model
#' result$active_set_Train

tuneandtrainIntLasso <- function(data, maxit = 120000, nlambda = 200, nfolds = 5) {
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  
  # Split data into predictors and response
  X <- as.matrix(data[, -1])
  y <- as.factor(data[, 1])
  
  # Fit initial Lasso model to obtain lambda sequence using glmnet package
  fit_Lasso <- glmnet::glmnet(x = X, y = y, family = "binomial", maxit = maxit, nlambda = nlambda, standardize = TRUE)
  lamseq <- fit_Lasso$lambda
  
  # Cross-validation
  partition <- sample(rep(1:nfolds, length.out = nrow(data)))
  AUC_CV <- matrix(NA, nrow = nlambda, ncol = nfolds)
  
  for (j in 1:nfolds) {
    XTrain <- X[partition != j, , drop = FALSE]
    yTrain <- y[partition != j]
    XTest <- X[partition == j, , drop = FALSE]
    yTest <- y[partition == j]
    
    if (length(unique(yTest)) == 1) {
      AUC_CV[, j] <- NA
    } else {
      fit_Lasso_CV <- glmnet::glmnet(x = XTrain, y = yTrain, family = "binomial", 
                                     maxit = maxit, lambda = lamseq, standardize = TRUE)
      pred_Lasso_CV <- stats::predict(fit_Lasso_CV, newx = XTest, s = lamseq, type = "response")
      
      for (i in 1:ncol(pred_Lasso_CV)) {
        AUC_CV[i, j] <- pROC::auc(response = yTest, predictor = pred_Lasso_CV[, i], quiet = TRUE)
      }
    }
  }
  
  # Determine the best lambda based on the highest average AUC
  mean_AUC <- rowMeans(AUC_CV, na.rm = TRUE)
  best_lambda_idx <- which.max(mean_AUC)
  best_lambda <- lamseq[best_lambda_idx]
  
  # Final model training with the best lambda using glmnet package
  final_model <- glmnet::glmnet(x = X, y = y, family = "binomial", 
                                maxit = maxit, lambda = best_lambda, standardize = TRUE)
  
  # Count the number of active coefficients
  active_set_Train <- length(stats::coef(final_model, s = best_lambda)@x)
  
  # Return results
  res <- list(
    best_lambda = best_lambda,
    best_model = final_model,
    active_set_Train = active_set_Train
  )
  
  # Set class
  class(res) <- "IntLasso"
  return(res)
}