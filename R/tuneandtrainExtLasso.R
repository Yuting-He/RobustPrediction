#' Tune and Train External Lasso
#'
#' This function tunes and trains a Lasso classifier using the \code{glmnet} package. 
#' It provides two strategies for tuning hyperparameters based on the \code{estperf} argument:
#' \itemize{
#'   \item When \code{estperf = FALSE} (default): Hyperparameters are tuned using the external validation dataset. 
#'         The lambda value that gives the highest AUC on the external dataset is selected as the best model.
#'         However, no AUC value is returned in this case, as per best practices.
#'   \item When \code{estperf = TRUE}: Hyperparameters are tuned internally using the training dataset. 
#'         The model is then validated on the external dataset to provide a conservative (slightly pessimistic) AUC estimate.
#' }
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), 
#'   and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response variable 
#'   (factor), and the remaining columns should be the predictor variables.
#' @param estperf A logical value indicating whether to use internal tuning with external validation (\code{TRUE}) 
#'   or external tuning (\code{FALSE}). Default is \code{FALSE}.
#' @param maxit An integer specifying the maximum number of iterations. Default is 120000.
#' @param nlambda An integer specifying the number of lambda values to use in the Lasso model. Default is 100.
#'
#' @return A list containing the following components:
#'   \itemize{
#'     \item \code{best_lambda}: The optimal lambda value determined during the tuning process.
#'     \item \code{best_model}: The trained Lasso model using the selected lambda value.
#'     \item \code{est_auc}: The AUC value evaluated on the external dataset. This is only returned when \code{estperf = TRUE}, 
#'       providing a conservative (slightly pessimistic) estimate of the model's performance.
#'     \item \code{active_set_Train}: The number of active coefficients (non-zero) in the model trained on the training dataset.
#'   }
#' 
#' @export
#'
#' @examples
#' # Load sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage with external tuning (default)
#' result <- tuneandtrainExtLasso(sample_data_train, sample_data_extern, maxit = 120000, nlambda = 100)
#' print(result$best_lambda)
#' print(result$best_model)
#' print(result$active_set_Train)
#'
#' # Example usage with internal tuning and external validation
#' result_internal <- tuneandtrainExtLasso(sample_data_train, sample_data_extern, 
#'   estperf = TRUE, maxit = 120000, nlambda = 100)
#' print(result_internal$best_lambda)
#' print(result_internal$best_model)
#' print(result_internal$est_auc)
#' print(result_internal$active_set_Train)


tuneandtrainExtLasso <- function(data, dataext, estperf = FALSE, maxit = 120000, nlambda = 100) {
  data <- as.data.frame(data)
  dataext <- as.data.frame(dataext)
  
  Train <- data
  Extern <- dataext
  
  # Train Lasso model using glmnet
  fit_Lasso <- glmnet::glmnet(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]), 
                              family = "binomial", maxit = maxit, nlambda = nlambda, standardize = TRUE)
  
  if (!estperf) {
    # External tuning strategy: select best parameters using the external dataset
    pred_Lasso <- stats::predict(fit_Lasso, newx = as.matrix(Extern[, -1]), s = fit_Lasso$lambda, type = "response")
    AUC <- numeric(ncol(pred_Lasso))
    for (i in seq_along(AUC)) {
      AUC[i] <- pROC::auc(response = as.factor(Extern[, 1]), predictor = pred_Lasso[, i], quiet = TRUE)
    }
    chosen_model <- which.max(AUC)
    chosen_lambda <- fit_Lasso$lambda[chosen_model]
  } else {
    # Internal tuning strategy: select best parameters using the training dataset
    pred_Lasso <- stats::predict(fit_Lasso, newx = as.matrix(Train[, -1]), s = fit_Lasso$lambda, type = "response")
    AUC <- numeric(ncol(pred_Lasso))
    for (i in seq_along(AUC)) {
      AUC[i] <- pROC::auc(response = as.factor(Train[, 1]), predictor = pred_Lasso[, i], quiet = TRUE)
    }
    chosen_model <- which.max(AUC)
    chosen_lambda <- fit_Lasso$lambda[chosen_model]
    
    # Validate the internally tuned model on the external dataset
    final_model <- glmnet::glmnet(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]), 
                                  family = "binomial", maxit = maxit, lambda = chosen_lambda, standardize = TRUE)
    pred_Extern <- stats::predict(final_model, newx = as.matrix(Extern[, -1]), type = "response")
    result_auc <- pROC::auc(response = as.factor(Extern[, 1]), predictor = as.numeric(pred_Extern), quiet = TRUE)
  }
  
  # Get the number of active coefficients
  coef_active <- stats::coef(fit_Lasso, s = chosen_lambda)
  active_set <- length(coef_active@x)
  
  # Return results
  res <- list(
    best_lambda = chosen_lambda,
    best_model = fit_Lasso,
    est_auc = if (estperf) result_auc else NULL,
    active_set_Train = active_set
  )
  
  class(res) <- "ExtLasso"
  return(res)
}
