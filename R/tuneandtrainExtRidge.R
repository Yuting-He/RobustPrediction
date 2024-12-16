#' Tune and Train External Ridge
#'
#' This function tunes and trains a Ridge classifier using the \code{glmnet} package. 
#' It provides two strategies for tuning the regularization parameter \code{lambda} based on the \code{estperf} argument:
#' \itemize{
#'   \item When \code{estperf = FALSE} (default): Hyperparameters are tuned using the external validation dataset. 
#'         The \code{lambda} value that gives the highest AUC on the external dataset is selected as the best model.
#'         However, no AUC value is returned in this case, as per best practices.
#'   \item When \code{estperf = TRUE}: Hyperparameters are tuned internally using the training dataset. 
#'         The model is then validated on the external dataset to provide a conservative (slightly pessimistic) AUC estimate.
#' }
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), 
#'   and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response 
#'   variable (factor), and the remaining columns should be the predictor variables.
#' @param estperf A logical value indicating whether to use internal tuning with external validation (\code{TRUE}) 
#'   or external tuning (\code{FALSE}). Default is \code{FALSE}.
#' @param maxit An integer specifying the maximum number of iterations. Default is 120000.
#' @param nlambda An integer specifying the number of lambda values to use in the Ridge model. Default is 100.
#'
#' @return A list containing the following components:
#'   \itemize{
#'     \item \code{best_lambda}: The optimal \code{lambda} value determined during the tuning process.
#'     \item \code{best_model}: The trained Ridge model using the selected \code{lambda}.
#'     \item \code{est_auc}: The AUC value evaluated on the external dataset. This is only returned when \code{estperf = TRUE}, 
#'       providing a conservative (slightly pessimistic) estimate of the model's performance.
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
#' result <- tuneandtrainExtRidge(sample_data_train, sample_data_extern, maxit = 120000, nlambda = 100)
#' print(result$best_lambda)       # Optimal lambda
#' print(result$best_model)        # Final trained model
#' # Note: est_auc is not returned when estperf = FALSE
#'
#' # Example usage with internal tuning and external validation
#' result_internal <- tuneandtrainExtRidge(sample_data_train, sample_data_extern, 
#'   estperf = TRUE, maxit = 120000, nlambda = 100)
#' print(result_internal$best_lambda)  # Optimal lambda
#' print(result_internal$best_model)   # Final trained model
#' print(result_internal$est_auc)      # AUC on external validation dataset

tuneandtrainExtRidge <- function(data, dataext, estperf = FALSE, maxit = 120000, nlambda = 100) {
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  dataext <- as.data.frame(dataext)
  
  Train <- data
  Extern <- dataext
  
  if (!estperf) {
    # External tuning: select lambda based on external validation AUC
    fit_Ridge <- glmnet::glmnet(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]), 
                                family = "binomial", maxit = maxit, nlambda = nlambda, alpha = 0, standardize = TRUE)
    
    # Predict on external validation data
    pred_Ridge <- stats::predict(fit_Ridge, newx = as.matrix(Extern[, -1]), type = "response")
    
    # Calculate AUC for all lambda values
    AUC <- numeric(ncol(pred_Ridge))
    for (i in seq_along(AUC)) {
      AUC[i] <- pROC::auc(response = as.factor(Extern[, 1]), predictor = as.numeric(pred_Ridge[, i]), quiet = TRUE)
    }
    
    # Select the lambda with the highest AUC
    chosen_model <- which.max(AUC)
    chosen_lambda <- fit_Ridge$lambda[chosen_model]
    
    # Train the final model with the chosen lambda
    final_model <- glmnet::glmnet(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]), 
                                  family = "binomial", maxit = maxit, lambda = chosen_lambda, alpha = 0, standardize = TRUE)
    
    # Return the result without AUC
    res <- list(
      best_lambda = chosen_lambda,
      best_model = final_model,
      est_auc = NULL  # No AUC returned when estperf = FALSE
    )
    
  } else {
    # Internal tuning: select lambda based on training data AUC
    fit_Ridge <- glmnet::glmnet(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]), 
                                family = "binomial", maxit = maxit, nlambda = nlambda, alpha = 0, standardize = TRUE)
    
    # Predict on training data
    pred_Ridge <- stats::predict(fit_Ridge, newx = as.matrix(Train[, -1]), type = "response")
    
    # Calculate AUC for all lambda values
    AUC <- numeric(ncol(pred_Ridge))
    for (i in seq_along(AUC)) {
      AUC[i] <- pROC::auc(response = as.factor(Train[, 1]), predictor = as.numeric(pred_Ridge[, i]), quiet = TRUE)
    }
    
    # Select the lambda with the highest AUC
    chosen_model <- which.max(AUC)
    chosen_lambda <- fit_Ridge$lambda[chosen_model]
    
    # Train the final model with the chosen lambda
    final_model <- glmnet::glmnet(x = as.matrix(Train[, -1]), y = as.factor(Train[, 1]), 
                                  family = "binomial", maxit = maxit, lambda = chosen_lambda, alpha = 0, standardize = TRUE)
    
    # Validate the model on external data
    pred_Extern <- stats::predict(final_model, newx = as.matrix(Extern[, -1]), type = "response")
    est_auc <- pROC::auc(response = as.factor(Extern[, 1]), predictor = as.numeric(pred_Extern), quiet = TRUE)
    
    # Return the result with AUC
    res <- list(
      best_lambda = chosen_lambda,
      best_model = final_model,
      est_auc = est_auc  # Conservative AUC estimate
    )
  }
  
  # Set class
  class(res) <- "ExtRidge"
  return(res)
}
