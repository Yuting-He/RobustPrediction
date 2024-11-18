#' Tune and Train RobustTuneC Ridge
#'
#' This function tunes and trains a Ridge classifier using the \code{glmnet} package with the "RobustTuneC" method.
#' The function evaluates a sequence of lambda (regularization) values using K-fold cross-validation (K specified by the user) 
#' on the training dataset and selects the best model based on Area Under the Curve (AUC). 
#'
#' The function first performs K-fold cross-validation on the training dataset to select the best lambda value based on AUC. 
#' Then, the model is further validated on an external dataset, and the lambda value that provides the best performance on 
#' the external dataset is chosen as the final model. The Ridge regression is fitted using the selected lambda value, and 
#' the final model's performance is evaluated using AUC on the external validation dataset.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), 
#'   and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response 
#'   variable (factor), and the remaining columns should be the predictor variables.
#' @param K Number of folds to use in cross-validation. Default is 5.
#' @param maxit Maximum number of iterations. Default is 120000.
#' @param nlambda The number of lambda values to use for cross-validation. Default is 100.
#'
#' @return A list containing the best lambda value (`best_lambda`), the final trained model (`best_model`), 
#'   the AUC of the final model (`final_auc`), and the chosen c value(`best_c`).
#' @export
#'
#' @examples
#' # Load sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage
#' result <- tuneandtrainRobustTuneCRidge(sample_data_train, sample_data_extern, 
#'   K = 5, maxit = 120000, nlambda = 100)
#' result$best_lambda
#' result$best_model
#' result$final_auc
#' result$best_c
tuneandtrainRobustTuneCRidge <- function(data, dataext, K = 5, maxit = 120000, nlambda = 100) {
  
  # Fit Ridge Model on training data using glmnet package
  fit_Ridge <- glmnet::glmnet(x = as.matrix(data[, 2:ncol(data)]), y = as.factor(data[, 1]), 
                              family = "binomial", maxit = maxit, nlambda = nlambda, alpha = 0, standardize = TRUE)
  # Get lambda sequence to use for CV
  lamseq <- fit_Ridge$lambda
  
  # Split Train in K parts
  n <- nrow(data)
  
  partition <- rep(1:K, length = n)
  partition <- partition[sample(n)]
  
  # Cross Validation
  AUC_CV <- matrix(NA, nrow = length(lamseq), ncol = K)
  
  for (j in 1:K) {
    XTrain <- data[partition != j, ]
    XTest <- data[partition == j, ]
    
    if (length(levels(as.factor(XTest[, 1]))) == 1) {
      AUC_CV[, j] <- NA
    } else {
      # Fit Ridge Model
      fit_Ridge_CV <- glmnet::glmnet(x = as.matrix(XTrain[, 2:ncol(XTrain)]), y = as.factor(XTrain[, 1]), 
                                     family = "binomial", maxit = maxit, lambda = lamseq, alpha = 0, standardize = TRUE)
      
      # External Validation
      pred_Ridge_CV <- stats::predict(fit_Ridge_CV, newx = as.matrix(XTest[, 2:ncol(XTest)]), s = lamseq, type = "response")
      
      # Determine AUC to choose 'best' model using pROC package
      for (i in 1:ncol(pred_Ridge_CV)) {
        AUC_CV[i, j] <- 1 - pROC::auc(response = XTest[, 1], predictor = pred_Ridge_CV[, i])
      }
    }
  }
  
  # Mean of error (1-AUC) for each lambda
  AUC_mean <- rowMeans(AUC_CV, na.rm = TRUE)
  # Which error (1-AUC) is minimal
  cvmin <- min(AUC_mean, na.rm = TRUE)
  
  cseq = c(1, 1.1, 1.3, 1.5, 2)
  AUC_Test.c <- numeric(length(cseq))
  
  done <- FALSE
  i <- 1
  
  while ((i <= length(cseq)) & !done) {
    if (cseq[i] * cvmin < 0.4) {
      lambda.c <- max(lamseq[which(AUC_mean <= cvmin * cseq[i])], na.rm = TRUE)
    } else {
      if (cvmin < 0.4) {
        lambda.c <- max(lamseq[which(AUC_mean <= 0.4)], na.rm = TRUE)
      } else {
        lambda.c <- max(lamseq[which(AUC_mean <= cvmin)], na.rm = TRUE)
      }
      done <- TRUE
    }
    
    pred_Ridge <- stats::predict(fit_Ridge, newx = as.matrix(dataext[, 2:ncol(dataext)]), s = lambda.c, type = "response")
    AUC_Test.c[i] <- pROC::auc(response = as.factor(dataext[, 1]), predictor = pred_Ridge[, 1])
    
    i <- i + 1
  }
  
  nctried <- i - 1
  c <- cseq[max(which(AUC_Test.c[1:(i-1)] == max(AUC_Test.c[1:(i-1)])))]
  
  if (c * cvmin < 0.4) {
    lambda.c <- max(lamseq[which(AUC_mean <= cvmin * c)], na.rm = TRUE)
  } else if (cvmin < 0.4) {
    lambda.c <- max(lamseq[which(AUC_mean <= 0.4)], na.rm = TRUE)
  } else {
    lambda.c <- max(lamseq[which(AUC_mean <= cvmin)], na.rm = TRUE)
  }
  
  # Train the final model
  final_model <- glmnet::glmnet(x = as.matrix(data[, 2:ncol(data)]), y = as.factor(data[, 1]), 
                                family = "binomial", lambda = lambda.c, alpha = 0, standardize = TRUE)
  
  # Calculate AUC on the external validation set using pROC package
  final_predictions <- stats::predict(final_model, newx = as.matrix(dataext[, 2:ncol(dataext)]), type = "response")
  final_auc <- pROC::auc(response = as.factor(dataext[, 1]), predictor = final_predictions[, 1])
  
  # Return result:
  res <- list(
    best_lambda = lambda.c,
    best_model = final_model,
    final_auc = final_auc,
    best_c = c
  )
  
  # Set class
  class(res) <- "RobustTuneCRidge"
  return(res)
  
}