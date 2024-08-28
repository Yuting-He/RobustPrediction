#' Tune and Train RobustTuneC Lasso
#'
#' This function tunes and trains a Lasso classifier using the "RobustTuneC" method. The function 
#' uses K-fold cross-validation (with K specified by the user) to select the best model based on AUC (Area Under the Curve).
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response variable (factor), and the remaining columns should be the predictor variables.
#' @param K Number of folds to use in cross-validation. Default is 5.
#' @param maxit Maximum number of iterations. Default is 120000.
#' @param nlambda The number of lambda values to use for cross-validation. Default is 100.
#'
#' @return A list containing the best lambda value (`best_lambda`), the final trained model (`best_model`), and the AUC of the final model (`final_auc`).
#' @export
#'
#' @import glmnet
#' @import pROC
#'
#' @examples
#' \dontrun{
#' # Load sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage
#' result <- tuneandtrainRobustTuneCLasso(sample_data_train, sample_data_extern, K = 5, maxit = 120000, nlambda = 100)
#' result$best_lambda
#' result$best_model
#' result$final_auc
#' }
tuneandtrainRobustTuneCLasso <- function(data, dataext, K = 5, maxit = 120000, nlambda = 100) {
  
  # library
  library(glmnet)
  library(pROC)
  
  # Fit Lasso Model on training data
  fit_Lasso <- glmnet(x = as.matrix(data[,2:ncol(data)]), y = as.factor(data[,1]), 
                      family = "binomial", maxit = maxit, nlambda = nlambda, standardize = TRUE)
  # Get lambda sequence to use for CV
  lamseq <- fit_Lasso$lambda
  
  # Split Train in K parts
  n <- nrow(data)
  
  partition <- rep(1:K, length = n)
  partition <- partition[sample(n)]
  
  # Cross Validation
  AUC_CV <- matrix(NA, nrow = length(lamseq), ncol = K)
  
  for (j in 1:K) {
    XTrain <- data[partition != j,]
    XTest <- data[partition == j,]
    
    if (length(levels(as.factor(XTest[,1]))) == 1) {
      AUC_CV[,j] <- NA
    } else {
      # Fit Lasso Model
      fit_Lasso_CV <- glmnet(x = as.matrix(XTrain[,2:ncol(XTrain)]), y = as.factor(XTrain[,1]), 
                             family = "binomial", maxit = maxit, lambda = lamseq, standardize = TRUE)
      
      # external validation
      pred_Lasso_CV <- predict(fit_Lasso_CV, newx = as.matrix(XTest[,2:ncol(XTest)]), s = lamseq, type = "response")
      
      # Determine 1-AUC to choose 'best' model
      for (i in 1:ncol(pred_Lasso_CV)) {
        AUC_CV[i,j] <- 1 - auc(response = XTest[,1], predictor = pred_Lasso_CV[,i])
      }
    }
  }
  
  # Mean of error (1-AUC) for each lambda
  AUC_mean <- rowMeans(AUC_CV, na.rm = TRUE)
  cvmin <- min(AUC_mean, na.rm = TRUE)
  
  # Which error (1-AUC) is minimal
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
    
    pred_Lasso <- predict(fit_Lasso, newx = as.matrix(dataext[,2:ncol(dataext)]), s = lambda.c, type = "response")
    AUC_Test.c[i] <- auc(response = as.factor(dataext[,1]), predictor = pred_Lasso[,1])
    
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
  
  # train the final model
  final_model <- glmnet(x = as.matrix(data[,2:ncol(data)]), y = as.factor(data[,1]), 
                        family = "binomial", lambda = lambda.c, standardize = TRUE)
  
  # Calculate AUC on the external validation set
  final_predictions <- predict(final_model, newx = as.matrix(dataext[,2:ncol(dataext)]), type = "response")
  final_auc <- auc(response = as.factor(dataext[,1]), predictor = final_predictions[,1])
  
  # return the result
  res <- list(
    best_lambda = lambda.c,
    best_model = final_model,
    final_auc = final_auc
  )
  
  # set class
  class(res) <- "RobustTuneCLasso"
  return(res)
  
}