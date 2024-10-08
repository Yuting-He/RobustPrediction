#' Tune and Train RobustTuneC Support Vector Machine (SVM)
#'
#' This function tunes and trains an SVM classifier using the "RobustTuneC" method. The function 
#' uses K-fold cross-validation (with K specified by the user) to select the best model based on AUC (Area Under the Curve).
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response variable (factor), and the remaining columns should be the predictor variables.
#' @param seed An integer specifying the random seed for reproducibility. Default is 123.
#' @param kernel A character string specifying the kernel type to be used in the SVM. It can be "linear", "polynomial", "radial", or "sigmoid". Default is "linear".
#' @param cost_seq A numeric vector of cost values to be evaluated. Default is `2^(-15:15)`.
#' @param scale A logical value indicating whether to scale the predictor variables. Default is `FALSE`.
#' @param K Number of folds to use in cross-validation. Default is 5.
#'
#' @return A list containing the best cost value (`best_cost`), the final trained model (`best_model`), and the AUC of the final model (`final_auc`).
#' @export
#'
#' @import mlr
#' @import e1071
#' @import pROC
#'
#' @examples
#' \dontrun{
#' # Load sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage
#' result <- tuneandtrainRobustTuneCSVM(sample_data_train, sample_data_extern, K = 5, seed = 123, 
#'                                      kernel = "linear", cost_seq = 2^(-15:15), scale = FALSE)
#' result$best_cost
#' result$best_model
#' result$final_auc
#' }
tuneandtrainRobustTuneCSVM <- function(data, dataext, K = 5, seed = 123, kernel = "linear", cost_seq = 2^(-15:15), scale = FALSE) {
  
  # library
  library(mlr)
  library(e1071)
  library(pROC)
  
  # Split Train in K parts
  n <- nrow(data)
  
  partition <- rep(1:K, length = n)
  partition <- partition[sample(n)]
  
  # Cross Validation
  auc_CV <- matrix(NA, nrow = length(cost_seq), ncol = K)
  
  for (j in 1:K) {
    XTrain <- data[partition != j, ]
    XTest <- data[partition == j, ]
    
    Combined_data <- rbind(XTrain, XTest)
    Combined_data <- as.data.frame(Combined_data)  # Ensure it's a data frame
    Combined_data$y <- as.factor(Combined_data$y)
    
    if (length(levels(as.factor(XTest[, 1]))) == 1) {
      auc_CV[, j] <- NA
    } else {
      i <- 1
      for (c in cost_seq) {
        # Fit SVM
        set.seed(seed)
        
        task <- makeClassifTask(data = Combined_data, target = "y", check.data = FALSE)
        lrn <- makeLearner("classif.svm", predict.type = "prob", kernel = kernel, cost = c, scale = scale)
        
        train.set <- 1:nrow(XTrain)
        test.set <- (nrow(XTrain) + 1):nrow(Combined_data)
        
        model <- train(lrn, task, subset = train.set)
        pred <- predict(model, task = task, subset = test.set)
        
        auc_CV[i, j] <- 1 - performance(pred, measures = list(mlr::auc))
        
        i <- i + 1
      }
    }
  }
  
  # Mean of error (1-AUC) for each cost value
  AUC_mean <- rowMeans(auc_CV, na.rm = TRUE)
  cvmin <- min(AUC_mean, na.rm = TRUE)
  
  # choose c
  cseq = c(1, 1.1, 1.3, 1.5, 2)
  AUC_Test.c <- numeric(length(cseq))
  
  done <- FALSE
  i <- 1
  
  while ((i <= length(cseq)) & !done) {
    if (cseq[i] * cvmin < 0.4) {
      cost.c <- min(cost_seq[which(AUC_mean <= cvmin * cseq[i])], na.rm = TRUE)
    } else {
      if (cvmin < 0.4) {
        cost.c <- min(cost_seq[which(AUC_mean <= 0.4)], na.rm = TRUE)
      } else {
        cost.c <- min(cost_seq[which(AUC_mean <= cvmin)], na.rm = TRUE)
      }
      done <- TRUE
    }
    
    CombinedTrainExtern <- rbind(data, dataext)
    CombinedTrainExtern <- as.data.frame(CombinedTrainExtern)  # Ensure it's a data frame
    CombinedTrainExtern$y <- as.factor(CombinedTrainExtern$y)
    train.set <- 1:nrow(data)
    extern.set <- (nrow(data) + 1):nrow(CombinedTrainExtern)
    
    task_Test <- makeClassifTask(data = CombinedTrainExtern, target = "y", check.data = FALSE)
    lrn_Test.c <- makeLearner("classif.svm", predict.type = "prob", kernel = kernel, cost = cost.c, scale = scale)
    
    model_Test.c <- train(lrn_Test.c, task_Test, subset = train.set)
    pred_Test.c <- predict(model_Test.c, task = task_Test, subset = extern.set)
    
    AUC_Test.c[i] <- performance(pred_Test.c, measures = list(mlr::auc))
    
    i <- i + 1
  }
  
  nctried <- i - 1
  c <- cseq[max(which(AUC_Test.c[1:(i-1)] == max(AUC_Test.c[1:(i-1)])))]
  
  if (c * cvmin < 0.4) {
    cost.c <- min(cost_seq[which(AUC_mean <= cvmin * c)], na.rm = TRUE)
  } else if (cvmin < 0.4) {
    cost.c <- min(cost_seq[which(AUC_mean <= 0.4)], na.rm = TRUE)
  } else {
    cost.c <- min(cost_seq[which(AUC_mean <= cvmin)], na.rm = TRUE)
  }
  
  # train the final model
  data <- as.data.frame(data)  # Ensure data is a data frame
  data$y <- as.factor(data$y)  # Ensure the target variable is a factor
  
  final_task <- makeClassifTask(data = data, target = "y", check.data = FALSE)
  final_learner <- makeLearner("classif.svm", predict.type = "prob", kernel = kernel, cost = cost.c, scale = scale)
  final_model <- train(final_learner, final_task)
  
  # Calculate AUC on the external validation set
  dataext <- as.data.frame(dataext)
  pred_final <- predict(final_model, newdata = dataext)
  final_auc <- performance(pred_final, measures = list(mlr::auc))
  
  # return result
  res <- list(
    best_cost = cost.c,
    best_model = final_model,
    final_auc = final_auc
  )
  
  # Set class
  class(res) <- "RobustTuneCSVM"
  return(res)
}