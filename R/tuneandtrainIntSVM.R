#' Tune and Train Internal SVM
#'
#' This function tunes and trains an SVM classifier using internal cross-validation. The function evaluates 
#' different cost values and selects the best model based on AUC (Area Under the Curve).
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), and the remaining columns should be the predictor variables.
#' @param kernel A character string specifying the kernel type to be used in the SVM. Default is "linear".
#' @param cost_seq A numeric vector of cost values to be evaluated. Default is `2^(-15:15)`.
#' @param scale A logical indicating whether to scale the predictor variables. Default is FALSE.
#' @param nfolds An integer specifying the number of folds for cross-validation. Default is 5.
#' @param seed An integer specifying the random seed for reproducibility. Default is 123.
#'
#' @return A list containing the best cost value (`best_cost`), the final trained model (`best_model`), and the AUC on the training data (`final_auc`).
#' @import e1071
#' @import mlr
#' @import pROC
#' @export
#'
#' @examples
#' \dontrun{
#' # Load sample data
#' data(sample_data_train)
#'
#' # Example usage
#' result <- tuneandtrainIntSVM(
#'   sample_data_train,
#'   kernel = "linear",
#'   cost_seq = 2^(-15:15),
#'   scale = FALSE,
#'   nfolds = 5,
#'   seed = 123
#' )
#' result$best_cost
#' result$best_model
#' result$final_auc
#' }
tuneandtrainIntSVM <- function(data, kernel = "linear", cost_seq = 2^(-15:15), scale = FALSE, nfolds = 5, seed = 123) {
  # Load necessary libraries
  library(e1071)
  library(mlr)
  library(pROC)
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  
  # Set random seed for reproducibility
  set.seed(seed)
  
  # Split data into predictors and response
  X <- as.matrix(data[, -1])
  y <- as.factor(data[, 1])
  
  # Combine data
  Combined_data <- data.frame(y, X)
  
  # Cross-validation
  partition <- sample(rep(1:nfolds, length.out = nrow(data)))
  auc_CV <- matrix(NA, nrow = length(cost_seq), ncol = nfolds)
  
  for (j in 1:nfolds) {
    XTrain <- Combined_data[partition != j, ]
    XTest <- Combined_data[partition == j, ]
    
    if (length(unique(XTest[, 1])) == 1) {
      auc_CV[, j] <- NA
    } else {
      for (i in seq_along(cost_seq)) {
        cost <- cost_seq[i]
        
        # Fit SVM model
        task <- makeClassifTask(data = rbind(XTrain, XTest), target = names(Combined_data)[1], check.data = FALSE)
        lrn <- makeLearner("classif.svm", predict.type = "prob", kernel = kernel, par.vals = list(cost = cost), scale = scale)
        
        train.set <- 1:nrow(XTrain)
        test.set <- (nrow(XTrain) + 1):nrow(rbind(XTrain, XTest))
        
        model <- train(lrn, task, subset = train.set)
        pred <- predict(model, task = task, subset = test.set)
        
        auc_CV[i, j] <- performance(pred, measures = list(mlr::auc))
      }
    }
  }
  
  # Determine the best cost based on the highest average AUC
  mean_AUC <- rowMeans(auc_CV, na.rm = TRUE)
  best_cost <- cost_seq[which.max(mean_AUC)]
  
  # Train the final model with the best cost
  final_task <- makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
  final_lrn <- makeLearner("classif.svm", predict.type = "prob", kernel = kernel, par.vals = list(cost = best_cost), scale = scale)
  
  final_model <- train(final_lrn, final_task, subset = 1:nrow(data))
  
  # Predict on the training data using the optimal cost value
  pred_SVM_Train <- predict(final_model, task = final_task, subset = 1:nrow(data))
  
  # Calculate AUC on the training data
  AUC_Train <- performance(pred_SVM_Train, measures = list(mlr::auc))
  
  # Return the result
  res <- list(
    best_cost = best_cost,
    best_model = final_model,
    final_auc = AUC_Train
  )
  
  # Set class
  class(res) <- "IntSVM"
  return(res)
}