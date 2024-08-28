#' Tune and Train Internal Random Forest
#'
#' This function tunes and trains a Random Forest classifier using internal cross-validation. The function evaluates 
#' different `min.node.size` values and selects the best model based on AUC (Area Under the Curve).
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), and the remaining columns should be the predictor variables.
#' @param num.trees An integer specifying the number of trees in the Random Forest. Default is 500.
#' @param nfolds An integer specifying the number of folds for cross-validation. Default is 5.
#' @param seed An integer specifying the random seed for reproducibility. Default is 123.
#'
#' @return A list containing the best `min.node.size` value, the final trained model (`best_model`), and the AUC on the training data (`AUC_Train`).
#' @import ranger
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
#' result <- tuneandtrainIntRF(sample_data_train, num.trees = 500, nfolds = 5, seed = 123)
#' result$best_min.node.size
#' result$best_model
#' result$AUC_Train
#' }
tuneandtrainIntRF <- function(data, num.trees = 500, nfolds = 5, seed = 123) {
  # Load necessary libraries
  library(ranger)
  library(mlr)
  library(pROC)
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  
  # Set random seed for reproducibility
  set.seed(seed)
  
  # Split data into predictors and response
  X <- data[, -1]
  y <- as.factor(data[, 1])
  
  # Combine data
  Combined_data <- cbind(y, X)
  
  # Cross-validation
  partition <- sample(rep(1:nfolds, length.out = nrow(data)))
  auc_CV <- matrix(NA, nrow = nrow(data) - 1, ncol = nfolds)
  
  for (j in 1:nfolds) {
    XTrain <- data[partition != j, ]
    XTest <- data[partition == j, ]
    
    if (length(unique(XTest[, 1])) == 1) {
      auc_CV[, j] <- NA
    } else {
      Combined_data <- rbind(XTrain, XTest)
      Combined_data[, 1] <- as.factor(Combined_data[, 1])
      
      for (i in 1:(nrow(XTrain) - 1)) {
        # Fit Random Forest Model
        task <- makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
        lrn <- makeLearner("classif.ranger", predict.type = "prob", num.threads = 1, num.trees = num.trees, min.node.size = i, save.memory = TRUE)
        
        train.set <- 1:nrow(XTrain)
        test.set <- (nrow(XTrain) + 1):nrow(Combined_data)
        
        model <- train(lrn, task, subset = train.set)
        pred <- predict(model, task = task, subset = test.set)
        
        auc_CV[i, j] <- performance(pred, measures = mlr::auc)
      }
    }
  }
  
  # Determine the best min.node.size based on the highest average AUC
  mean_AUC <- rowMeans(auc_CV, na.rm = TRUE)
  best_min.node.size <- which.max(mean_AUC)
  
  # Train the final model with the best min.node.size
  final_task <- makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
  final_lrn <- makeLearner("classif.ranger", predict.type = "prob", num.trees = num.trees, min.node.size = best_min.node.size, save.memory = TRUE)
  
  final_model <- train(final_lrn, final_task, subset = 1:nrow(data))
  
  # Predict on the training data using the optimal min.node.size
  pred_Lasso_Train <- predict(final_model, task = final_task, subset = 1:nrow(data))
  
  # Calculate AUC on the training data
  AUC_Train <- performance(pred_Lasso_Train, measures = mlr::auc)
  
  # Return the result
  res <- list(
    best_min.node.size = best_min.node.size,
    best_model = final_model,
    AUC_Train = AUC_Train
  )
  
  # Set class
  class(res) <- "IntRF"
  return(res)
}