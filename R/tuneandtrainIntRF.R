#' Tune and Train Internal Random Forest
#'
#' This function tunes and trains a Random Forest classifier using the \code{ranger} package with internal cross-validation. 
#' The function evaluates a sequence of \code{min.node.size} values on the training dataset and selects 
#' the best model based on the Area Under the Curve (AUC).
#'
#' Random Forest constructs multiple decision trees and aggregates their predictions. 
#' The \code{min.node.size} parameter controls the minimum number of samples in each terminal node, affecting model complexity. 
#' This function performs cross-validation within the training dataset to evaluate the impact of different \code{min.node.size} values. 
#' The \code{min.node.size} value that results in the highest AUC is selected as the best model.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), 
#'   and the remaining columns should be the predictor variables.
#' @param num.trees An integer specifying the number of trees in the Random Forest. Default is 500.
#' @param nfolds An integer specifying the number of folds for cross-validation. Default is 5.
#' @param seed An integer specifying the random seed for reproducibility. Default is 123.
#'
#' @return A list containing the best `min.node.size` value (`best_min_node_size`) and
#'   the final trained model (`best_model`).
#' @importFrom ranger ranger
#' @export
#'
#' @examples
#' \donttest{
#' # Load sample data
#' data(sample_data_train)
#'
#' # Example usage
#' result <- tuneandtrainIntRF(sample_data_train, num.trees = 500, nfolds = 5, seed = 123)
#' result$best_min_node_size
#' result$best_model
#' }
tuneandtrainIntRF <- function(data, num.trees = 500, nfolds = 5, seed = 123) {
  
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
        # Fit Random Forest Model using mlr package
        task <- mlr::makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
        lrn <- mlr::makeLearner("classif.ranger", predict.type = "prob", 
                                num.threads = 1, num.trees = num.trees, min.node.size = i, save.memory = TRUE)
        
        train.set <- 1:nrow(XTrain)
        test.set <- (nrow(XTrain) + 1):nrow(Combined_data)
        
        model <- mlr::train(lrn, task, subset = train.set)
        pred <- stats::predict(model, task = task, subset = test.set)
        
        auc_CV[i, j] <- mlr::performance(pred, measures = mlr::auc)
      }
    }
  }
  
  # Determine the best min.node.size based on the highest average AUC
  mean_AUC <- rowMeans(auc_CV, na.rm = TRUE)
  best_min.node.size <- which.max(mean_AUC)
  
  # Train the final model with the best min.node.size
  final_task <- mlr::makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
  final_lrn <- mlr::makeLearner("classif.ranger", predict.type = "prob", 
                                num.trees = num.trees, min.node.size = best_min.node.size, save.memory = TRUE)
  
  final_model <- mlr::train(final_lrn, final_task, subset = 1:nrow(data))
  
  # Return the result
  res <- list(
    best_min_node_size = best_min.node.size,
    best_model = final_model
  )
  
  # Set class
  class(res) <- "IntRF"
  return(res)
}