#' Tune and Train External Random Forest
#'
#' This function tunes and trains a Random Forest classifier using an external validation dataset. The function 
#' tunes the `min.node.size` parameter based on the performance on the external dataset.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response variable (factor), and the remaining columns should be the predictor variables.
#' @param num.trees An integer specifying the number of trees in the Random Forest. Default is 500.
#'
#' @return A list containing the best `min.node.size` value, the final trained model (`best_model`), and the AUC of the final model (`final_auc`).
#' @import ranger
#' @import mlr
#' @import pROC
#' @export
#'
#' @examples
#' \dontrun{
#' # Load sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage
#' result <- tuneandtrainExtRF(sample_data_train, sample_data_extern, num.trees = 500)
#' result$best_min.node.size
#' result$best_model
#' result$final_auc
#' }
tuneandtrainExtRF <- function(data, dataext, num.trees = 500) {
  # Load necessary libraries
  library(ranger)
  library(mlr)
  library(pROC)
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  dataext <- as.data.frame(dataext)
  
  Train <- data
  Extern <- dataext
  
  Combined_data <- rbind(Train, Extern)
  Combined_data[, 1] <- as.factor(Combined_data[, 1])
  
  # Initialize AUC vector
  auc_value <- numeric(nrow(Train) - 1)
  
  # Tune min.node.size parameter
  for (i in 1:(nrow(Train) - 1)) {
    # Fit Random Forest model
    task <- makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
    lrn <- makeLearner("classif.ranger", predict.type = "prob", num.threads = 1, num.trees = num.trees, min.node.size = i, save.memory = TRUE)
    
    train.set <- 1:nrow(Train)
    test.set <- (nrow(Train) + 1):nrow(Combined_data)
    
    model <- train(lrn, task, subset = train.set)
    pred <- predict(model, task = task, subset = test.set)
    
    auc_value[i] <- performance(pred, measures = list(mlr::auc))
  }
  
  chosen_min.node.size <- which.max(auc_value)
  
  # Train the final model with the chosen min.node.size
  final_task <- makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
  final_lrn <- makeLearner("classif.ranger", predict.type = "prob", num.trees = num.trees, min.node.size = chosen_min.node.size, save.memory = TRUE)
  
  final_model <- train(final_lrn, final_task, subset = 1:nrow(Train))
  
  # Calculate AUC on the external validation set with the final model
  pred_final <- predict(final_model, newdata = Extern)
  final_auc <- performance(pred_final, measures = list(mlr::auc))
  
  # Return the result
  res <- list(
    best_min.node.size = chosen_min.node.size,
    best_model = final_model,
    final_auc = final_auc
  )

  # Set class
  class(res) <- "ExtRF"
  return(res)
}