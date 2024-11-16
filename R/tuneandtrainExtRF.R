#' Tune and Train External Random Forest
#'
#' This function tunes and trains a Random Forest classifier using the \code{ranger} package. The function 
#' evaluates a sequence of \code{min.node.size} values on an external validation dataset and selects 
#' the best model based on the Area Under the Curve (AUC).
#'
#' Random Forest is an ensemble learning method that constructs multiple decision trees and aggregates their predictions. 
#' The \code{min.node.size} parameter controls the minimum number of samples in each terminal node, affecting model complexity. 
#' This function trains a Random Forest model on the training dataset and validates it using the external validation dataset. 
#' The \code{min.node.size} value that results in the highest AUC on the external validation dataset is chosen as the best model.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), 
#'   and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response 
#'   variable (factor), and the remaining columns should be the predictor variables.
#' @param num.trees An integer specifying the number of trees in the Random Forest. Default is 500.
#'
#' @return A list containing the best `min.node.size` value(`best_min_node_size`), 
#'   the final trained model (`best_model`), and the AUC of the final model (`final_auc`).
#' @importFrom ranger ranger
#' @export
#'
#' @examples
#' \donttest{
#' # Load sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage
#' result <- tuneandtrainExtRF(sample_data_train, sample_data_extern, num.trees = 500)
#' result$best_min_node_size
#' result$best_model
#' result$final_auc
#' }
tuneandtrainExtRF <- function(data, dataext, num.trees = 500) {
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  dataext <- as.data.frame(dataext)
  
  Train <- data
  Extern <- dataext
  
  Combined_data <- rbind(Train, Extern)
  Combined_data[, 1] <- as.factor(Combined_data[, 1])
  
  # Initialize AUC vector
  auc_value <- numeric(nrow(Train) - 1)
  
  # Tune min.node.size parameter using mlr and ranger
  for (i in 1:(nrow(Train) - 1)) {
    # Fit Random Forest model
    task <- mlr::makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
    lrn <- mlr::makeLearner("classif.ranger", predict.type = "prob", num.threads = 1, 
                            num.trees = num.trees, min.node.size = i, save.memory = TRUE)
    
    train.set <- 1:nrow(Train)
    test.set <- (nrow(Train) + 1):nrow(Combined_data)
    
    model <- mlr::train(lrn, task, subset = train.set)
    pred <- stats::predict(model, task = task, subset = test.set)
    
    auc_value[i] <- mlr::performance(pred, measures = list(mlr::auc))
  }
  
  chosen_min.node.size <- which.max(auc_value)
  
  # Train the final model with the chosen min.node.size
  final_task <- mlr::makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
  final_lrn <- mlr::makeLearner("classif.ranger", predict.type = "prob", num.trees = num.trees, 
                                min.node.size = chosen_min.node.size, save.memory = TRUE)
  
  final_model <- mlr::train(final_lrn, final_task, subset = 1:nrow(Train))
  
  # Calculate AUC on the external validation set with the final model
  pred_final <- stats::predict(final_model, newdata = Extern)
  final_auc <- mlr::performance(pred_final, measures = list(mlr::auc))
  
  # Return the result
  res <- list(
    best_min_node_size = chosen_min.node.size,
    best_model = final_model,
    final_auc = final_auc
  )
  
  # Set class
  class(res) <- "ExtRF"
  return(res)
}