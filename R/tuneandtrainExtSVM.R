#' Tune and Train External SVM
#'
#' This function tunes and trains a Support Vector Machine (SVM) classifier using the \code{mlr} package. 
#' The function evaluates a sequence of cost values on an external validation dataset and selects 
#' the best model based on the Area Under the Curve (AUC).
#'
#' In Support Vector Machines, The \code{cost} parameter controls the trade-off between 
#' achieving a low training error and a low testing error. 
#' This function trains an SVM model on the training dataset and validates it using the external validation dataset. 
#' The cost value that results in the highest AUC on the external validation dataset is chosen as the best model.
#'
#' @param data A data frame containing the training data. The first column should be the response variable (factor), 
#'   and the remaining columns should be the predictor variables.
#' @param dataext A data frame containing the external validation data. The first column should be the response 
#'   variable (factor), and the remaining columns should be the predictor variables.
#' @param kernel A character string specifying the kernel type to be used in the SVM. Default is "linear".
#' @param cost_seq A numeric vector of cost values to be evaluated. Default is `2^(-15:15)`.
#' @param scale A logical indicating whether to scale the predictor variables. Default is FALSE.
#'
#' @return A list containing the best cost value (`best_cost`), the final trained model (`best_model`), 
#'   the AUC on the training data (`final_auc`).
#' @importFrom e1071 svm
#' @export
#'
#' @examples
#' \donttest{
#' # Load sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage
#' result <- tuneandtrainExtSVM(sample_data_train, sample_data_extern, kernel = "linear", 
#'   cost_seq = 2^(-15:15), scale = FALSE)
#' result$best_cost
#' result$best_model
#' result$final_auc
#' }
tuneandtrainExtSVM <- function(data, dataext, kernel = "linear", cost_seq = 2^(-15:15), scale = FALSE) {
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  dataext <- as.data.frame(dataext)
  
  Train <- data
  Extern <- dataext
  
  Combined_data <- rbind(Train, Extern)
  Combined_data[, 1] <- as.factor(Combined_data[, 1])
  
  # Initialize AUC vector
  auc_value <- numeric(length(cost_seq))
  
  # Define AUC measure from mlr package
  auc_measure <- mlr::auc
  
  # Tune cost parameter
  for (i in seq_along(cost_seq)) {
    cost <- cost_seq[i]
    
    # Fit SVM model using mlr package
    task <- mlr::makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
    lrn <- mlr::makeLearner("classif.svm", predict.type = "prob", kernel = kernel, cost = cost, scale = scale)
    
    train.set <- 1:nrow(Train)
    test.set <- (nrow(Train) + 1):nrow(Combined_data)
    
    model <- mlr::train(lrn, task, subset = train.set)
    pred <- stats::predict(model, task = task, subset = test.set)
    
    # Calculate AUC using mlr package
    auc_value[i] <- mlr::performance(pred, measures = auc_measure)
  }
  
  # Choose the best cost
  chosen_cost <- cost_seq[which.max(auc_value)]
  
  # Train the final model with the chosen cost
  final_task <- mlr::makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
  final_lrn <- mlr::makeLearner("classif.svm", predict.type = "prob", kernel = kernel, cost = chosen_cost, scale = scale)
  
  final_model <- mlr::train(final_lrn, final_task, subset = 1:nrow(Train))
  
  # Predict on the external validation data
  pred_Extern <- stats::predict(final_model, task = final_task, subset = (nrow(Train) + 1):nrow(Combined_data))
  
  # Calculate AUC for external validation
  AUC_Extern <- mlr::performance(pred_Extern, measures = list(mlr::auc))
  
  # Predict on the training data
  pred_Train <- stats::predict(final_model, task = final_task, subset = 1:nrow(Train))
  AUC_Train <- mlr::performance(pred_Train, measures = list(mlr::auc))
  
  # Return the result
  res <- list(
    best_cost = chosen_cost,
    best_model = final_model,
    final_auc = AUC_Train  # This is the training set AUC
  )
  
  # Set class
  class(res) <- "ExtSVM"
  return(res)
}