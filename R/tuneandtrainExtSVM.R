#' Tune and Train External SVM
#'
#' This function tunes and trains a Support Vector Machine (SVM) classifier using the \code{mlr} package. 
#' It provides two strategies for tuning the cost parameter based on the \code{estperf} argument:
#' \itemize{
#'   \item When \code{estperf = FALSE} (default): Hyperparameters are tuned using the external validation dataset. 
#'         The \code{cost} value that gives the highest AUC on the external dataset is selected as the best model.
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
#' @param kernel A character string specifying the kernel type to be used in the SVM. Default is \code{"linear"}.
#' @param cost_seq A numeric vector specifying the sequence of cost values to evaluate. Default is \code{2^(-15:15)}.
#' @param scale A logical value indicating whether to scale the predictor variables. Default is \code{FALSE}.
#'
#' @return A list containing the following components:
#'   \itemize{
#'     \item \code{best_cost}: The optimal cost value determined during the tuning process.
#'     \item \code{best_model}: The trained SVM model using the selected \code{cost}.
#'     \item \code{est_auc}: The AUC value evaluated on the external dataset. This is only returned when \code{estperf = TRUE}, 
#'       providing a conservative (slightly pessimistic) estimate of the model's performance.
#'   }
#'
#' @importFrom mlr makeClassifTask makeLearner train performance
#' @export
#'
#' @examples
#' \donttest{
#' # Load sample data
#' data(sample_data_train)
#' data(sample_data_extern)
#'
#' # Example usage with external tuning (default)
#' result <- tuneandtrainExtSVM(sample_data_train, sample_data_extern, kernel = "linear", 
#'   cost_seq = 2^(-15:15), scale = FALSE)
#' print(result$best_cost)        # Optimal cost
#' print(result$best_model)       # Final trained model
#' # Note: est_auc is not returned when estperf = FALSE
#'
#' # Example usage with internal tuning and external validation
#' result_internal <- tuneandtrainExtSVM(sample_data_train, sample_data_extern, 
#'   estperf = TRUE, kernel = "linear", cost_seq = 2^(-15:15), scale = FALSE)
#' print(result_internal$best_cost)  # Optimal cost
#' print(result_internal$best_model) # Final trained model
#' print(result_internal$est_auc)    # AUC on external validation dataset
#' }

tuneandtrainExtSVM <- function(data, dataext, estperf = FALSE, kernel = "linear", cost_seq = 2^(-15:15), scale = FALSE) {
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  dataext <- as.data.frame(dataext)
  
  Train <- data
  Extern <- dataext
  
  # Explicitly convert target variable to factor
  Train[, 1] <- as.factor(Train[, 1])
  Extern[, 1] <- as.factor(Extern[, 1])
  
  if (!estperf) {
    # External tuning: tune cost based on external validation AUC
    Combined_data <- rbind(Train, Extern)
    Combined_data[, 1] <- as.factor(Combined_data[, 1])
    
    auc_value <- numeric(length(cost_seq))
    
    for (i in seq_along(cost_seq)) {
      cost <- cost_seq[i]
      task <- mlr::makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
      lrn <- mlr::makeLearner("classif.svm", predict.type = "prob", kernel = kernel, cost = cost, scale = scale)
      
      train.set <- 1:nrow(Train)
      test.set <- (nrow(Train) + 1):nrow(Combined_data)
      
      model <- mlr::train(lrn, task, subset = train.set)
      pred <- stats::predict(model, task = task, subset = test.set)
      
      # Calculate AUC
      auc_value[i] <- mlr::performance(pred, measures = list(mlr::auc))
    }
    
    # Select the best cost
    chosen_cost <- cost_seq[which.max(auc_value)]
    
    # Train the final model with the chosen cost
    final_task <- mlr::makeClassifTask(data = Combined_data, target = names(Combined_data)[1], check.data = FALSE)
    final_lrn <- mlr::makeLearner("classif.svm", predict.type = "prob", kernel = kernel, cost = chosen_cost, scale = scale)
    final_model <- mlr::train(final_lrn, final_task, subset = 1:nrow(Train))
    
    # Return result without AUC
    res <- list(
      best_cost = chosen_cost,
      best_model = final_model,
      est_auc = NULL  # No AUC returned for estperf = FALSE
    )
  } else {
    # Internal tuning: tune cost based on training data AUC
    auc_value <- numeric(length(cost_seq))
    
    for (i in seq_along(cost_seq)) {
      cost <- cost_seq[i]
      task <- mlr::makeClassifTask(data = Train, target = names(Train)[1], check.data = FALSE)
      lrn <- mlr::makeLearner("classif.svm", predict.type = "prob", kernel = kernel, cost = cost, scale = scale)
      
      model <- mlr::train(lrn, task)
      pred <- stats::predict(model, task = task)
      
      # Calculate AUC on training data
      auc_value[i] <- mlr::performance(pred, measures = list(mlr::auc))
    }
    
    # Select the best cost
    chosen_cost <- cost_seq[which.max(auc_value)]
    
    # Train the final model with the chosen cost
    final_task <- mlr::makeClassifTask(data = Train, target = names(Train)[1], check.data = FALSE)
    final_lrn <- mlr::makeLearner("classif.svm", predict.type = "prob", kernel = kernel, cost = chosen_cost, scale = scale)
    final_model <- mlr::train(final_lrn, final_task)
    
    # Validate the model on external data
    pred_Extern <- stats::predict(final_model, newdata = Extern)
    est_auc <- mlr::performance(pred_Extern, measures = list(mlr::auc))
    
    # Return result with AUC
    res <- list(
      best_cost = chosen_cost,
      best_model = final_model,
      est_auc = est_auc  # Conservative AUC estimate
    )
  }
  
  # Set class
  class(res) <- "ExtSVM"
  return(res)
}