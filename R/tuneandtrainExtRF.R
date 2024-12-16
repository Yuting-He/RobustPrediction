#' Tune and Train External Random Forest
#'
#' This function tunes and trains a Random Forest classifier using the \code{ranger} package. 
#' It provides two strategies for tuning the \code{min.node.size} parameter based on the \code{estperf} argument:
#' \itemize{
#'   \item When \code{estperf = FALSE} (default): Hyperparameters are tuned using the external validation dataset. 
#'         The \code{min.node.size} value that gives the highest AUC on the external dataset is selected as the best model.
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
#' @param num.trees An integer specifying the number of trees in the Random Forest. Default is 500.
#'
#' @return A list containing the following components:
#'   \itemize{
#'     \item \code{best_min_node_size}: The optimal \code{min.node.size} value determined during the tuning process.
#'     \item \code{best_model}: The trained Random Forest model using the selected \code{min.node.size}.
#'     \item \code{est_auc}: The AUC value evaluated on the external dataset. This is only returned when \code{estperf = TRUE}, 
#'       providing a conservative (slightly pessimistic) estimate of the model's performance.
#'   }
#'
#' @importFrom ranger ranger
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
#' result <- tuneandtrainExtRF(sample_data_train, sample_data_extern, num.trees = 500)
#' print(result$best_min_node_size)  # Optimal min.node.size
#' print(result$best_model)          # Trained Random Forest model
#' # Note: est_auc is not returned when estperf = FALSE
#' 
#' # Example usage with internal tuning and external validation
#' result_internal <- tuneandtrainExtRF(sample_data_train, sample_data_extern, 
#'   estperf = TRUE, num.trees = 500)
#' print(result_internal$best_min_node_size)  # Optimal min.node.size
#' print(result_internal$best_model)          # Trained Random Forest model
#' print(result_internal$est_auc)             # AUC on external validation dataset
#' }

tuneandtrainExtRF <- function(data, dataext, estperf = FALSE, num.trees = 500) {
  
  # Ensure data is in data frame format
  data <- as.data.frame(data)
  dataext <- as.data.frame(dataext)
  
  Train <- data
  Extern <- dataext
  
  # Ensure target column is a factor
  Train[, 1] <- as.factor(Train[, 1])
  Extern[, 1] <- as.factor(Extern[, 1])
  positive_class <- levels(Train[, 1])[2]  # Set the positive class explicitly
  
  if (!estperf) {
    # External tuning
    Combined_data <- rbind(Train, Extern)
    Combined_data[, 1] <- as.factor(Combined_data[, 1])
    
    auc_value <- numeric(nrow(Train) - 1)
    
    for (i in 1:(nrow(Train) - 1)) {
      task <- mlr::makeClassifTask(data = Combined_data, target = names(Combined_data)[1],
                                   positive = positive_class, check.data = FALSE)
      lrn <- mlr::makeLearner("classif.ranger", predict.type = "prob", num.trees = num.trees, 
                              min.node.size = i, save.memory = TRUE)
      train.set <- 1:nrow(Train)
      test.set <- (nrow(Train) + 1):nrow(Combined_data)
      
      model <- mlr::train(lrn, task, subset = train.set)
      pred <- stats::predict(model, task = task, subset = test.set)
      
      auc_value[i] <- mlr::performance(pred, measures = list(mlr::auc))
    }
    
    chosen_min.node.size <- which.max(auc_value)
    final_task <- mlr::makeClassifTask(data = Combined_data, target = names(Combined_data)[1],
                                       positive = positive_class, check.data = FALSE)
    final_lrn <- mlr::makeLearner("classif.ranger", predict.type = "prob", num.trees = num.trees, 
                                  min.node.size = chosen_min.node.size, save.memory = TRUE)
    final_model <- mlr::train(final_lrn, final_task, subset = 1:nrow(Train))
    
    res <- list(
      best_min_node_size = chosen_min.node.size,
      best_model = final_model,
      est_auc = NULL
    )
  } else {
    # Internal tuning
    auc_value <- numeric(nrow(Train) - 1)
    
    for (i in 1:(nrow(Train) - 1)) {
      task <- mlr::makeClassifTask(data = Train, target = names(Train)[1], 
                                   positive = positive_class, check.data = FALSE)
      lrn <- mlr::makeLearner("classif.ranger", predict.type = "prob", num.trees = num.trees, 
                              min.node.size = i, save.memory = TRUE)
      model <- mlr::train(lrn, task)
      pred <- stats::predict(model, task = task)
      
      auc_value[i] <- mlr::performance(pred, measures = list(mlr::auc))
    }
    
    chosen_min.node.size <- which.max(auc_value)
    final_task <- mlr::makeClassifTask(data = Train, target = names(Train)[1],
                                       positive = positive_class, check.data = FALSE)
    final_lrn <- mlr::makeLearner("classif.ranger", predict.type = "prob", num.trees = num.trees, 
                                  min.node.size = chosen_min.node.size, save.memory = TRUE)
    final_model <- mlr::train(final_lrn, final_task)
    
    pred_Extern <- stats::predict(final_model, newdata = Extern)
    est_auc <- mlr::performance(pred_Extern, measures = list(mlr::auc))
    
    res <- list(
      best_min_node_size = chosen_min.node.size,
      best_model = final_model,
      est_auc = est_auc
    )
  }
  
  class(res) <- "ExtRF"
  return(res)
}