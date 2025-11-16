# ==============================================================================
# CAP6768 Final Project: Retail Store Analytics
# Classification & Forecasting for Business Optimization
# ==============================================================================

# Load required libraries
library(ggplot2)
library(dplyr)
library(lubridate)
library(forecast)
library(prophet)
library(caret)
library(randomForest)
library(xgboost)
library(vip)
library(tseries)
library(MLmetrics)
library(gridExtra)
library(scales)
# install.packages("Ckmeans.1d.dp")
library(Ckmeans.1d.dp)

# Set random seed for reproducibility
set.seed(123)

# ==============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION_Member 2
# ==============================================================================

# Load the data
retail <- read.csv("data_analytics_retail.csv")

# Basic overview
cat("=== DATA OVERVIEW ===\n")
str(retail)
cat("\n=== FIRST 6 ROWS ===\n")
head(retail)
cat("\n=== MISSING VALUES ===\n")
colSums(is.na(retail))

# Convert date and create additional features
retail <- retail %>%
  mutate(
    date = as.Date(date),
    day_of_week = factor(day_of_week, 
                         levels = c("Monday", "Tuesday", "Wednesday", 
                                    "Thursday", "Friday", "Saturday", "Sunday")),
    month = factor(month, levels = c("Jun", "Jul", "Aug")),
    day_type = ifelse(weekend, "Weekend", "Weekday")
  )

# ==============================================================================
# 2. EXPLORATORY DATA ANALYSIS & VISUALS Member 2
# ==============================================================================

# Create output directory for plots
if(!dir.exists("plots")) dir.create("plots")

# 2.1 Time Series Plot of Daily Revenue
p1 <- ggplot(retail, aes(x = date, y = daily_revenue)) +
  geom_line(color = "steelblue", size = 0.8) +
  geom_point(aes(color = day_type), size = 2, alpha = 0.7) +
  geom_smooth(method = "loess", color = "red", se = FALSE) +
  scale_color_manual(values = c("Weekday" = "darkorange", "Weekend" = "purple")) +
  labs(title = "Daily Revenue Over Time with Trend Line",
       subtitle = "Red line shows overall trend, Colors indicate weekend/weekday",
       x = "Date", y = "Daily Revenue ($)", color = "Day Type") +
  theme_minimal() +
  theme(legend.position = "top")

ggsave("plots/1_time_series_revenue.png", p1, width = 12, height = 6)

# 2.2 Revenue Distribution by Day Type
p2 <- ggplot(retail, aes(x = day_type, y = daily_revenue, fill = day_type)) +
  geom_boxplot(alpha = 0.7) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "red") +
  scale_fill_manual(values = c("Weekday" = "lightblue", "Weekend" = "lightcoral")) +
  labs(title = "Revenue Distribution: Weekend vs Weekday",
       subtitle = "Red diamond shows mean revenue",
       x = "Day Type", y = "Daily Revenue ($)") +
  theme_minimal()

p3 <- ggplot(retail, aes(x = day_of_week, y = daily_revenue, fill = day_of_week)) +
  geom_boxplot() +
  labs(title = "Revenue by Day of Week",
       x = "Day of Week", y = "Daily Revenue ($)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

# Combine plots
grid_plot <- grid.arrange(p2, p3, ncol = 2)
ggsave("plots/2_revenue_by_day_type.png", grid_plot, width = 14, height = 6)

# 2.3 Promotion Impact Analysis
p4 <- ggplot(retail, aes(x = factor(promotion, labels = c("No Promotion", "Promotion")), 
                         y = daily_revenue, fill = factor(promotion))) +
  geom_boxplot(alpha = 0.7) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "red") +
  scale_fill_manual(values = c("FALSE" = "lightgreen", "TRUE" = "orange")) +
  labs(title = "Impact of Promotions on Daily Revenue",
       subtitle = "Red diamond shows mean revenue",
       x = "Promotion Status", y = "Daily Revenue ($)") +
  theme_minimal() +
  theme(legend.position = "none")

# 2.4 Temperature vs Revenue (with missing values handled)
retail_temp <- retail %>% filter(!is.na(temperature))

p5 <- ggplot(retail_temp, aes(x = temperature, y = daily_revenue)) +
  geom_point(aes(color = day_type), alpha = 0.7, size = 2) +
  geom_smooth(method = "lm", color = "red", se = TRUE) +
  scale_color_manual(values = c("Weekday" = "blue", "Weekend" = "red")) +
  labs(title = "Temperature vs Daily Revenue",
       subtitle = "Colored by day type, Red line shows linear relationship",
       x = "Temperature (°F)", y = "Daily Revenue ($)", color = "Day Type") +
  theme_minimal()

ggsave("plots/3_promotion_temperature_analysis.png", 
       grid.arrange(p4, p5, ncol = 2), width = 14, height = 6)

# 2.5 Customer Behavior Analysis
p6 <- ggplot(retail, aes(x = daily_customers, y = daily_revenue)) +
  geom_point(aes(color = day_type), alpha = 0.7, size = 2) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  scale_color_manual(values = c("Weekday" = "navy", "Weekend" = "darkred")) +
  labs(title = "Daily Customers vs Revenue",
       subtitle = "Strong positive correlation expected",
       x = "Number of Customers", y = "Daily Revenue ($)", color = "Day Type") +
  theme_minimal()

p7 <- ggplot(retail, aes(x = avg_transaction, y = daily_revenue)) +
  geom_point(aes(color = day_type), alpha = 0.7, size = 2) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  scale_color_manual(values = c("Weekday" = "navy", "Weekend" = "darkred")) +
  labs(title = "Average Transaction vs Revenue",
       x = "Average Transaction Amount ($)", y = "Daily Revenue ($)", color = "Day Type") +
  theme_minimal()

ggsave("plots/4_customer_behavior.png", 
       grid.arrange(p6, p7, ncol = 2), width = 14, height = 6)

# ==============================================================================
# 3. DATA PREPARATION FOR MODELING Member 2
# ==============================================================================

# 3.1 Handle missing values (temperature)
retail_clean <- retail %>%
  mutate(
    temperature = ifelse(is.na(temperature), mean(temperature, na.rm = TRUE), temperature)
  )

# 3.2 Create binary classification target (High Revenue = 1, Low = 0)
median_revenue <- median(retail_clean$daily_revenue)
retail_clean$high_revenue <- ifelse(retail_clean$daily_revenue > median_revenue, 1, 0)

cat("=== CLASSIFICATION TARGET SUMMARY ===\n")
table(retail_clean$high_revenue)
cat("Median Revenue:", median_revenue, "\n")

# 3.3 Create features for classification
retail_class <- retail_clean %>%
  mutate(
    day_of_week_num = as.numeric(day_of_week),
    month_num = as.numeric(month),
    week_num = week(date)
  ) %>%
  select(high_revenue, daily_customers, avg_transaction, temperature, promotion, 
         weekend, day_of_week_num, month_num, week_num, day_type)

# 3.4 Prepare time series data for forecasting
ts_data <- retail_clean %>%
  select(date, daily_revenue) %>%
  arrange(date)

# ==============================================================================
# 4. CLASSIFICATION MODELING Member 3
# ==============================================================================

# 4.1 Split data into training and testing sets (time-based split)
train_idx <- 1:75  # First 75 days for training
test_idx <- 76:90  # Last 15 days for testing

train_data <- retail_class[train_idx, ]
test_data <- retail_class[test_idx, ]

cat("=== TRAIN/TEST SPLIT ===\n")
cat("Training days:", nrow(train_data), "\n")
cat("Testing days:", nrow(test_data), "\n")

# 4.2 Logistic Regression
logit_model <- glm(high_revenue ~ ., 
                   data = train_data %>% select(-day_type), 
                   family = binomial)

logit_pred <- predict(logit_model, newdata = test_data, type = "response")
logit_pred_class <- ifelse(logit_pred > 0.5, 1, 0)

# 4.3 Random Forest
rf_model <- randomForest(factor(high_revenue) ~ ., 
                         data = train_data %>% select(-day_type),
                         ntree = 500, importance = TRUE)

rf_pred <- predict(rf_model, newdata = test_data, type = "prob")[,2]
rf_pred_class <- predict(rf_model, newdata = test_data)

# 4.4 XGBoost
# Prepare data for XGBoost
xgb_train <- model.matrix(high_revenue ~ . -1, data = train_data %>% select(-day_type))
xgb_test <- model.matrix(high_revenue ~ . -1, data = test_data %>% select(-day_type))

xgb_model <- xgboost(
  data = xgb_train,
  label = train_data$high_revenue,
  nrounds = 100,
  objective = "binary:logistic",
  verbose = 0
)

xgb_pred <- predict(xgb_model, xgb_test)
xgb_pred_class <- ifelse(xgb_pred > 0.5, 1, 0)

# 4.5 Model Evaluation
results <- data.frame(
  Actual = test_data$high_revenue,
  Logistic = logit_pred_class,
  RandomForest = as.numeric(as.character(rf_pred_class)),
  XGBoost = xgb_pred_class
)

# Calculate metrics
metrics <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  Accuracy = c(
    Accuracy(results$Logistic, results$Actual),
    Accuracy(results$RandomForest, results$Actual),
    Accuracy(results$XGBoost, results$Actual)
  ),
  Precision = c(
    Precision(results$Actual, results$Logistic),
    Precision(results$Actual, results$RandomForest),
    Precision(results$Actual, results$XGBoost)
  ),
  Recall = c(
    Recall(results$Actual, results$Logistic),
    Recall(results$Actual, results$RandomForest),
    Recall(results$Actual, results$XGBoost)
  ),
  F1_Score = c(
    F1_Score(results$Actual, results$Logistic),
    F1_Score(results$Actual, results$RandomForest),
    F1_Score(results$Actual, results$XGBoost)
  )
)

cat("=== CLASSIFICATION MODEL PERFORMANCE ===\n")
print(metrics)

# 4.6 Feature Importance
# Random Forest Importance
rf_importance <- importance(rf_model)
rf_imp_df <- data.frame(
  Feature = rownames(rf_importance),
  Importance = rf_importance[, "MeanDecreaseGini"]
) %>% arrange(desc(Importance))

p8 <- ggplot(rf_imp_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
  coord_flip() +
  labs(title = "Random Forest Feature Importance",
       subtitle = "Higher values indicate more important features",
       x = "Features", y = "Importance (Mean Decrease Gini)") +
  theme_minimal()

# XGBoost Importance
xgb_importance <- xgb.importance(model = xgb_model)
p9 <- xgb.ggplot.importance(xgb_importance) +
  labs(title = "XGBoost Feature Importance") +
  theme_minimal()

ggsave("plots/5_feature_importance.png", 
       grid.arrange(p8, p9, ncol = 2), width = 14, height = 6)

# ==============================================================================
# 5. TIME SERIES FORECASTING Member 3
# ==============================================================================

# 5.1 Prepare time series object
ts_revenue <- ts(ts_data$daily_revenue, frequency = 7)  # Weekly seasonality

# 5.2 SARIMA Model
sarima_model <- auto.arima(ts_revenue, seasonal = TRUE)
sarima_forecast <- forecast(sarima_model, h = 7)

# 5.3 Prophet Model
prophet_data <- data.frame(
  ds = ts_data$date,
  y = ts_data$daily_revenue
)

prophet_model <- prophet(prophet_data, weekly.seasonality = TRUE)
future <- make_future_dataframe(prophet_model, periods = 7)
prophet_forecast <- predict(prophet_model, future)

# 5.4 Simple Linear Model with Trend and Seasonality
lm_data <- ts_data %>%
  mutate(
    trend = 1:n(),
    day_of_week = factor(weekdays(date)),
    weekend = weekdays(date) %in% c("Saturday", "Sunday")
  )

lm_model <- lm(daily_revenue ~ trend + day_of_week, data = lm_data)
lm_future <- data.frame(
  trend = (nrow(lm_data)+1):(nrow(lm_data)+7),
  day_of_week = factor(weekdays(seq(max(ts_data$date)+1, by = "day", length.out = 7)))
)
lm_forecast <- predict(lm_model, newdata = lm_future)

# 5.5 Forecasting Visualization -Member 4
# SARIMA plot
p10 <- autoplot(sarima_forecast) +
  labs(title = "SARIMA Forecast - Next 7 Days",
       x = "Time", y = "Daily Revenue ($)") +
  theme_minimal()

# Prophet plot
p11 <- plot(prophet_model, prophet_forecast) +
  labs(title = "Prophet Forecast - Next 7 Days") +
  theme_minimal()

# Combined forecast comparison
forecast_dates <- seq(max(ts_data$date)+1, by = "day", length.out = 7)
forecast_comparison <- data.frame(
  Date = forecast_dates,
  SARIMA = as.numeric(sarima_forecast$mean),
  Prophet = tail(prophet_forecast$yhat, 7),
  Linear = lm_forecast
)

p12 <- forecast_comparison %>%
  pivot_longer(cols = -Date, names_to = "Model", values_to = "Forecast") %>%
  ggplot(aes(x = Date, y = Forecast, color = Model, group = Model)) +
  geom_line(size = 1) +
  geom_point(size = 2) +
  labs(title = "Comparison of Forecasting Models - Next 7 Days",
       subtitle = "Different approaches to revenue prediction",
       y = "Predicted Daily Revenue ($)", color = "Model") +
  theme_minimal() +
  scale_color_brewer(palette = "Set1")

ggsave("plots/6_forecasting_comparison.png", 
       grid.arrange(p10, p11, p12, ncol = 1), width = 12, height = 15)

# ==============================================================================
# 6. BUSINESS INSIGHTS AND RECOMMENDATIONS
# ==============================================================================

# 6.1 Key Statistics Summary
cat("=== KEY BUSINESS INSIGHTS ===\n")

# Revenue patterns
weekend_stats <- retail_clean %>%
  group_by(weekend) %>%
  summarise(
    Avg_Revenue = mean(daily_revenue),
    Avg_Customers = mean(daily_customers),
    Avg_Transaction = mean(avg_transaction),
    Count = n()
  )

print(weekend_stats)

# Promotion effectiveness
promotion_stats <- retail_clean %>%
  group_by(promotion) %>%
  summarise(
    Avg_Revenue = mean(daily_revenue),
    Revenue_Increase = (mean(daily_revenue) - median_revenue) / median_revenue * 100
  )

print(promotion_stats)

# 6.2 Best Performing Days Analysis
day_stats <- retail_clean %>%
  group_by(day_of_week) %>%
  summarise(
    Avg_Revenue = mean(daily_revenue),
    High_Revenue_Rate = mean(high_revenue),
    Count = n()
  ) %>%
  arrange(desc(Avg_Revenue))

print(day_stats)

# 6.3 Staffing Optimization Recommendation
# Calculate required staff based on customers (assuming 1 staff per 20 customers)
staffing_recommendation <- retail_clean %>%
  mutate(
    required_staff = ceiling(daily_customers / 20),
    day_type = ifelse(weekend, "Weekend", "Weekday")
  ) %>%
  group_by(day_type) %>%
  summarise(
    avg_staff_needed = mean(required_staff),
    max_staff_needed = max(required_staff)
  )

print(staffing_recommendation)

# ==============================================================================
# 7. FINAL RESULTS SUMMARY
# ==============================================================================

cat("=== FINAL PROJECT SUMMARY ===\n")
cat("Classification Task:\n")
cat("- Best Model:", metrics$Model[which.max(metrics$F1_Score)], "\n")
cat("- Best Accuracy:", round(max(metrics$Accuracy), 3), "\n")
cat("- Best F1-Score:", round(max(metrics$F1_Score), 3), "\n\n")

cat("Forecasting Task:\n")
if(exists("sarima_model")) {
  sarima_accuracy <- accuracy(sarima_model)
  cat("- SARIMA RMSE:", round(sarima_accuracy["RMSE"], 2), "\n")
  cat("- SARIMA MAPE:", round(sarima_accuracy["MAPE"], 2), "%\n")
} else {
  cat("- SARIMA model not available\n")
}

# Alternative approach for forecasting accuracy
if(exists("sarima_forecast") && exists("ts_revenue")) {
  # Calculate out-of-sample accuracy if we have test data
  train_size <- floor(0.8 * length(ts_revenue))
  train_ts <- ts(ts_revenue[1:train_size], frequency = 7)
  test_ts <- ts_revenue[(train_size+1):length(ts_revenue)]
  
  # Fit model on training data
  sarima_train <- auto.arima(train_ts, seasonal = TRUE)
  sarima_test_forecast <- forecast(sarima_train, h = length(test_ts))
  
  # Calculate test accuracy
  test_accuracy <- accuracy(sarima_test_forecast, test_ts)
  cat("- SARIMA Test RMSE:", round(test_accuracy["Test set", "RMSE"], 2), "\n")
  cat("- SARIMA Test MAPE:", round(test_accuracy["Test set", "MAPE"], 2), "%\n")
}
cat("Key Business Findings:\n")
cat("- Weekend revenue is", 
    round((weekend_stats$Avg_Revenue[2] - weekend_stats$Avg_Revenue[1]) / weekend_stats$Avg_Revenue[1] * 100, 1), 
    "% higher than weekdays\n")
cat("- Promotions increase revenue by", 
    round(promotion_stats$Revenue_Increase[2], 1), "% on average\n")
cat("- Best performing day:", day_stats$day_of_week[1], 
    "with average revenue of $", round(day_stats$Avg_Revenue[1], 2), "\n")

# ==============================================================================
# 8. SAVE ALL RESULTS Member 2
# ==============================================================================

# Save model results
save(logit_model, rf_model, xgb_model, sarima_model, prophet_model, 
     metrics, forecast_comparison, file = "retail_models_results.RData")

# Save summary statistics
write.csv(metrics, "classification_metrics.csv", row.names = FALSE)
write.csv(forecast_comparison, "forecast_predictions.csv", row.names = FALSE)
write.csv(day_stats, "day_analysis.csv", row.names = FALSE)

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("All results saved to:\n")
cat("- Model results: retail_models_results.RData\n")
cat("- Classification metrics: classification_metrics.csv\n")
cat("- Forecast predictions: forecast_predictions.csv\n")
cat("- Day analysis: day_analysis.csv\n")
cat("- Visualizations: plots/ directory\n")

