# ==============================================================================
# CAP6768 Final Project: Retail Store Analytics
# Classification & Forecasting 
# Team Members: Taiwo, Kayla, Fehmida, Vadym, Grace
# ==============================================================================
#install.packages("randomForest")
# ==========================
# Load Required Libraries
# ==========================
#For visualizations :)
library(ggplot2)
library(gridExtra)

#Simply for data manipulation and transformation
library(dplyr)
library(lubridate)


# ==============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION_T.O.
# ==============================================================================
# Load the csv data file for data_analytics
retail <- read.csv("data_analytics_retail.csv")

# Basic overview to show the stucture of the dataset
cat("=== DATA OVERVIEW ===\n")
str(retail)
cat("\n=== FIRST 6 ROWS ===\n")#Showing the first 6 rows
head(retail)
cat("\n=== MISSING VALUES ===\n")#Checking for missing values in each column
colSums(is.na(retail))

# Data cleaning and feature engineering
#Convert date and create additional features (new categorical variables)
retail <- retail %>%
  mutate(
    date = as.Date(date),#Convert our date string to date type
    day_of_week = factor(day_of_week, 
                         levels = c("Monday", "Tuesday", "Wednesday", 
                                    "Thursday", "Friday", "Saturday", "Sunday")),
    month = factor(month, levels = c("Jun", "Jul", "Aug")),
    day_type = ifelse(weekend, "Weekend", "Weekday")
  )

# ==============================================================================
# 2. EXPLORATORY DATA ANALYSIS & VISUALS_T.O.
# ==============================================================================

# Create output directory for saving our plots
if(!dir.exists("plots")) dir.create("plots")

# 2.1 Time Series Plot of Daily Revenue aka daily revenue over time
p1 <- ggplot(retail, aes(x = date, y = daily_revenue)) +
  geom_line(color = "steelblue", size = 0.8) +
  geom_point(aes(color = day_type), size = 2, alpha = 0.7) +
  geom_smooth(method = "loess", color = "red", se = FALSE) +
  scale_color_manual(values = c("Weekday" = "darkorange", "Weekend" = "purple")) +
  labs(title = "Daily Revenue Over Time with Trend Line",
       subtitle = "Red line shows overall trend, Colors indicate weekend/weekday",
       x = "Date", y = "Daily Revenue ($)", color = "Day Type") +
  theme_minimal() +
  theme(legend.position = "top")# Line plot showing revenue trend and daily variation


ggsave("plots/1_time_series_revenue.png", p1, width = 12, height = 6)

# 2.2 Revenue Distribution by Day Type and Day of the Week
p2 <- ggplot(retail, aes(x = day_type, y = daily_revenue, fill = day_type)) +
  geom_boxplot(alpha = 0.7) +
  stat_summary(fun = mean, geom = "point", shape = 18, size = 3, color = "red") +
  scale_fill_manual(values = c("Weekday" = "lightblue", "Weekend" = "lightcoral")) +
  labs(title = "Revenue Distribution: Weekend vs Weekday",
       subtitle = "Red diamond shows mean revenue",
       x = "Day Type", y = "Daily Revenue ($)") +
  theme_minimal() # Boxplot comparing weekday vs weekend revenue


p3 <- ggplot(retail, aes(x = day_of_week, y = daily_revenue, fill = day_of_week)) +
  geom_boxplot() +
  labs(title = "Revenue by Day of Week",
       x = "Day of Week", y = "Daily Revenue ($)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")# Boxplot showing revenue patterns by day of the week


# Combined  both plots into one figure
grid_plot <- grid.arrange(p2, p3, ncol = 2)
ggsave("plots/2_revenue_by_day_type.png", grid_plot, width = 14, height = 6)

# 2.3 Promotion Impact on Revenue Analysis
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

# 2.4 Temperature vs Revenue (with the missing values accounted for)
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
  theme_minimal() # Scatter plot: number of customers vs revenue


p7 <- ggplot(retail, aes(x = avg_transaction, y = daily_revenue)) +
  geom_point(aes(color = day_type), alpha = 0.7, size = 2) +
  geom_smooth(method = "lm", color = "red", se = FALSE) +
  scale_color_manual(values = c("Weekday" = "navy", "Weekend" = "darkred")) +
  labs(title = "Average Transaction vs Revenue",
       x = "Average Transaction Amount ($)", y = "Daily Revenue ($)", color = "Day Type") +
  theme_minimal() # Scatter plot: average transaction vs revenue


ggsave("plots/4_customer_behavior.png", 
       grid.arrange(p6, p7, ncol = 2), width = 14, height = 6)

# ==============================================================================
# 3. DATA PREPARATION FOR MODELING Member 2:
#
# 3.1 Handles the  missing values (temperature)
retail_clean <- retail %>%
  mutate(
    temperature = ifelse(is.na(temperature), mean(temperature, na.rm = TRUE), temperature)
  )

# 3.2 Creates the binary classification target (High Revenue = 1, Low = 0)
median_revenue <- median(retail_clean$daily_revenue)
retail_clean$high_revenue <- ifelse(retail_clean$daily_revenue > median_revenue, 1, 0)

cat("=== CLASSIFICATION TARGET SUMMARY ===\n")
table(retail_clean$high_revenue)
cat("Median Revenue:", median_revenue, "\n")

# 3.3 Creates features for classification
retail_class <- retail_clean %>%
  mutate(
    day_of_week_num = as.numeric(day_of_week),
    month_num = as.numeric(month),
    week_num = week(date)
  ) %>%
  select(high_revenue, daily_customers, avg_transaction, temperature, promotion, 
         weekend, day_of_week_num, month_num, week_num, day_type)

# 3.4 Prepareing time series data for forecasting
ts_data <- retail_clean %>%
  select(date, daily_revenue) %>%
  arrange(date)
# ==============================================================================


#For Data Mdoeling
library(randomForest)
library(xgboost)
library(caret)
library(MLmetrics)
library(tidyr)


# ==============================================================================
# 4. CLASSIFICATION MODELING Member 3
#
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

metrics_long <- metrics %>%
  pivot_longer(cols = c("Accuracy", "Precision", "Recall", "F1_Score"), 
               names_to = "Metric", values_to = "Value")

# Plot model performance
p_model_perf <- ggplot(metrics_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Classification Model Performance",
       subtitle = "Comparison of Accuracy, Precision, Recall, and F1-Score",
       x = "Model", y = "Score") +
  ylim(0,1) +
  theme_minimal() +
  theme(legend.position = "top", axis.text.x = element_text(angle = 15, hjust = 1))

# Display the plot
print(p_model_perf)

# Save the plot
ggsave("plots/6_model_performance.png", p_model_perf, width = 10, height = 6)
# ==============================================================================



# ==============================================================================
# 5. TIME SERIES FORECASTING Member 3
#
#Tips to assits in getting started
# - Prepare time series objects
# - Fit forecasting models (SARIMA, Prophet, etc.)
# - Visualize and compare forecast accuracy
# ==============================================================================



# ==============================================================================
# 6. BUSINESS INSIGHTS AND RECOMMENDATIONS
#
#Tips to assits in getting started
# - Identify key revenue patterns
# - Evaluate promotion effectiveness
# - Recommend staffing strategies based on customer flow
# ==============================================================================



# ==============================================================================
# 7. FINAL RESULTS SUMMARY
#
#Tips to assits in getting started
# Alternative approach for forecasting accuracy
# ==============================================================================


# ==============================================================================
# 8. SAVE ALL RESULTS 
#
#Tips to assits in getting started
# Save model results
# Save summary statistics
# ==============================================================================

