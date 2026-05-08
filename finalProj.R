# ==============================
# STAT 385 FINAL PROJECT
# GROUP MEMBERS:
# Mahi Patel
# Usman Tahir
# Mustafah Khan
# Rahee Chauhan

# ==============================
# RESEARCH QUESTION
# ==============================
# Which weather variables (temperature, humidity, or wind speed) 
# are the best predictors of Zone 1 power consumption in Tetouan City?
#
# We will address this question using:
# - Multiple Linear Regression
# - LASSO Regression
# - Decision Tree Regression
# - Random Forest
# We will compare model performance using RMSE on the same test set.
# ==============================



# ==============================
# Data Cleaning + EDA
# ==============================

library(ggplot2)
library(lubridate)
library(corrplot)


# 1. Loading data

getwd()
setwd("C:/Users/itsma/OneDrive - University of Illinois Chicago/Desktop/STAT 385/Proejct")
data <- read.csv("Tetuan City power consumption.csv", stringsAsFactors = FALSE)


# 2. Cleaning column names

colnames(data) <- make.names(colnames(data))
print(colnames(data))  


# 3. Converting datetime

data$DateTime <- as.POSIXct(data$DateTime, format = "%m/%d/%Y %H:%M")


# 4. Checking missing values

missing_values <- colSums(is.na(data))
cat("Missing values per column:\n")
print(missing_values)  # Only 6 rows had missing DateTime values

# Remove missing values
data <- na.omit(data)


# 5. Checking duplicates

duplicates <- sum(duplicated(data))
cat("\nNumber of duplicate rows:", duplicates, "\n")  # No duplicate rows found

# Removing duplicates if any
data <- data[!duplicated(data), ]


# 6. Creating time variables

data$Hour <- hour(data$DateTime)
data$Day <- day(data$DateTime)
data$Month <- month(data$DateTime)
data$Year <- year(data$DateTime)# Added time features for later analysis

write.csv(data, "cleaned_tetuan_data.csv", row.names = FALSE) #new csv cleaned 

# 7. Basic inspection

cat("\nDataset dimensions:\n")
print(dim(data))  # Final cleaned dataset has 52,410 rows and 13 columns

cat("\nColumn names:\n")
print(colnames(data))

cat("\nStructure:\n")
str(data)

cat("\nSummary statistics:\n")
summary(data)  # Summary statistics for all variables


# 8. Histograms

par(mfrow = c(2, 2))

hist(data$Temperature,
     main = "Temperature",
     xlab = "Temperature",
     col = "lightblue")  # Roughly bell-shaped, centered around 18.8

hist(data$Humidity,
     main = "Humidity",
     xlab = "Humidity",
     col = "lightgreen")  # Concentrated more at higher humidity values

hist(data$Wind.Speed,
     main = "Wind Speed",
     xlab = "Wind Speed",
     col = "pink")  # Wind speed is concentrated at a few values

hist(data$Zone.1.Power.Consumption,
     main = "Zone 1 Power Consumption",
     xlab = "Zone 1 Power",
     col = "orange")  # Power is centered around 32,346 with moderate spread

par(mfrow = c(1, 1))


# 9. Boxplots

par(mfrow = c(2, 2))

boxplot(data$Temperature, main = "Boxplot: Temperature", col = "lightblue")  # A few outliers are visible
boxplot(data$Humidity, main = "Boxplot: Humidity", col = "lightgreen")  # Some low-end outliers are visible
boxplot(data$Wind.Speed, main = "Boxplot: Wind Speed", col = "pink")  # Spread is uneven because values cluster at a few points
boxplot(data$Zone.1.Power.Consumption, main = "Boxplot: Zone 1 Power", col = "orange")  # No major extreme outliers

par(mfrow = c(1, 1))


# 10. Scatterplots

plot(data$Temperature, data$Zone.1.Power.Consumption,
     main = "Temperature vs Zone 1 Power",
     xlab = "Temperature",
     ylab = "Zone 1 Power",
     pch = 16, col = rgb(0, 0, 1, 0.3))  # Clear positive relationship

plot(data$Humidity, data$Zone.1.Power.Consumption,
     main = "Humidity vs Zone 1 Power",
     xlab = "Humidity",
     ylab = "Zone 1 Power",
     pch = 16, col = rgb(0, 0.5, 0, 0.3))  # Moderate negative relationship


plot(data$Wind.Speed, data$Zone.1.Power.Consumption,
     main = "Wind Speed vs Zone 1 Power",
     xlab = "Wind Speed",
     ylab = "Zone 1 Power",
     pch = 16, col = rgb(1, 0, 0, 0.3))  # Relationship looks weak and not clearly linear


# 11. Time series plot

ggplot(data, aes(x = DateTime, y = Zone.1.Power.Consumption)) +
  geom_line(color = "blue") +
  labs(title = "Zone 1 Power Consumption Over Time",
       x = "DateTime",
       y = "Zone 1 Power Consumption") +
  theme_minimal()  # Power changes over time and appears seasonal


# 12. Correlation matrix

corr_vars <- data[, c("Temperature",
                      "Humidity",
                      "Wind.Speed",
                      "general.diffuse.flows",
                      "diffuse.flows",
                      "Zone.1.Power.Consumption",
                      "Zone.2..Power.Consumption",
                      "Zone.3..Power.Consumption")]

cor_matrix <- cor(corr_vars)
cat("\nCorrelation matrix:\n")
print(cor_matrix)  # Temperature has the strongest positive correlation with Zone 1 power among these weather variables

corrplot(cor_matrix, method = "color", type = "upper",
         tl.col = "black", tl.cex = 0.8)


# 13. Pair plot

pairs(data[, c("Temperature",
               "Humidity",
               "Wind.Speed",
               "Zone.1.Power.Consumption")],
      main = "Pair Plot")  # Gives a quick view of pairwise relationships


# 14. Simple linear regression

model <- lm(Zone.1.Power.Consumption ~ Temperature + Humidity + Wind.Speed, data = data)

cat("\nLinear regression summary:\n")
summary(model)  # All predictors are significant; R-squared is about 0.204


# ==============================
# Train-Test Split 
# ==============================

set.seed(2026)  # For reproducibility

# Create training (80%) and test (20%) indices
train_index <- sample(1:nrow(data), size = 0.8 * nrow(data))
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")


# ==============================
# Method 1: Multiple Linear Regression
# ==============================

# Model equation:
# Power = beta0 + beta1 x Temperature + beta2 x Humidity + beta3 × Wind.Speed 
# + beta4 × general.diffuse.flows + beta5 × diffuse.flows + E

# Fit model on training data
lm_multiple <- lm(Zone.1.Power.Consumption ~ Temperature + Humidity + Wind.Speed + 
                    general.diffuse.flows + diffuse.flows, 
                  data = train_data)

# Model summary
cat("Multiple Linear Regression Results\n")
summary(lm_multiple)

# Predict on test data
lm_pred <- predict(lm_multiple, newdata = test_data)

# Calculate test RMSE
lm_rmse <- sqrt(mean((test_data$Zone.1.Power.Consumption - lm_pred)^2))
cat("\nTest RMSE:", round(lm_rmse, 2), "\n")

# Interpretation:

# Temperature, Humidity, Wind Speed, and general.diffuse.flows are all 
# significant predictors (p < 0.001).  
# diffuse.flows is NOT significant (p = 0.578).

# Temperature has a positive effect (+535.70). When temperature goes up, 
# power consumption goes up.
# Humidity (-55.70), Wind Speed (-153.57), and general.diffuse.flows (-1.67) 
# all have negative effects. 
# When these increase, power consumption goes down.

# Temperature has the largest coefficient, making it the strongest predictor 
# of power consumption. The model explains about 20.6% of the variation in 
# power (R-squared = 0.2057). The average prediction error on test data is 
# 6305 power units (RMSE).



# ==============================
# Method 2: LASSO Regression
# ==============================

#install.packages("glmnet")
library(glmnet)

# Prepare data for glmnet (needs matrix format)
x_train <- as.matrix(train_data[, c("Temperature", "Humidity", "Wind.Speed", 
                                    "general.diffuse.flows", "diffuse.flows")])
y_train <- train_data$Zone.1.Power.Consumption

x_test <- as.matrix(test_data[, c("Temperature", "Humidity", "Wind.Speed", 
                                  "general.diffuse.flows", "diffuse.flows")])
y_test <- test_data$Zone.1.Power.Consumption

# Cross-validation to find best lambda (tuning parameter)
set.seed(2026)
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1)  # alpha=1 for LASSO

# Plot cross-validation error vs lambda
plot(cv_lasso, main = "LASSO Cross-Validation")

# Best lambda (tuning parameter)
best_lambda <- cv_lasso$lambda.min
cat("LASSO Regression Results\n")
cat("Best lambda (tuning parameter):", round(best_lambda, 4), "\n")

# Fit final LASSO model with best lambda
lasso_model <- glmnet(x_train, y_train, alpha = 1, lambda = best_lambda)

# Show coefficients (some will be zero - those variables are "selected out")
cat("\nLASSO Coefficients:\n")
coef_lasso <- as.matrix(coef(lasso_model))
print(coef_lasso)
coef_lasso1c<- coef(lasso_model)
coef_lasso1c
# Predict on test data
lasso_pred <- predict(cv_lasso, newx = x_test, s = "lambda.min")

# Calculate test RMSE
lasso_rmse <- sqrt(mean((y_test - lasso_pred)^2))
cat("\nTest RMSE:", round(lasso_rmse, 2), "\n")

# Interpretation:
# All five predictors have non-zero coefficients, meaning LASSO selected all 
# of them. No variables were completely eliminated (none have coefficient = 0).

# Temperature has the largest positive coefficient (+532.52), confirming it 
# is the strongest predictor.
# Humidity (-55.09), Wind Speed (-147.00), general.diffuse.flows (-1.61), 
# and diffuse.flows (-0.13) all have negative effects on power consumption.

# The coefficients are very similar to the linear regression results, just 
# slightly smaller. This is because LASSO shrinks coefficients to reduce overfitting.

# Test RMSE = 6305.17, which is almost identical to linear regression (6305.16).

# Conclusion: Temperature remains the best predictor of Zone 1 power consumption.


# ==============================
# Method 3: Decision Tree Regression
# ==============================

#install.packages("rpart")
library(rpart)
library(rpart.plot)

# Fit decision tree on training data
# Model equation: tree splits the data into regions, each region predicts average power
tree_model <- rpart(Zone.1.Power.Consumption ~ Temperature + Humidity + Wind.Speed + 
                      general.diffuse.flows + diffuse.flows,
                    data = train_data,
                    control = rpart.control(minsplit = 20, cp = 0.01))

# Find best cp (complexity parameter) using cross-validation
printcp(tree_model)
plotcp(tree_model, main = "CP vs Cross-Validation Error")

# Prune tree using best cp (optional: use 1-SE rule)
best_cp <- tree_model$cptable[which.min(tree_model$cptable[, "xerror"]), "CP"]
cat("Decision Tree Results\n")
cat("Best CP (tuning parameter):", round(best_cp, 4), "\n")

# Prune tree
pruned_tree <- prune(tree_model, cp = best_cp)

# Plot the tree (great for presentation!)
prp(pruned_tree, type = 2, extra = 1, main = "Decision Tree for Power Consumption")

# Predict on test data
tree_pred <- predict(pruned_tree, newdata = test_data)

# Calculate test RMSE
tree_rmse <- sqrt(mean((test_data$Zone.1.Power.Consumption - tree_pred)^2))
cat("Test RMSE:", round(tree_rmse, 2), "\n")

# Interpretation:
# The tree only used TWO predictors: Temperature and Humidity.
# Wind Speed, general.diffuse.flows, and diffuse.flows were NOT used.

# This means Temperature and Humidity are the most important variables 
# for predicting power consumption according to the tree.

# The tree creates simple "if-then" rules. For example:
# - If Temperature is low, go left branch
# - If Temperature is high, go right branch
# - Then check Humidity for further splits

# Each final leaf (end point) gives a predicted power consumption value.

# Test RMSE = 6325.27, which is slightly higher than linear regression (6305) 
# and LASSO (6305), meaning the tree is slightly less accurate.

# However, the tree is much easier to explain to a non-technical audience 
# because you can visually show the decision rules.

# Conclusion: Temperature is the most important predictor, followed by Humidity.


# Zone 2 Analysis
# Histograms
par(mfrow=c(2,2))
hist(data$Temperature, 
     main = "Temperature",
     xlab = "Temperature",
     col = "lightblue")

hist(data$Humidity, 
     main = "Humidity",
     xlab = "Humidity",
     col = "lightgreen")

hist(data$Wind.Speed, 
     main = "Wind Speed",
     xlab = "Wind Speed",
     col = "pink")

hist(data$Zone.2..Power.Consumption, 
     main = "Zone 2 Power Consumption",
     xlab = "Zone 2 Power Consumption",
     col = "orange")
par(mfrow = c(1,1))

# Box plots
par(mfrow = c(2,2))
boxplot(data$Temperature,
        main = "Boxplot : Temperature",
        col = "lightblue")
boxplot(data$Humidity, 
        main = "Boxplot : Humidity",
        col = "lightgreen")
boxplot(data$Wind.Speed,
        main = "Boxplot: Wind Speed",
        col = "pink")
boxplot(data$Zone.2..Power.Consumption,
        main = "Boxplot: Zone 2 power consumption",
        col = "orange")
par(mfrow = c(1,1))

# Scatter plots
plot(data$Temperature, data$Zone.2..Power.Consumption,
     main = " Temperature vs Zone 2 Power consumption",
     xlab = "Temperature",
     ylab = "Zone 2 power consumption",
     pch = 16, col = rgb(0,0,1,0.3))

plot(data$Humidity, data$Zone.2..Power.Consumption,
     main = "Humidity vs Zone 2 Power Consumption",
     xlab = "Humidity",
     ylab = "zone 2 Power Consumption",
     pch = 16, col = rgb(0,0,1,0.3))

plot(data$Wind.Speed, data$Zone.2..Power.Consumption,
     main = "Wind Speed vs Zone 2 Power Consumption",
     xlab = "Wind Speed",
     ylab = "zone 2 Power Consumption",
     pch = 16, col = rgb(0,0,1,0.3))

# Time Series
ggplot(data,aes(x = DateTime, y = Zone.2..Power.Consumption)) + 
  geom_line(color = "darkgreen")+
  labs(title = "Zone 2 Power Consumption over time",
       x = "DateTime", y = "Zone 2 Power Consumption") + 
  theme_minimal()

# Multiple linear Regression
lm_z2 <- lm(Zone.2..Power.Consumption ~ Temperature + Humidity + Wind.Speed + general.diffuse.flows + diffuse.flows,
            data = train_data)
cat("Zone 2 - Multiplt linear Regression\n")
summary(lm_z2)
lm_z2Pred <- predict(lm_z2, newdata = test_data)
lm_z2Rmse <- sqrt(mean((test_data$Zone.2..Power.Consumption - lm_z2Pred)^2))
cat("Zone 2 MLR Test RMSE:", round(lm_z2Rmse,2),"\n")

# LASSO Regression
x_train_z2 <- as.matrix(train_data[,c("Temperature","Humidity","Wind.Speed",
                                      "general.diffuse.flows","diffuse.flows")])
y_train_z2 <- train_data$Zone.2..Power.Consumption
x_test_z2 <- as.matrix(test_data[,c("Temperature","Humidity","Wind.Speed",
                                    "general.diffuse.flows","diffuse.flows")])
y_test_z2 <- test_data$Zone.2..Power.Consumption
set.seed(2026)
cv_lasso_z2 <- cv.glmnet(x_train_z2,y_train_z2,alpha = 1)
plot(cv_lasso_z2, main = "Lasso Cross Validation - Zone 2")
best_lambda_z2 <- cv_lasso_z2$lambda.min
cat("Zone 2 LASSO Best Lambda:", round(best_lambda_z2,4),"\n")
lasso_pred_z2 <- predict(cv_lasso_z2, newx = x_test_z2, s = "lambda.min")
lasso_z2_rmse <- sqrt(mean((y_test_z2 - lasso_pred_z2)^2))
cat("Zone 2 LASSO Test RMSE:", round(lasso_z2_rmse,2),"\n")


# Decision Tree
tree_z2 <- rpart(Zone.2..Power.Consumption ~ Temperature + Humidity + Wind.Speed + general.diffuse.flows + diffuse.flows,
                 data = train_data,
                 control = rpart.control(minsplit = 20, cp = 0.01))

best_cp_z2 <- tree_z2$cptable[which.min(tree_z2$cptable[, "xerror"]), "CP"]
pruned_tree_z2 <- prune(tree_z2, cp = best_cp_z2)
prp(pruned_tree_z2, type = 2, extra = 1, main = "Zone 2 Decision Tree")

tree_z2_pred <- predict(pruned_tree_z2, newdata = test_data)
tree_z2_rmse <- sqrt(mean((test_data$Zone.2..Power.Consumption - tree_z2_pred)^2))
cat("Zone 2 Decision Tree Test RMSE:", round(tree_z2_rmse, 2), "\n")

#  RMSE Summary

cat("\n    Zone 2 Model Comparison    \n")
cat("MLR RMSE: ", round(lm_z2Rmse, 2), "\n")
cat("LASSO RMSE: ", round(lasso_z2_rmse, 2), "\n")
cat("Decision Tree RMSE: ", round(tree_z2_rmse, 2), "\n")


# Zone 3 Analysis

# Histograms
par(mfrow = c(2, 2))
hist(data$Temperature,
     main = "Temperature",
     xlab = "Temperature",
     col = "lightblue")
hist(data$Humidity,
     main = "Humidity",
     xlab = "Humidity",
     col = "lightgreen")
hist(data$Wind.Speed, 
     main = "Wind Speed",
     xlab = "Wind Speed",
     col = "pink")
hist(data$Zone.3..Power.Consumption,
     main = "Zone 3 Power Consumption",
     xlab = "Zone 3 Power",
     col = "purple")
par(mfrow = c(1, 1))

# Boxplots

par(mfrow = c(2, 2))
boxplot(data$Temperature,
        main = "Temperature",
        col = "lightblue")
boxplot(data$Humidity,
        main = "Humidity",
        col = "lightgreen")
boxplot(data$Wind.Speed,
        main = "Wind Speed", 
        col = "pink")
boxplot(data$Zone.3..Power.Consumption,
        main = "Zone 3 Power",
        col = "purple")
par(mfrow = c(1, 1))

# Scatterplots

plot(data$Temperature, data$Zone.3..Power.Consumption,
     main = "Temperature vs Zone 3 Power",
     xlab = "Temperature", ylab = "Zone 3 Power",
     pch = 16, col = rgb(0, 0, 1, 0.3))

plot(data$Humidity, data$Zone.3..Power.Consumption,
     main = "Humidity vs Zone 3 Power",
     xlab = "Humidity", ylab = "Zone 3 Power",
     pch = 16, col = rgb(0, 0.5, 0, 0.3))

plot(data$Wind.Speed, data$Zone.3..Power.Consumption,
     main = "Wind Speed vs Zone 3 Power",
     xlab = "Wind Speed", ylab = "Zone 3 Power",
     pch = 16, col = rgb(1, 0, 0, 0.3))

# Time series
ggplot(data, aes(x = DateTime, y = Zone.3..Power.Consumption)) +
  geom_line(color = "purple") +
  labs(title = "Zone 3 Power Consumption Over Time",
       x = "DateTime", y = "Zone 3 Power Consumption") +
  theme_minimal()


# Zone 3 — Method 1: Multiple Linear Regression
lm_z3 <- lm(Zone.3..Power.Consumption ~ Temperature + Humidity + Wind.Speed +
              general.diffuse.flows + diffuse.flows,
            data = train_data)
cat("Zone 3 — Multiple Linear Regression\n")
summary(lm_z3)
lm_z3_pred <- predict(lm_z3, newdata = test_data)
lm_z3_rmse <- sqrt(mean((test_data$Zone.3..Power.Consumption - lm_z3_pred)^2))
cat("Zone 3 MLR Test RMSE:", round(lm_z3_rmse, 2), "\n")

# Zone 3 — Method 2: LASSO Regression

x_train_z3 <- as.matrix(train_data[, c("Temperature", "Humidity", "Wind.Speed",
                                       "general.diffuse.flows", "diffuse.flows")])
y_train_z3 <- train_data$Zone.3..Power.Consumption
x_test_z3  <- as.matrix(test_data[,  c("Temperature", "Humidity", "Wind.Speed",
                                       "general.diffuse.flows", "diffuse.flows")])
y_test_z3  <- test_data$Zone.3..Power.Consumption
set.seed(2026)
cv_lasso_z3 <- cv.glmnet(x_train_z3, y_train_z3, alpha = 1)
plot(cv_lasso_z3, main = "Zone 3 LASSO Cross-Validation")

best_lambda_z3 <- cv_lasso_z3$lambda.min
cat("Zone 3 LASSO Best Lambda:", round(best_lambda_z3, 4), "\n")

lasso_z3_pred <- predict(cv_lasso_z3, newx = x_test_z3, s = "lambda.min")
lasso_z3_rmse <- sqrt(mean((y_test_z3 - lasso_z3_pred)^2))
cat("Zone 3 LASSO Test RMSE:", round(lasso_z3_rmse, 2), "\n")

# Zone 3 — Method 3: Decision Tree

tree_z3 <- rpart(Zone.3..Power.Consumption ~ Temperature + Humidity + Wind.Speed +
                   general.diffuse.flows + diffuse.flows,
                 data = train_data,
                 control = rpart.control(minsplit = 20, cp = 0.01))

best_cp_z3 <- tree_z3$cptable[which.min(tree_z3$cptable[, "xerror"]), "CP"]
pruned_tree_z3 <- prune(tree_z3, cp = best_cp_z3)
prp(pruned_tree_z3, type = 2, extra = 1, main = "Zone 3 Decision Tree")

tree_z3_pred <- predict(pruned_tree_z3, newdata = test_data)
tree_z3_rmse <- sqrt(mean((test_data$Zone.3..Power.Consumption - tree_z3_pred)^2))
cat("Zone 3 Decision Tree Test RMSE:", round(tree_z3_rmse, 2), "\n")

# Zone 3 — RMSE Summary
cat("\n   Zone 3 Model Comparison    \n")
cat("MLR RMSE: ", round(lm_z3_rmse, 2), "\n")
cat("LASSO RMSE: ", round(lasso_z3_rmse, 2), "\n")
cat("Decision Tree RMSE: ", round(tree_z3_rmse, 2), "\n")


# Save all plots to one PDF

# Start PDF
pdf("All_Plots+final.pdf", width = 10, height = 8)

# 8. Histograms
par(mfrow = c(2, 2))
hist(data$Temperature, main = "Temperature", xlab = "Temperature", col = "lightblue")
hist(data$Humidity, main = "Humidity", xlab = "Humidity", col = "lightgreen")
hist(data$Wind.Speed, main = "Wind Speed", xlab = "Wind Speed", col = "pink")
hist(data$Zone.1.Power.Consumption, main = "Zone 1 Power", xlab = "Zone 1 Power", col = "orange")
par(mfrow = c(1, 1))

par(mfrow = c(2, 2))
hist(data$Temperature, main = "Temperature",xlab = "Temperature",col = "lightblue")
hist(data$Humidity, main = "Humidity",xlab = "Humidity",col = "lightgreen")
hist(data$Wind.Speed, main = "Wind Speed",xlab = "Wind Speed",col = "pink")
hist(data$Zone.2..Power.Consumption, main = "Zone 2 Power Consumption",xlab = "Zone 2 Power Consumption",col = "orange")
par(mfrow = c(1, 1))

par(mfrow = c(2, 2))
hist(data$Temperature,main = "Temperature",xlab = "Temperature",col = "lightblue")
hist(data$Humidity,main = "Humidity",xlab = "Humidity",col = "lightgreen")
hist(data$Wind.Speed, main = "Wind Speed",xlab = "Wind Speed",col = "pink")
hist(data$Zone.3..Power.Consumption,main = "Zone 3 Power Consumption",xlab = "Zone 3 Power",col = "purple")
par(mfrow = c(1, 1))


# 9. Boxplots
par(mfrow = c(2, 2))
boxplot(data$Temperature, main = "Boxplot: Temperature", col = "lightblue")
boxplot(data$Humidity, main = "Boxplot: Humidity", col = "lightgreen")
boxplot(data$Wind.Speed, main = "Boxplot: Wind Speed", col = "pink")
boxplot(data$Zone.1.Power.Consumption, main = "Boxplot: Zone 1 Power", col = "orange")
par(mfrow = c(1, 1))

par(mfrow = c(2, 2))
boxplot(data$Temperature,main = "Boxplot : Temperature",col = "lightblue")
boxplot(data$Humidity, main = "Boxplot : Humidity",col = "lightgreen")
boxplot(data$Wind.Speed, main = "Boxplot: Wind Speed", col = "pink")
boxplot(data$Zone.2..Power.Consumption,main = "Boxplot: Zone 2 power consumption",col = "orange")
par(mfrow = c(1, 1))

par(mfrow = c(2, 2))
boxplot(data$Temperature,main = "Boxplot: Temperature",col = "lightblue")
boxplot(data$Humidity,main = "Boxplot: Humidity",col = "lightgreen")
boxplot(data$Wind.Speed,main = "Boxplot: Wind Speed", col = "pink")
boxplot(data$Zone.3..Power.Consumption,main = "Boxplot: Zone 3 Power",col = "purple")
par(mfrow = c(1, 1))

# 10. Scatterplots
plot(data$Temperature, data$Zone.1.Power.Consumption,
     main = "Temperature vs Zone 1 Power",
     xlab = "Temperature", ylab = "Zone 1 Power",
     pch = 16, col = rgb(0, 0, 1, 0.3))

plot(data$Humidity, data$Zone.1.Power.Consumption,
     main = "Humidity vs Zone 1 Power",
     xlab = "Humidity", ylab = "Zone 1 Power",
     pch = 16, col = rgb(0, 0.5, 0, 0.3))

plot(data$Wind.Speed, data$Zone.1.Power.Consumption,
     main = "Wind Speed vs Zone 1 Power",
     xlab = "Wind Speed", ylab = "Zone 1 Power",
     pch = 16, col = rgb(1, 0, 0, 0.3))

plot(data$Temperature, data$Zone.2..Power.Consumption,
     main = " Temperature vs Zone 2 Power consumption",
     xlab = "Temperature",
     ylab = "Zone 2 power consumption",
     pch = 16, col = rgb(0,0,1,0.3))

plot(data$Humidity, data$Zone.2..Power.Consumption,
     main = "Humidity vs Zone 2 Power Consumption",
     xlab = "Humidity",
     ylab = "zone 2 Power Consumption",
     pch = 16, col = rgb(0,0,1,0.3))

plot(data$Wind.Speed, data$Zone.2..Power.Consumption,
     main = "Wind Speed vs Zone 2 Power Consumption",
     xlab = "Wind Speed",
     ylab = "zone 2 Power Consumption",
     pch = 16, col = rgb(0,0,1,0.3))

plot(data$Temperature, data$Zone.3..Power.Consumption,
     main = "Temperature vs Zone 3 Power",
     xlab = "Temperature", ylab = "Zone 3 Power",
     pch = 16, col = rgb(0, 0, 1, 0.3))

plot(data$Humidity, data$Zone.3..Power.Consumption,
     main = "Humidity vs Zone 3 Power",
     xlab = "Humidity", ylab = "Zone 3 Power",
     pch = 16, col = rgb(0, 0.5, 0, 0.3))

plot(data$Wind.Speed, data$Zone.3..Power.Consumption,
     main = "Wind Speed vs Zone 3 Power",
     xlab = "Wind Speed", ylab = "Zone 3 Power",
     pch = 16, col = rgb(1, 0, 0, 0.3))


# 11. Time series plot (ggplot)
print(ggplot(data, aes(x = DateTime, y = Zone.1.Power.Consumption)) +
        geom_line(color = "blue") +
        labs(title = "Zone 1 Power Consumption Over Time",
             x = "DateTime", y = "Zone 1 Power Consumption") +
        theme_minimal())

print(ggplot(data,aes(x = DateTime, y = Zone.2..Power.Consumption)) + 
  geom_line(color = "darkgreen")+
  labs(title = "Zone 2 Power Consumption over time",
       x = "DateTime", y = "Zone 2 Power Consumption") + 
  theme_minimal())

print(ggplot(data, aes(x = DateTime, y = Zone.3..Power.Consumption)) +
  geom_line(color = "purple") +
  labs(title = "Zone 3 Power Consumption Over Time",
       x = "DateTime", y = "Zone 3 Power Consumption") +
  theme_minimal())

# 12. Correlation plot
corrplot(cor_matrix, method = "color", type = "upper", tl.col = "black", tl.cex = 0.8)

# 13. Pair plot
pairs(data[, c("Temperature", "Humidity", "Wind.Speed", "Zone.1.Power.Consumption")],
      main = "Pair Plot")
plot(cv_lasso, main = "LASSO Cross-Validation")
plotcp(tree_model, main = "CP vs Cross-Validation Error")
prp(pruned_tree, type = 2, extra = 1, main = "Decision Tree for Power Consumption")

# Close PDF
dev.off()

cat("All plots saved to All_Plots_final.pdf\n")


