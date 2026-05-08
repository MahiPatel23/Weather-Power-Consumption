# Weather-Driven Power Consumption Prediction

## Overview
Predicted electricity consumption in Zone 2 of Tetouan City (Morocco) using weather and time based 
features.Analyzed 52,000+ time series energy records and compared three machine learning models.

## Research Questions
1) Can weather variables predict Zone 2 power output?
2) Which model provides the best prediction performance?
3) Do nonlinear models outperform simpler linear ones?

## Dataset 
- **Source:** Tetouan City Power Consumption Dataset
- **Size:** 52,000+ records(recorded every 10 minutes, full year 2017)
- **Features:** Temperatrue, Humnidity, Wind Speed, General Diffuse Flows, Diffuse Flows

## Models & Results
| Model | Test RMSE (MW) | R-squared |
|-------|----------------|-----------|
| Multiple Linear Regression | 6,305.16 | 0.206 |
| LASSO Regression | 6,305.17 | 0.206 |
| Decision Tree (Pruned) | 6,325.27 | ~0.198 |

**Best Model:** LASSO Regression - matches MLR accuracy while handling multicollinearity between 
difference flow variables.

## Key Findings 
- Temperature was the strongest predictor with U-shaped relationship (high demand in both winter and summer)
- Humidity showed a moderate negative correlation with power usage
- Wind Speed was the weakest predictor
- LASSO retained all 5 predictors with non-zero coefficients.

## Tools & Technologies
-**Language:** R
-**Libraries:** glmnet, rpart, ggplot2, tidyverse

## Course
STAT 385 - UIC, Spring 2026

## Team 
- Mahi Patel - Presentation, final report, Zone 2 & 3 analysis
- Rahee Chauhan - EDA, data cleaning
- Mustafa Khan - Model building
- Usam Tahir - Model evaluation & comparison
  
