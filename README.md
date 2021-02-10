![](/images/profile_pic.png)

## John Owusu Duah

### About
I am an engineer who values making a shift from the prevailing anecdotal, political and social way of making decisions within organizations to a more precise, evidence-based data-driven process which can be replicated and scaled in a cost effective manner.

I am currently working as a transportation engineer at the Department of Urban Roads, an agency under the Ghana's Ministry of Roads and Highways. Among other responsibilities, I manage the road maintenance database system for the Western Region of Ghana to generate insights for project selection, validation and appraisal using SQL, Python(scikit-learn, pandas, numpy, matplotlib, SciPy and Folium) and AutoCAD Civil 3D. 

## Projects

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/johnowusuduah/ds_portfolio/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Comprehensive Regression Modeling Using Python (A Test Case to Predict House Prices in King County, Seattle, U.S.A)
![](/images/regression3.0.png)
#### Executive Summary
How does one use regression modeling with Python to build predictive models when given data with numerous possible predictor variables? The project sought to answer this question while serving as a comprehensive guide for students and practitioners to gain a deeper understanding of the concepts and code behind regression modeling with Python. The dataset is available online at www.kaggle.com and has details of houses sold in King County between May 2014 and May 2015.

After the data was imported into a dataframe, it was cleaned and explored to gain a better understanding of the story behind it. At this stage, the correlation between all possible independent variables and the dependent variable of house price was evaluated. Checks were made to determine which of the following models would produce the most accurate model:
 - [1. Linear Regression Model]
 - [2. Polynomial Regression Model with a single independent variable]
 - [3. Polynomial Regression Model with multiple independent variables]
 - [4. Ridge Regression Model with Grid Search]
 
 In-sample and out-of-sample evaluation was carried to break down the concepts and make them easy to understand for the reader. Issues pertaining to **multicollinearity**, which is often ignored by most students and practitioners, was discussed exhaustively and dealt with.
 
 In the end, our analysis revealed that the the choice of the most optimum model would be dependent on its use. If the overall objective of carrying out the regression modeling is to make predictions while determining coefficients and p values then the ridge regression in the Section 6 of the notebook would be the model to use. However, if the sole purpose of carrying out the analysis is to make predictions, then the second order multivariate polynomial regression model would be ideal.
