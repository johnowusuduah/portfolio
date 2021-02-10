<p align="center">
  <img width="180" height="180" src="images/profile_pic.png">
</p>

<h1 style="text-align:center">John Owusu Duah</h1>


# Data Science Portfolio
## About
I am an engineer who values making a shift from the prevailing anecdotal, political and social way of making decisions within organizations to a more precise, evidence-based data-driven process which can be replicated and scaled in a cost effective manner.

I am currently working as a transportation engineer at the Department of Urban Roads, an agency under the Ghana's Ministry of Roads and Highways. Among other responsibilities, I manage the road maintenance database system for the Western Region of Ghana to generate insights for project selection, validation and appraisal using SQL, Python(scikit-learn, pandas, numpy, matplotlib, SciPy and Folium) and AutoCAD Civil 3D. 

## Projects

### Clustering of Geospatial Data Based on Number of Businesses in Close Proximity and Vizualization on OpenStreetMap (A Test Case to Cluster Neighborhoods in Toronto Based On Venue Categories) - January 2021
#### Executive Summary
This project was carried out as real-world application of Python's clustering algorithm to segregate neighorhoods in Toronto, Canada into clusters with similar high numbers of venue categories. By using insight from this project, marketing campaigns can be targeted and made efficient by companies who wish to market products and services to targeted businesses in Toronto.

Why Toronto, Canada?
Initially, I set out to carry out this project in my home country of Ghana, which, on one end of the spectrum, does not have enough data available on the internet to draw meaninful conclusions from. On the other end of the spectrum, developed nations like the U.S.A, England, France, and Germany had superfluous data available in different formats on the internet. I needed to find a happy medium.

A large part of data scientist's job is gathering, selecting and transforming data to answer and analytical question. Hence Toronto, as a test case, has just enough data available on the web to enable me apply Python's Beautiful Soup and pgeocode libraries, which I will need in the future. Additionally, Foursquare has curated data on businesses in Toronto, so I was able to practice making calls to Foursquare's API (Application Programming Interface) and parsing the resulting json file to extract needed data. Toronto, afforded me the opportunity to get real feedback on skills and concept I learnt and pushed me to learn more and better ways to collect and clean data.

After merging georeferenced data of neighborhoods in Toronto with data on venue categories obtained from making calls to Foursquare's API, the data was transformed and fed into the k-means clustering algorithm to segregate the neighborhoods into the following five (5) clusters:
- Cluster 0: Neighborhoods with high number of dining venues. Color on map = Red
- Cluster 1: Neighborhoods with high number of photography studios. Color on map = Purple
- Cluster 2: Neighborhoods with high number of home service venues. Color on map = Cyan
- Cluster 3: Neighborhoods with high number of residential buildings close to parks. Color on map = Light Green
- Cluster 4: Neighborhoods with high number of outdoor playground and recreational venues. Color on map = Light Brown

The clusters were superimposed on the OpenStreetMap of Toronto using Python's Folium Library for easy referencing.
![](/images/clustering.png)


### Comprehensive Regression Modeling Using Python (A Test Case to Predict House Prices in King County, Seattle, U.S.A) - October 2020
![](/images/regression3.0.png)
#### Executive Summary
How does one use regression modeling with Python to build predictive models when given data with numerous possible predictor variables? The project sought to answer this question while serving as a comprehensive guide for students and practitioners to gain a deeper understanding of the concepts and code behind regression modeling with Python. The dataset is available online at www.kaggle.com and has details of houses sold in King County between May 2014 and May 2015.

After the data was imported into a dataframe, it was cleaned and explored to gain a better understanding of the story behind it. At this stage, the correlation between all possible independent variables and the dependent variable of house price was evaluated. Checks were made to determine which of the following models would produce the most accurate model:
 -  Linear Regression Model
 -  Polynomial Regression Model with a single independent variable
 -  Polynomial Regression Model with multiple independent variables
 -  Ridge Regression Model with Grid Search
 
 In-sample and out-of-sample evaluation was carried to break down the concepts and make them easy to understand for the reader. Issues pertaining to **multicollinearity**, which is often ignored by most students and practitioners, was discussed exhaustively and dealt with.
 
In the end, our analysis revealed that the the choice of the most optimum model would be dependent on its use. If the overall objective of carrying out the regression modeling is to make predictions while determining coefficients and p values then the ridge regression in the Section 6 of the notebook would be the model to use. However, if the sole purpose of carrying out the analysis is to make predictions, then the second order multivariate polynomial regression model would be ideal.

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/johnowusuduah/ds_portfolio/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.
