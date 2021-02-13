<p align="center">
  <img width="200" height="280" src="images/portfolio.png">
</p>

<h1 style="text-align:center">John Owusu Duah</h1>



## About
I am an engineer who values shifting from the prevailing anecdotal, political and social way of making decisions within organizations to a more precise, evidence-based, data-driven process that can be replicated and scaled in a cost effective manner.

I am currently working as a transportation engineer at the Department of Urban Roads, an agency under Ghana's Ministry of Roads and Highways. Among other responsibilities, I lead the preparation of reports on optimum project selection, validation, appraisal, and supply chain risk analysis after extracting and analyzing data from Intelligent Road Network Management Database (IRNMD) system  using SQL, Python and ArcGIS. 

## Passion Projects
### 1. Classification Predictive Modeling (A test case to predict the price of Bitcoin in 2026) - March 2021
#### Executive Summary (In Progress)
With Tesla, Twitter, and other technology companies buying large amounts of Bitcoin prompting an increase in demand for the cryptocurrency and a sharp surge in its price (1 Bitcoin = $45,191.00 as of February 10, 2021), the question on the minds of the those looking on in askance is, is Bitcoin real or is it a bubble? To help answer this question, I will seek to predict the price of Bitcoin in five years so that we can at least determine if indeed Bitcoin is a bubble or an genuine asset class for wealth creation.
![](/images/bitcoin.jpg)



### [2. Clustering of Geospatial Data Based on Number of Businesses Near Neighborhoods in Toronto and Vizualization on OpenStreetMap (A Test Case to Cluster Neighborhoods in Toronto Based On Venue Categories) - January 2021](/clustering_neighborhoods_toronto.html)
#### Executive Summary
Cluster Neighborhoods in Toronto Based On Venue Categories) - January 2021](/clustering_neighborhoods_toronto.html)
#### Executive Summary
This project was carried out as real-world application of Python's clustering algorithm to segregate neighorhoods in Toronto, Canada into clusters with similar high numbers of venue categories. Using insight from this project, marketing campaigns can be targeted and made efficient by companies who wish to market products and services to targeted businesses in Toronto.

Why Toronto, Canada?
Initially, I set out to carry out this project in my home country of Ghana, which does not have enough data available on the internet to draw meaninful conclusions from on one end of the spectrum. On the other end of the spectrum, developed nations like the U.S.A, England, France, and Germany had superfluous data available in different formats on the internet. I needed to find a happy medium.

A large part of a data scientist's job is gathering, selecting, and transforming data to answer analytical questions. Hence, as a test case, Toronto has just enough data available on the web to enable me to apply Python's Beautiful Soup and pgeocode libraries, which I will need in the future. Additionally, Foursquare has curated data on businesses in Toronto, so I was able to practice making calls to Foursquare's API and parsing the resulting json file to extract needed data. Toronto allowed me to get real feedback on skills and concepts I learned and pushed me to learn more and better ways to collect and clean data.

After merging georeferenced data of neighborhoods in Toronto with data on venue categories obtained from making calls to Foursquare's API, the data was transformed and fed into the k-means clustering algorithm to segregate the neighborhoods into the following five (5) clusters:
- Cluster 0: Neighborhoods with high number of dining venues. Color on map = Red
- Cluster 1: Neighborhoods with high number of photography studios. Color on map = Purple
- Cluster 2: Neighborhoods with high number of home service venues. Color on map = Cyan
- Cluster 3: Neighborhoods with high number of residential buildings close to parks. Color on map = Light Green
- Cluster 4: Neighborhoods with high number of outdoor playground and recreational venues. Color on map = Light Brown

The clusters were superimposed on the OpenStreetMap of Toronto using Python's Folium Library for easy referencing.
![](/images/clustering.png)



### [3. Comprehensive Regression Modeling Using Python (A Test Case to Predict House Prices in King County, Seattle, U.S.A) - October 2020](/predictive_regression_kingcounty_seattle.html)
#### Executive Summary
How does one use regression modeling with Python to build predictive models when given data with numerous possible predictor variables? The project sought to answer this question while serving as a comprehensive guide for students and practitioners to understand the concepts and code behind regression modeling with Python. The dataset is available online at www.kaggle.com and has details of houses sold in King County between May 2014 and May 2015.

After the data was imported into a dataframe, it was cleaned and explored to better understand the story behind it. At this stage, the correlation between all possible independent variables and the dependent variable of house price was evaluated. Checks were made to determine which of the following models would produce the most accurate model:
 -  Linear Regression Model
 -  Polynomial Regression Model with a single independent variable
 -  Polynomial Regression Model with multiple independent variables
 -  Ridge Regression Model with Grid Search
 
 In-sample and out-of-sample evaluation were carried to break down the concepts and make them easy to understand for the reader. Issues of **multicollinearity**, which most students and practitioners often ignore, was discussed exhaustively and dealt with.
 
In the end, our analysis revealed that the choice of the most optimum model would be dependent on its use. If the overall objective of carrying out the regression modeling is to make predictions while determining coefficients and p values, then the ridge regression in the Section 6 of the notebook would be the model to use. However, if the sole purpose of carrying out the analysis is to make predictions, then the second-order multivariate polynomial regression model would be ideal.
![](/images/regression3.0.png)
