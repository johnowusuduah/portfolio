
# Clustering of Neighorhoods in Toronto Based on Venue Categories & Visualization on Open Street Map

### Aim of Project

This project was carried out in order to segregate neighorhoods in Toronto, Canada into clusters with different numbers of venue categories. After running the k-means clustering algorithm for five clusters, the following clusters were identified:

1.
2.
3.
4.
5.

By using insight from this project, marketing campaigns can be streamlined and made efficient by companies who wish to market products and services to targeted businesses in Toronto. Also, real estate agencies can gain insight into the unique selling points of different neighborhoods with regard to the proximity to venues of different categories. This could influence pricing of properties and ultimately boost profits. By using the results of this analysis, travel companies like AirBnB can provide customers with high level of information that can minimize churn rate.


### Motivation

Why Toronto, Canada?
Initially, I set out to carry out this project in my home country of Ghana, which, on one end of the spectrum, does not have enough data available on the internet to draw meaninful conclusions from. On the other end of the spectrum, developed nations like the U.S.A, England, France, and Germany had superfluous data available in different formats on the internet. I needed to find a happy medium.

I have come to learn that a large part of data scientist's job is data wrangling. Hence Toronto, as a test case, has just enough data available on the web to enable me apply Python's Beautiful Soup and pgeocode libraries, which I will need in the future. Additionally, Foursquare has curated data on businesses in Toronto, so I was able to practice making calls to Foursquare's API and parsing the resulting json file to extract needed data. Toronto, afforded me the opportunity to get real feedback on skills and concept I learnt in courses and pushed me to learn more and better ways to collect and clean data.

### Source of Data

The postal codes, neighborhood and borough data used for this project was web scraped, cleaned, imported into pandas and cleaned again. However, coordinates for the neighborhoods was not readily available on any webpage so I had to use pgeocode (pyhon's library for postal geocoding and distance calculations) to generate the coordinates. Data on venues within a 500m radius of neighborhoods in Toronto was collected by making calls to Foursquare's API.

### Project Details (The code for this project can be found in jupyter notebooks in my Gitbub and Kaggle Repositories)

### Import Libraries


```python
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors

from bs4 import BeautifulSoup
import requests 

from sklearn.cluster import KMeans

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values
import geocoder # import geocoder

import folium 


print('Libraries imported.')
```

    Libraries imported.


### Scrap Data on Canada's Postal Codes, Boroughs and Neighborhoods from the Internet

We utilize Python's Beautiful Soup Library.


```python
# Extract data on Canada's postal codes, boroughs and neighborhoods
wikipedia = requests.get("https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M")
wiki_page = wikipedia.content
canadian_soup = BeautifulSoup(wiki_page, "html.parser")
```

### Import Scrapped Data into a Dataframe and Display Raw Dataframe


```python
# Create an empty dataframe will predetermined column labels
column_names = ['Postalcode','Borough','Neighborhood']
df_raw = pd.DataFrame(columns = column_names)

# Parse through HTML for postal data in Table
content = canadian_soup.find('div', class_='mw-parser-output')
table = content.table.tbody
postcode = 0
borough = 0
neighborhood = 0

for tr in table.find_all('tr'):
    i = 0
    for td in tr.find_all('td'):
        if i == 0:
            postcode = td.text
            i = i + 1
        elif i == 1:
            borough = td.text
            i = i + 1
        elif i == 2: 
            neighborhood = td.text.strip('\n').replace(']','')
    df_raw = df_raw.append({'Postalcode': postcode,'Borough': borough,'Neighborhood': neighborhood},ignore_index=True)

df_raw.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postalcode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1A\n</td>
      <td>Not assigned\n</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M2A\n</td>
      <td>Not assigned\n</td>
      <td>Not assigned</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M3A\n</td>
      <td>North York\n</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M4A\n</td>
      <td>North York\n</td>
      <td>Victoria Village</td>
    </tr>
  </tbody>
</table>
</div>



### Clean Dataframe and Check the Shape of the Dataframe


```python
# Remove '\n' from all entries
df_raw = df_raw.replace('\n',' ', regex=True)

# Remove rows with 0 values in all columns
df_raw = df_raw[df_raw['Borough'] != 0]

# Remove rows with unassigned Borough names
df = df_raw[~df_raw['Borough'].str.contains("Not assigned")]

# Drops current unordered index and replaces it with on of increasing integers
df.reset_index(drop = True, inplace = True)

# Check for null/nan values
df.isnull().sum()
```




    Postalcode      0
    Borough         0
    Neighborhood    0
    dtype: int64




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postalcode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor, Lawrence Heights</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (103, 3)



### Import Python's Library for Postal Code Geolocating

With only the postal codes for Canada, pgeocode generates the latitude and longitude of every neighborhood in Canada.


```python
import pgeocode

nomi = pgeocode.Nominatim('ca')
postal_code = df["Postalcode"].values.tolist()
location = nomi.query_postal_code(postal_code)
```

Add the latitude and longitude columns to the original dataframe to make it georeferenced. Now that this is done, we are ready to filter our data for only neighborhoods in Toronto and visualize our data on a map.


```python
df["Latitude"] = location.latitude
df["Longitude"] = location.longitude
df.head(10)
```

    /anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """Entry point for launching an IPython kernel.
    /anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postalcode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
      <td>43.7545</td>
      <td>-79.3300</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
      <td>43.7276</td>
      <td>-79.3148</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
      <td>43.6555</td>
      <td>-79.3626</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor, Lawrence Heights</td>
      <td>43.7223</td>
      <td>-79.4504</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
      <td>43.6641</td>
      <td>-79.3889</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M9A</td>
      <td>Etobicoke</td>
      <td>Islington Avenue, Humber Valley Village</td>
      <td>43.6662</td>
      <td>-79.5282</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern, Rouge</td>
      <td>43.8113</td>
      <td>-79.1930</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M3B</td>
      <td>North York</td>
      <td>Don Mills</td>
      <td>43.7450</td>
      <td>-79.3590</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M4B</td>
      <td>East York</td>
      <td>Parkview Hill, Woodbine Gardens</td>
      <td>43.7063</td>
      <td>-79.3094</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M5B</td>
      <td>Downtown Toronto</td>
      <td>Garden District, Ryerson</td>
      <td>43.6572</td>
      <td>-79.3783</td>
    </tr>
  </tbody>
</table>
</div>



### Filter Dataframe for Only Neighborhoods in Toronto


```python
df_tor = df[df['Borough'].str.contains("Toronto")]
df_tor.reset_index(drop = True, inplace = True)
df_tor.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postalcode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
      <td>43.6555</td>
      <td>-79.3626</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
      <td>43.6641</td>
      <td>-79.3889</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5B</td>
      <td>Downtown Toronto</td>
      <td>Garden District, Ryerson</td>
      <td>43.6572</td>
      <td>-79.3783</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M5C</td>
      <td>Downtown Toronto</td>
      <td>St. James Town</td>
      <td>43.6513</td>
      <td>-79.3756</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.6784</td>
      <td>-79.2941</td>
    </tr>
  </tbody>
</table>
</div>



### A Visualization of Unclustered Neighborhoods in Toronto on an Open Street May Utilizing Python's Folium Library


```python
# create map of Manhattan using latitude and longitude values
map_toronto = folium.Map(location=[43.653226, -79.383184], zoom_start=11)

# add markers to map
for lat, lng, label in zip(df_tor['Latitude'], df_tor['Longitude'], df_tor['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='green',
        fill=True,
        fill_color='yellow',
        fill_opacity=0.7,
        parse_html=False).add_to(map_toronto)  
    
map_toronto
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=%3C%21DOCTYPE%20html%3E%0A%3Chead%3E%20%20%20%20%0A%20%20%20%20%3Cmeta%20http-equiv%3D%22content-type%22%20content%3D%22text/html%3B%20charset%3DUTF-8%22%20/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%3Cscript%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20L_NO_TOUCH%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20L_DISABLE_3D%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%3C/script%3E%0A%20%20%20%20%0A%20%20%20%20%3Cstyle%3Ehtml%2C%20body%20%7Bwidth%3A%20100%25%3Bheight%3A%20100%25%3Bmargin%3A%200%3Bpadding%3A%200%3B%7D%3C/style%3E%0A%20%20%20%20%3Cstyle%3E%23map%20%7Bposition%3Aabsolute%3Btop%3A0%3Bbottom%3A0%3Bright%3A0%3Bleft%3A0%3B%7D%3C/style%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//code.jquery.com/jquery-1.12.4.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css%22/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cmeta%20name%3D%22viewport%22%20content%3D%22width%3Ddevice-width%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20initial-scale%3D1.0%2C%20maximum-scale%3D1.0%2C%20user-scalable%3Dno%22%20/%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cstyle%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%23map_aff8975b381d4e56902015cc48c55ffa%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20position%3A%20relative%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20width%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20height%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20left%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20top%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%3C/style%3E%0A%20%20%20%20%20%20%20%20%0A%3C/head%3E%0A%3Cbody%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cdiv%20class%3D%22folium-map%22%20id%3D%22map_aff8975b381d4e56902015cc48c55ffa%22%20%3E%3C/div%3E%0A%20%20%20%20%20%20%20%20%0A%3C/body%3E%0A%3Cscript%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20map_aff8975b381d4e56902015cc48c55ffa%20%3D%20L.map%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22map_aff8975b381d4e56902015cc48c55ffa%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20center%3A%20%5B43.653226%2C%20-79.383184%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20crs%3A%20L.CRS.EPSG3857%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoom%3A%2011%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoomControl%3A%20true%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20preferCanvas%3A%20false%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20tile_layer_f7a2f645e9474af39dc5aa5fde71c3f6%20%3D%20L.tileLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22https%3A//%7Bs%7D.tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22attribution%22%3A%20%22Data%20by%20%5Cu0026copy%3B%20%5Cu003ca%20href%3D%5C%22http%3A//openstreetmap.org%5C%22%5Cu003eOpenStreetMap%5Cu003c/a%5Cu003e%2C%20under%20%5Cu003ca%20href%3D%5C%22http%3A//www.openstreetmap.org/copyright%5C%22%5Cu003eODbL%5Cu003c/a%5Cu003e.%22%2C%20%22detectRetina%22%3A%20false%2C%20%22maxNativeZoom%22%3A%2018%2C%20%22maxZoom%22%3A%2018%2C%20%22minZoom%22%3A%200%2C%20%22noWrap%22%3A%20false%2C%20%22opacity%22%3A%201%2C%20%22subdomains%22%3A%20%22abc%22%2C%20%22tms%22%3A%20false%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4c97a23506e440f0ae1b66a57af82fc1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6555%2C%20-79.3626%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_eb107362f97442e4b1f265f04d4d81b0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a665803c2a6141159bc7b61342344f46%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_a665803c2a6141159bc7b61342344f46%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERegent%20Park%2C%20Harbourfront%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_eb107362f97442e4b1f265f04d4d81b0.setContent%28html_a665803c2a6141159bc7b61342344f46%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4c97a23506e440f0ae1b66a57af82fc1.bindPopup%28popup_eb107362f97442e4b1f265f04d4d81b0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9d699d4a94e4458ba98dd7af8c69cce9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6641%2C%20-79.3889%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a43ddf14dc774bb0a31932d5ba4be608%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_04f1a9c95a87484a9db684b7017c5a4b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_04f1a9c95a87484a9db684b7017c5a4b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EQueen%26%2339%3Bs%20Park%2C%20Ontario%20Provincial%20Government%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a43ddf14dc774bb0a31932d5ba4be608.setContent%28html_04f1a9c95a87484a9db684b7017c5a4b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9d699d4a94e4458ba98dd7af8c69cce9.bindPopup%28popup_a43ddf14dc774bb0a31932d5ba4be608%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c77f97eed2d84d0ab993489b42bbea45%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6572%2C%20-79.3783%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b79d9685a81f4eb3b3ba33b9095f69db%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_5d13d059c69c4768848020eeb385c70e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_5d13d059c69c4768848020eeb385c70e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGarden%20District%2C%20Ryerson%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b79d9685a81f4eb3b3ba33b9095f69db.setContent%28html_5d13d059c69c4768848020eeb385c70e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c77f97eed2d84d0ab993489b42bbea45.bindPopup%28popup_b79d9685a81f4eb3b3ba33b9095f69db%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c037e3597ed34c7eb492e205562e2e67%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6513%2C%20-79.3756%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a24a4ea2eaa54c72b13ba516ab080afc%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b773881ebeb04bd98362dbf84dacc097%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b773881ebeb04bd98362dbf84dacc097%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESt.%20James%20Town%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a24a4ea2eaa54c72b13ba516ab080afc.setContent%28html_b773881ebeb04bd98362dbf84dacc097%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c037e3597ed34c7eb492e205562e2e67.bindPopup%28popup_a24a4ea2eaa54c72b13ba516ab080afc%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f4e7e25246d545ba9c7e1db2938049cc%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6784%2C%20-79.2941%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f82bc0e7d13d4bc2b3867a2a99290c3b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_aa8ed7f2de384002b6eb0347710dc71a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_aa8ed7f2de384002b6eb0347710dc71a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThe%20Beaches%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f82bc0e7d13d4bc2b3867a2a99290c3b.setContent%28html_aa8ed7f2de384002b6eb0347710dc71a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f4e7e25246d545ba9c7e1db2938049cc.bindPopup%28popup_f82bc0e7d13d4bc2b3867a2a99290c3b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f1af39a359ab448fba10b6ea095ee09b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6456%2C%20-79.3754%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_196ae1c3236740ab9420f7aea9726044%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_47b139e4d40649fb9ff95a979863d7aa%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_47b139e4d40649fb9ff95a979863d7aa%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBerczy%20Park%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_196ae1c3236740ab9420f7aea9726044.setContent%28html_47b139e4d40649fb9ff95a979863d7aa%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f1af39a359ab448fba10b6ea095ee09b.bindPopup%28popup_196ae1c3236740ab9420f7aea9726044%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5966aed2a2d740e3bf8346a9658f58db%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6564%2C%20-79.38600000000002%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5dc1af8455654c91b94774cc9966ced8%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_dbbc51e75f2142158ef27e446ddd924e%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_dbbc51e75f2142158ef27e446ddd924e%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECentral%20Bay%20Street%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5dc1af8455654c91b94774cc9966ced8.setContent%28html_dbbc51e75f2142158ef27e446ddd924e%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_5966aed2a2d740e3bf8346a9658f58db.bindPopup%28popup_5dc1af8455654c91b94774cc9966ced8%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e076d80d77e94d47bd8339238cbdc1e1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6683%2C%20-79.4205%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_78937af19cc745f0ad4ad8129a2702f7%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3b2e4eb9e3ce4f80b0645aec6082ba16%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_3b2e4eb9e3ce4f80b0645aec6082ba16%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EChristie%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_78937af19cc745f0ad4ad8129a2702f7.setContent%28html_3b2e4eb9e3ce4f80b0645aec6082ba16%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e076d80d77e94d47bd8339238cbdc1e1.bindPopup%28popup_78937af19cc745f0ad4ad8129a2702f7%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5ea13d7da94349bba248ff4542d5cdec%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6496%2C%20-79.3833%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b17929ff64a4439985aef528fe13f1a1%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_285062b3a2f442df969f594e12bd83d6%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_285062b3a2f442df969f594e12bd83d6%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERichmond%2C%20Adelaide%2C%20King%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b17929ff64a4439985aef528fe13f1a1.setContent%28html_285062b3a2f442df969f594e12bd83d6%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_5ea13d7da94349bba248ff4542d5cdec.bindPopup%28popup_b17929ff64a4439985aef528fe13f1a1%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_8f68dd0189b34160a62ec03fab75fbaf%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6655%2C%20-79.4378%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_960e4d6d8d644ca3940e85b473d9b5c3%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_14f7ebda01f94c7396beeb7421102d6a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_14f7ebda01f94c7396beeb7421102d6a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDufferin%2C%20Dovercourt%20Village%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_960e4d6d8d644ca3940e85b473d9b5c3.setContent%28html_14f7ebda01f94c7396beeb7421102d6a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_8f68dd0189b34160a62ec03fab75fbaf.bindPopup%28popup_960e4d6d8d644ca3940e85b473d9b5c3%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6dccd713531d436288975a9ecd55616a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.62300000000001%2C%20-79.3936%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_befce897a70e4ee88e19e368aae0a85b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c3b5a0c47e5643f19d67397272daa157%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c3b5a0c47e5643f19d67397272daa157%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHarbourfront%20East%2C%20Union%20Station%2C%20Toronto%20Islands%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_befce897a70e4ee88e19e368aae0a85b.setContent%28html_c3b5a0c47e5643f19d67397272daa157%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_6dccd713531d436288975a9ecd55616a.bindPopup%28popup_befce897a70e4ee88e19e368aae0a85b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fde8400a9ba6403baa27f5ed3e209c9a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.648%2C%20-79.4177%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ef6e5c7a155f40afaeb547292e9e1345%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_092db6a9178e495fb230e03d5a07a5fa%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_092db6a9178e495fb230e03d5a07a5fa%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELittle%20Portugal%2C%20Trinity%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ef6e5c7a155f40afaeb547292e9e1345.setContent%28html_092db6a9178e495fb230e03d5a07a5fa%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_fde8400a9ba6403baa27f5ed3e209c9a.bindPopup%28popup_ef6e5c7a155f40afaeb547292e9e1345%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4068c3352a0941709a4bbc87b4d3c749%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6803%2C%20-79.3538%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_2d51b3ef81d5407686a54299f28247a6%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_613c777ca8d242d1b1e2e6a8d4037cfc%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_613c777ca8d242d1b1e2e6a8d4037cfc%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThe%20Danforth%20West%2C%20Riverdale%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_2d51b3ef81d5407686a54299f28247a6.setContent%28html_613c777ca8d242d1b1e2e6a8d4037cfc%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4068c3352a0941709a4bbc87b4d3c749.bindPopup%28popup_2d51b3ef81d5407686a54299f28247a6%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_be6329ac36bb405c8c0bca68c14bab14%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6469%2C%20-79.3823%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7429809f84d24cf68b834444ef649e6f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_18a413558bf64a829b134b7339658cc3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_18a413558bf64a829b134b7339658cc3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EToronto%20Dominion%20Centre%2C%20Design%20Exchange%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7429809f84d24cf68b834444ef649e6f.setContent%28html_18a413558bf64a829b134b7339658cc3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_be6329ac36bb405c8c0bca68c14bab14.bindPopup%28popup_7429809f84d24cf68b834444ef649e6f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_bcc1b58f89c84b7b9fc6c80a312c01d4%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6383%2C%20-79.4301%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_63912ae417f74af78e8f20f5cc5f2a4b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_067d2ad700564e4f9ad99c0e2663ccc3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_067d2ad700564e4f9ad99c0e2663ccc3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBrockton%2C%20Parkdale%20Village%2C%20Exhibition%20Place%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_63912ae417f74af78e8f20f5cc5f2a4b.setContent%28html_067d2ad700564e4f9ad99c0e2663ccc3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_bcc1b58f89c84b7b9fc6c80a312c01d4.bindPopup%28popup_63912ae417f74af78e8f20f5cc5f2a4b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_300b13a93e1f4582b74feddbbb80d6e1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6693%2C%20-79.3155%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_51edd448707343ac8b0d8be764858ddb%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d4948d2fd8ec45469a1ac2745729a1fd%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d4948d2fd8ec45469a1ac2745729a1fd%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EIndia%20Bazaar%2C%20The%20Beaches%20West%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_51edd448707343ac8b0d8be764858ddb.setContent%28html_d4948d2fd8ec45469a1ac2745729a1fd%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_300b13a93e1f4582b74feddbbb80d6e1.bindPopup%28popup_51edd448707343ac8b0d8be764858ddb%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_097d6a41fa9e48a49f5f7dea3eaa9cee%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6492%2C%20-79.3823%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_94e9612a00f545998d64153268a669bd%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7e68e36b03684fa993014631d700b931%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7e68e36b03684fa993014631d700b931%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECommerce%20Court%2C%20Victoria%20Hotel%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_94e9612a00f545998d64153268a669bd.setContent%28html_7e68e36b03684fa993014631d700b931%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_097d6a41fa9e48a49f5f7dea3eaa9cee.bindPopup%28popup_94e9612a00f545998d64153268a669bd%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_cc745147c2554ab1b698bbdccdc9bf56%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6561%2C%20-79.3406%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d9c5edbece7741dfa2154ba12c13631d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_34bee8840bc34f81b84897be1805cf8b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_34bee8840bc34f81b84897be1805cf8b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EStudio%20District%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d9c5edbece7741dfa2154ba12c13631d.setContent%28html_34bee8840bc34f81b84897be1805cf8b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_cc745147c2554ab1b698bbdccdc9bf56.bindPopup%28popup_d9c5edbece7741dfa2154ba12c13631d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a7c39d24c1074e2395016cc89bb4d48e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7301%2C%20-79.3935%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_abb7f979f701411baf03a1ec56165a1a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_4f063ab6e86544afb4a84c89887082fe%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_4f063ab6e86544afb4a84c89887082fe%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELawrence%20Park%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_abb7f979f701411baf03a1ec56165a1a.setContent%28html_4f063ab6e86544afb4a84c89887082fe%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a7c39d24c1074e2395016cc89bb4d48e.bindPopup%28popup_abb7f979f701411baf03a1ec56165a1a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_bba9614c710448b6a93a205f1d1c5e03%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7113%2C%20-79.4195%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_8e86c952670b4594923c560c15589cc9%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_f39a7c87e4f0420ebb8e6bebcdc7d49a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_f39a7c87e4f0420ebb8e6bebcdc7d49a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERoselawn%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_8e86c952670b4594923c560c15589cc9.setContent%28html_f39a7c87e4f0420ebb8e6bebcdc7d49a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_bba9614c710448b6a93a205f1d1c5e03.bindPopup%28popup_8e86c952670b4594923c560c15589cc9%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_042a10f57a6b48efbbc2adfe7c7cc1c3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7135%2C%20-79.3887%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_a5b29291bc184048a9794481886e606b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_3a34c49a39b94458bbf5de48ee99e975%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_3a34c49a39b94458bbf5de48ee99e975%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDavisville%20North%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_a5b29291bc184048a9794481886e606b.setContent%28html_3a34c49a39b94458bbf5de48ee99e975%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_042a10f57a6b48efbbc2adfe7c7cc1c3.bindPopup%28popup_a5b29291bc184048a9794481886e606b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2811c6051ad540df91c63266e09f1a1a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6966%2C%20-79.41199999999998%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_2e3a9b41be24465987461851d56171a7%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_58b92d1cf39148fb8cab1ec622799b3c%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_58b92d1cf39148fb8cab1ec622799b3c%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EForest%20Hill%20North%20%26amp%3B%20West%2C%20Forest%20Hill%20Road%20Park%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_2e3a9b41be24465987461851d56171a7.setContent%28html_58b92d1cf39148fb8cab1ec622799b3c%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2811c6051ad540df91c63266e09f1a1a.bindPopup%28popup_2e3a9b41be24465987461851d56171a7%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6c7a68f13448436589a598a3cccbaff0%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6605%2C%20-79.4633%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1cca8f91f09445be805ee7ac243f4c4d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_27eef7d3d7434027a19a4fdad16d0ea1%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_27eef7d3d7434027a19a4fdad16d0ea1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHigh%20Park%2C%20The%20Junction%20South%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1cca8f91f09445be805ee7ac243f4c4d.setContent%28html_27eef7d3d7434027a19a4fdad16d0ea1%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_6c7a68f13448436589a598a3cccbaff0.bindPopup%28popup_1cca8f91f09445be805ee7ac243f4c4d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a18a81af10b74c0c9ec58c9259c1dc0f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7143%2C%20-79.4065%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_92c7b356d76e40cea898ceb648db7da6%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e1cdb80c1cfe4346870d19faa84d3841%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e1cdb80c1cfe4346870d19faa84d3841%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENorth%20Toronto%20West%2C%20%20Lawrence%20Park%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_92c7b356d76e40cea898ceb648db7da6.setContent%28html_e1cdb80c1cfe4346870d19faa84d3841%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a18a81af10b74c0c9ec58c9259c1dc0f.bindPopup%28popup_92c7b356d76e40cea898ceb648db7da6%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_518e20d562f94ed5abf201b153639e9f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6736%2C%20-79.4035%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f4c145f3472e465a9265b2d5a22cbacc%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c86f831b43374870a3c9b11505a0e752%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c86f831b43374870a3c9b11505a0e752%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThe%20Annex%2C%20North%20Midtown%2C%20Yorkville%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f4c145f3472e465a9265b2d5a22cbacc.setContent%28html_c86f831b43374870a3c9b11505a0e752%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_518e20d562f94ed5abf201b153639e9f.bindPopup%28popup_f4c145f3472e465a9265b2d5a22cbacc%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ca250e653c474306a238ad9405794df7%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6469%2C%20-79.4521%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_08d1365e1d334d9683fcbf5f8a94e72b%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_901166acb1e64913b706241589999bb1%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_901166acb1e64913b706241589999bb1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EParkdale%2C%20Roncesvalles%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_08d1365e1d334d9683fcbf5f8a94e72b.setContent%28html_901166acb1e64913b706241589999bb1%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ca250e653c474306a238ad9405794df7.bindPopup%28popup_08d1365e1d334d9683fcbf5f8a94e72b%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9a8fb657a0444681b07a7daff2c615db%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.702%2C%20-79.3853%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_460cc33b52f348b1ad486a14f95c8786%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d662ad8c0fd042e88a99069a93bc2797%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d662ad8c0fd042e88a99069a93bc2797%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDavisville%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_460cc33b52f348b1ad486a14f95c8786.setContent%28html_d662ad8c0fd042e88a99069a93bc2797%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9a8fb657a0444681b07a7daff2c615db.bindPopup%28popup_460cc33b52f348b1ad486a14f95c8786%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_693d2922a2d140b68abe23d962dc0745%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6629%2C%20-79.3987%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_5740bff262494f0783c5296ba37a6e4a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_57d8acc24aec46fc90b37632cd901640%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_57d8acc24aec46fc90b37632cd901640%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EUniversity%20of%20Toronto%2C%20Harbord%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_5740bff262494f0783c5296ba37a6e4a.setContent%28html_57d8acc24aec46fc90b37632cd901640%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_693d2922a2d140b68abe23d962dc0745.bindPopup%28popup_5740bff262494f0783c5296ba37a6e4a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ec8414200aed4ee9b6565af898fd5411%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6512%2C%20-79.4828%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e1092736505642e5bd420584658286bf%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_94264502dce84fe0a96568be0bb37868%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_94264502dce84fe0a96568be0bb37868%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERunnymede%2C%20Swansea%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e1092736505642e5bd420584658286bf.setContent%28html_94264502dce84fe0a96568be0bb37868%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ec8414200aed4ee9b6565af898fd5411.bindPopup%28popup_e1092736505642e5bd420584658286bf%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_86a1923f024e49a583061e9bdbb31039%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6899%2C%20-79.3853%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_391088e7491140b0b69aa44e6c582cd2%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e59b01b071b747e3aae6d80cda108742%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e59b01b071b747e3aae6d80cda108742%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMoore%20Park%2C%20Summerhill%20East%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_391088e7491140b0b69aa44e6c582cd2.setContent%28html_e59b01b071b747e3aae6d80cda108742%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_86a1923f024e49a583061e9bdbb31039.bindPopup%28popup_391088e7491140b0b69aa44e6c582cd2%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_ecc74261494546e99d89f88a87328715%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6541%2C%20-79.3978%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_fee9fcc5498f4b1a83ead235e869afec%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9aff0967ed5a4ee4859aa43c1e0186d8%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9aff0967ed5a4ee4859aa43c1e0186d8%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EKensington%20Market%2C%20Chinatown%2C%20Grange%20Park%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_fee9fcc5498f4b1a83ead235e869afec.setContent%28html_9aff0967ed5a4ee4859aa43c1e0186d8%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_ecc74261494546e99d89f88a87328715.bindPopup%28popup_fee9fcc5498f4b1a83ead235e869afec%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_13e0f1b290a74517a3f9341e75b7862f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6861%2C%20-79.4025%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_778014a55e5847169185880a989a3182%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_75083be924a8434c85ca19577ac1e5e9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_75083be924a8434c85ca19577ac1e5e9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESummerhill%20West%2C%20Rathnelly%2C%20South%20Hill%2C%20Forest%20Hill%20SE%2C%20Deer%20Park%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_778014a55e5847169185880a989a3182.setContent%28html_75083be924a8434c85ca19577ac1e5e9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_13e0f1b290a74517a3f9341e75b7862f.bindPopup%28popup_778014a55e5847169185880a989a3182%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_8e9c73bf0a0f47aea594056effedce3e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6404%2C%20-79.3995%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_1dc54aa212ee4bc6a989262bcd635776%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_52d464a3ef574dd89089874a176603f4%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_52d464a3ef574dd89089874a176603f4%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECN%20Tower%2C%20King%20and%20Spadina%2C%20Railway%20Lands%2C%20Harbourfront%20West%2C%20Bathurst%20Quay%2C%20South%20Niagara%2C%20Island%20airport%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_1dc54aa212ee4bc6a989262bcd635776.setContent%28html_52d464a3ef574dd89089874a176603f4%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_8e9c73bf0a0f47aea594056effedce3e.bindPopup%28popup_1dc54aa212ee4bc6a989262bcd635776%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e341d030f4e5462281f9c3e91e8b7651%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6827%2C%20-79.373%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_add1478b78704bc6950c93d7123bf191%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_9ffed1e4a7184c5b95f8ccc0bf9879b8%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_9ffed1e4a7184c5b95f8ccc0bf9879b8%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERosedale%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_add1478b78704bc6950c93d7123bf191.setContent%28html_9ffed1e4a7184c5b95f8ccc0bf9879b8%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e341d030f4e5462281f9c3e91e8b7651.bindPopup%28popup_add1478b78704bc6950c93d7123bf191%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_71bcd1ed04df4776ab3022f03e861d97%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6437%2C%20-79.3787%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_04f79047b7ac494d9a74f2500e31f108%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7e9c87bc204e44c2a780143f7982518d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7e9c87bc204e44c2a780143f7982518d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EStn%20A%20PO%20Boxes%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_04f79047b7ac494d9a74f2500e31f108.setContent%28html_7e9c87bc204e44c2a780143f7982518d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_71bcd1ed04df4776ab3022f03e861d97.bindPopup%28popup_04f79047b7ac494d9a74f2500e31f108%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_54dd64aee04949d2942172b66dd56ccb%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6684%2C%20-79.3689%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_59e8683a77754e81ab0858a9dec110b1%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e97c4fa5583547a985ff6cad1e5fb873%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e97c4fa5583547a985ff6cad1e5fb873%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESt.%20James%20Town%2C%20Cabbagetown%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_59e8683a77754e81ab0858a9dec110b1.setContent%28html_e97c4fa5583547a985ff6cad1e5fb873%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_54dd64aee04949d2942172b66dd56ccb.bindPopup%28popup_59e8683a77754e81ab0858a9dec110b1%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3446bf40f3b5435e9db4ffd3c0b3a6c3%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6492%2C%20-79.3823%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_4a520562111b48a6bb2361f2f504039a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_799dbcd88ef7488c831324161856814b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_799dbcd88ef7488c831324161856814b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFirst%20Canadian%20Place%2C%20Underground%20city%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_4a520562111b48a6bb2361f2f504039a.setContent%28html_799dbcd88ef7488c831324161856814b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_3446bf40f3b5435e9db4ffd3c0b3a6c3.bindPopup%28popup_4a520562111b48a6bb2361f2f504039a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_fb2d6031e66f487f8336b45fd56a3c1a%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6656%2C%20-79.383%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_798b2de9ef154290819fff00b0f187ed%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_15057b4fec1d45439fd5259e3273128d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_15057b4fec1d45439fd5259e3273128d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EChurch%20and%20Wellesley%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_798b2de9ef154290819fff00b0f187ed.setContent%28html_15057b4fec1d45439fd5259e3273128d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_fb2d6031e66f487f8336b45fd56a3c1a.bindPopup%28popup_798b2de9ef154290819fff00b0f187ed%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9832e555fcd540488c6aa5d7f3e0f7e9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7804%2C%20-79.2505%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22green%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22yellow%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_aff8975b381d4e56902015cc48c55ffa%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b238c4ac0ccc4686a99aba3da8519ebc%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_32e212e805654fe58aa642c5fa87df0f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_32e212e805654fe58aa642c5fa87df0f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBusiness%20reply%20mail%20Processing%20Centre%2C%20South%20Central%20Letter%20Processing%20Plant%20Toronto%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b238c4ac0ccc4686a99aba3da8519ebc.setContent%28html_32e212e805654fe58aa642c5fa87df0f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9832e555fcd540488c6aa5d7f3e0f7e9.bindPopup%28popup_b238c4ac0ccc4686a99aba3da8519ebc%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%3C/script%3E onload="this.contentDocument.open();this.contentDocument.write(    decodeURIComponent(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>





### Make Calls to Foursquare's API Using my Credentials For Top 100 Venues

### Create variables of Foursquare Credentials to be used later on to obtain a json file.


```python
# Fill in John Owusu Duah's Foursquare API credentials
CLIENT_ID = 'TNXDY4LAQETOXCXRS4EMLJ5DWZ2H2GOTMRKAJANYVT5Q0OO1' 
CLIENT_SECRET = 'LUD1GSVT2QMVXDEQX3QRB30NKAC4MZWDT1QCMIEUDBCXBXRJ' 
VERSION = '20180605' 
LIMIT = 100 
```

### Define function to collect data on the Top 100 Venues within a 500m Radius of all Neighbourhoods in Dataframe 

'getNearbyVenues' was defined to extract top 100 venues within a 500m radius of all neighborhoods in dataframe; extract the name of the neighborhood, latitude, longitude and categories of the venues and package the data into a dataframe named nearby_venues in one fell swoop


```python
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
```

Utilize getNearbyVenues for our all the neighborhoods in Toronto, earlier defined as df_tor


```python
toronto_venues = getNearbyVenues(names=df_tor['Neighborhood'],latitudes=df_tor['Latitude'],longitudes=df_tor['Longitude'])
```

    Regent Park, Harbourfront
    Queen's Park, Ontario Provincial Government
    Garden District, Ryerson
    St. James Town
    The Beaches
    Berczy Park
    Central Bay Street
    Christie
    Richmond, Adelaide, King
    Dufferin, Dovercourt Village
    Harbourfront East, Union Station, Toronto Islands
    Little Portugal, Trinity
    The Danforth West, Riverdale
    Toronto Dominion Centre, Design Exchange
    Brockton, Parkdale Village, Exhibition Place
    India Bazaar, The Beaches West
    Commerce Court, Victoria Hotel
    Studio District
    Lawrence Park
    Roselawn
    Davisville North
    Forest Hill North & West, Forest Hill Road Park
    High Park, The Junction South
    North Toronto West,  Lawrence Park
    The Annex, North Midtown, Yorkville
    Parkdale, Roncesvalles
    Davisville
    University of Toronto, Harbord
    Runnymede, Swansea
    Moore Park, Summerhill East
    Kensington Market, Chinatown, Grange Park
    Summerhill West, Rathnelly, South Hill, Forest Hill SE, Deer Park
    CN Tower, King and Spadina, Railway Lands, Harbourfront West, Bathurst Quay, South Niagara, Island airport
    Rosedale
    Stn A PO Boxes
    St. James Town, Cabbagetown
    First Canadian Place, Underground city
    Church and Wellesley
    Business reply mail Processing Centre, South Central Letter Processing Plant Toronto


### Check the shape of our dataframe that contains the categories and coordinates of top 100 the venues located within a 500m radius of all the neighborhoods in Toronto and display the first five rows of the dataframe


```python
print(toronto_venues.shape)
toronto_venues.head()
```

    (1519, 7)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.6555</td>
      <td>-79.3626</td>
      <td>Tandem Coffee</td>
      <td>43.653559</td>
      <td>-79.361809</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.6555</td>
      <td>-79.3626</td>
      <td>Roselle Desserts</td>
      <td>43.653447</td>
      <td>-79.362017</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.6555</td>
      <td>-79.3626</td>
      <td>Figs Breakfast &amp; Lunch</td>
      <td>43.655675</td>
      <td>-79.364503</td>
      <td>Breakfast Spot</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.6555</td>
      <td>-79.3626</td>
      <td>The Yoga Lounge</td>
      <td>43.655515</td>
      <td>-79.364955</td>
      <td>Yoga Studio</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Regent Park, Harbourfront</td>
      <td>43.6555</td>
      <td>-79.3626</td>
      <td>Body Blitz Spa East</td>
      <td>43.654735</td>
      <td>-79.359874</td>
      <td>Spa</td>
    </tr>
  </tbody>
</table>
</div>



Group the dataframe containing the venues within a 500m radius of all the neighbourhoods in Toronto by neighbourhood to find distribution of venues around all the neighborhoods.


```python
toronto_venues.groupby('Neighborhood').Venue.count()
```




    Neighborhood
    Berczy Park                                                                                                    92
    Brockton, Parkdale Village, Exhibition Place                                                                   38
    Business reply mail Processing Centre, South Central Letter Processing Plant Toronto                           14
    CN Tower, King and Spadina, Railway Lands, Harbourfront West, Bathurst Quay, South Niagara, Island airport     59
    Central Bay Street                                                                                             60
    Christie                                                                                                       11
    Church and Wellesley                                                                                           74
    Commerce Court, Victoria Hotel                                                                                100
    Davisville                                                                                                     22
    Davisville North                                                                                                6
    Dufferin, Dovercourt Village                                                                                   21
    First Canadian Place, Underground city                                                                        100
    Forest Hill North & West, Forest Hill Road Park                                                                 4
    Garden District, Ryerson                                                                                      100
    Harbourfront East, Union Station, Toronto Islands                                                               4
    High Park, The Junction South                                                                                   2
    India Bazaar, The Beaches West                                                                                 21
    Kensington Market, Chinatown, Grange Park                                                                      49
    Lawrence Park                                                                                                   2
    Little Portugal, Trinity                                                                                       44
    Moore Park, Summerhill East                                                                                     6
    North Toronto West,  Lawrence Park                                                                              4
    Parkdale, Roncesvalles                                                                                         48
    Queen's Park, Ontario Provincial Government                                                                    28
    Regent Park, Harbourfront                                                                                      22
    Richmond, Adelaide, King                                                                                      100
    Rosedale                                                                                                        4
    Roselawn                                                                                                        2
    Runnymede, Swansea                                                                                             50
    St. James Town                                                                                                 88
    St. James Town, Cabbagetown                                                                                    39
    Stn A PO Boxes                                                                                                 99
    Studio District                                                                                                 7
    Summerhill West, Rathnelly, South Hill, Forest Hill SE, Deer Park                                               6
    The Annex, North Midtown, Yorkville                                                                            24
    The Beaches                                                                                                     7
    The Danforth West, Riverdale                                                                                   36
    Toronto Dominion Centre, Design Exchange                                                                      100
    University of Toronto, Harbord                                                                                 26
    Name: Venue, dtype: int64



### Create a dataframe which has normalized data of each venue category so that I can feed it into the k-means algorithm to cluster the neighborhoods according to venue categories. 

First, I created dummy variables of all the venue categories of each neighborhood. To do this, I had to make sure that my new dataframe with dummy variables had corresponding rows of neighbors. To do this efficiently, I had to concatenate the dummy variable dataframe and the torornto_venues dataframe, after which I ensured that there was only one column named 'neighborhood' and dropped unneccessary columns.


```python
# Creating dummy columns of venue categorie for each neighborhood in Toronto
toronto_dum = pd.get_dummies(toronto_venues[['Venue Category']], prefix="", prefix_sep="")

# Concatenate dummy dataframe with the original dataframe to get the neighborhoods and dummies of venue categorie 
# in one DataFrame
df_one = pd.concat([toronto_venues, toronto_dum], axis=1)

# Afer concatenating, we realised that there were two neighbourhood column labels so we looped through the column 
# names and assigned separate suffixes to them so that we can drop the duplicate next.
cols = []
count = 1
for column in df_one.columns:
    if column == 'Neighborhood':
        cols.append(f'Neighborhood_{count}')
        count+=1
        continue
    cols.append(column)
df_one.columns = cols

# Remove unnecessary columns
df_one.drop(['Neighborhood Latitude', 'Neighborhood Longitude', 'Venue', 'Venue Latitude', 'Venue Longitude','Venue Category','Neighborhood_2'], axis=1, inplace=True)

# Rename neighborhood_1 back to neighborhood
df_one = df_one.rename(columns={"Neighborhood_1":"Neighborhood"})
```


```python
# Explore dummy dataframe
df_one.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Accessories Store</th>
      <th>Adult Boutique</th>
      <th>American Restaurant</th>
      <th>Art Gallery</th>
      <th>Art Museum</th>
      <th>Arts &amp; Crafts Store</th>
      <th>Asian Restaurant</th>
      <th>Athletics &amp; Sports</th>
      <th>BBQ Joint</th>
      <th>...</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Veterinarian</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Whisky Bar</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Wings Joint</th>
      <th>Yoga Studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 212 columns</p>
</div>



The dataframe with the dummy venue categries is aggregated by neighborhood and the mean of dummy entries for each venue category is computed. Computing the mean is equivalent normalizing data since the data is scaled to a range of 0 - 1 depending on the frequency. 

###  The k-means algorithm will be run using the daframe below without the neighborhood columm


```python
df_one_grouped = df_one.groupby(by='Neighborhood', axis=0).mean().reset_index()
df_one_grouped
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Accessories Store</th>
      <th>Adult Boutique</th>
      <th>American Restaurant</th>
      <th>Art Gallery</th>
      <th>Art Museum</th>
      <th>Arts &amp; Crafts Store</th>
      <th>Asian Restaurant</th>
      <th>Athletics &amp; Sports</th>
      <th>BBQ Joint</th>
      <th>...</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Veterinarian</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Whisky Bar</th>
      <th>Wine Bar</th>
      <th>Wine Shop</th>
      <th>Wings Joint</th>
      <th>Yoga Studio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Berczy Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010870</td>
      <td>0.021739</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.010870</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010870</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brockton, Parkdale Village, Exhibition Place</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.026316</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Business reply mail Processing Centre, South C...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CN Tower, King and Spadina, Railway Lands, Har...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.016949</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.016949</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.016949</td>
      <td>0.016949</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Central Bay Street</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.016667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.016667</td>
      <td>0.016667</td>
      <td>0.000000</td>
      <td>0.016667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Christie</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Church and Wellesley</td>
      <td>0.000000</td>
      <td>0.013514</td>
      <td>0.013514</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027027</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Commerce Court, Victoria Hotel</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Davisville</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Davisville North</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Dufferin, Dovercourt Village</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.047619</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>First Canadian Place, Underground city</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Forest Hill North &amp; West, Forest Hill Road Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Garden District, Ryerson</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Harbourfront East, Union Station, Toronto Islands</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>High Park, The Junction South</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>India Bazaar, The Beaches West</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kensington Market, Chinatown, Grange Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.081633</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.040816</td>
      <td>0.000000</td>
      <td>0.020408</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Lawrence Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Little Portugal, Trinity</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.022727</td>
      <td>0.022727</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.022727</td>
      <td>0.000000</td>
      <td>0.022727</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.022727</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Moore Park, Summerhill East</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>North Toronto West,  Lawrence Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Parkdale, Roncesvalles</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020833</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Queen's Park, Ontario Provincial Government</td>
      <td>0.000000</td>
      <td>0.035714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Regent Park, Harbourfront</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.045455</td>
      <td>0.000000</td>
      <td>0.045455</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Richmond, Adelaide, King</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Rosedale</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Roselawn</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Runnymede, Swansea</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
    </tr>
    <tr>
      <th>29</th>
      <td>St. James Town</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.034091</td>
      <td>0.022727</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011364</td>
      <td>0.000000</td>
      <td>0.011364</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.011364</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011364</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>St. James Town, Cabbagetown</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Stn A PO Boxes</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010101</td>
      <td>0.010101</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.010101</td>
      <td>0.010101</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010101</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Studio District</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Summerhill West, Rathnelly, South Hill, Forest...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>The Annex, North Midtown, Yorkville</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.041667</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>The Beaches</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>The Danforth West, Riverdale</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.027778</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Toronto Dominion Centre, Design Exchange</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>38</th>
      <td>University of Toronto, Harbord</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.038462</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.038462</td>
    </tr>
  </tbody>
</table>
<p>39 rows × 212 columns</p>
</div>



The function 'return_most_common_venues' was created to be used to sort our venues in descending order of frequency in display them in our dataframe.


```python
# Define a function to sort out the venues in descending order of frequency
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
```

### Create a new dataframe and display the top 8 venues for each neighborhood


```python
# Lets  us create the new dataframe and display the top 8 venues for each neighborhood.
num_top_venues = 8

indicators = ['st', 'nd', 'rd']

# Create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = df_one_grouped['Neighborhood']

for ind in np.arange(df_one_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(df_one_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Berczy Park</td>
      <td>Coffee Shop</td>
      <td>Hotel</td>
      <td>Café</td>
      <td>Seafood Restaurant</td>
      <td>Bakery</td>
      <td>Cocktail Bar</td>
      <td>Restaurant</td>
      <td>Japanese Restaurant</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Brockton, Parkdale Village, Exhibition Place</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Breakfast Spot</td>
      <td>Gift Shop</td>
      <td>Thrift / Vintage Store</td>
      <td>Pet Store</td>
      <td>Brewery</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Business reply mail Processing Centre, South C...</td>
      <td>Restaurant</td>
      <td>Coffee Shop</td>
      <td>Yoga Studio</td>
      <td>Bank</td>
      <td>Breakfast Spot</td>
      <td>Bookstore</td>
      <td>Sushi Restaurant</td>
      <td>Japanese Restaurant</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CN Tower, King and Spadina, Railway Lands, Har...</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Italian Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Bar</td>
      <td>Speakeasy</td>
      <td>Restaurant</td>
      <td>Bank</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Central Bay Street</td>
      <td>Coffee Shop</td>
      <td>Middle Eastern Restaurant</td>
      <td>Sandwich Place</td>
      <td>Bubble Tea Shop</td>
      <td>Café</td>
      <td>Clothing Store</td>
      <td>Italian Restaurant</td>
      <td>Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



### The k-means algorithm is run to segment the neighborhoods into five clusters


```python
# set number of clusters
kclusters = 5

df_one_grouped_clustering = df_one_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(df_one_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)



A new dataframe that includes the cluster labels as well as the top 10 venues for each neighborhood will be created so that it can serve as the data for a Folium map for visualization of clustered neighborhoods


```python
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

df_merged = df_tor

# merge manhattan_grouped with manhattan_data to add latitude/longitude for each neighborhood
df_merged = df_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

df_merged.head() # check the last columns!
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postalcode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park, Harbourfront</td>
      <td>43.6555</td>
      <td>-79.3626</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Breakfast Spot</td>
      <td>Yoga Studio</td>
      <td>Theater</td>
      <td>Distribution Center</td>
      <td>Pub</td>
      <td>Restaurant</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M7A</td>
      <td>Downtown Toronto</td>
      <td>Queen's Park, Ontario Provincial Government</td>
      <td>43.6641</td>
      <td>-79.3889</td>
      <td>0</td>
      <td>Gym</td>
      <td>Sushi Restaurant</td>
      <td>Coffee Shop</td>
      <td>Ethiopian Restaurant</td>
      <td>Bubble Tea Shop</td>
      <td>Café</td>
      <td>Martial Arts School</td>
      <td>Mexican Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5B</td>
      <td>Downtown Toronto</td>
      <td>Garden District, Ryerson</td>
      <td>43.6572</td>
      <td>-79.3783</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Clothing Store</td>
      <td>Cosmetics Shop</td>
      <td>Italian Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Middle Eastern Restaurant</td>
      <td>Café</td>
      <td>Movie Theater</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M5C</td>
      <td>Downtown Toronto</td>
      <td>St. James Town</td>
      <td>43.6513</td>
      <td>-79.3756</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Seafood Restaurant</td>
      <td>Cocktail Bar</td>
      <td>Bakery</td>
      <td>American Restaurant</td>
      <td>Gastropub</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M4E</td>
      <td>East Toronto</td>
      <td>The Beaches</td>
      <td>43.6784</td>
      <td>-79.2941</td>
      <td>0</td>
      <td>Pub</td>
      <td>Gastropub</td>
      <td>Health Food Store</td>
      <td>Bakery</td>
      <td>Trail</td>
      <td>Donut Shop</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



### A Visualization of Clustered Neighborhoods in Toronto on an Open Street May Utilizing Python's Folium Library


```python
# create follium map
map_cluster_tor = folium.Map(location=[43.653226, -79.383184], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(df_merged['Latitude'], df_merged['Longitude'], df_merged['Neighborhood'], df_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_cluster_tor)
       
map_cluster_tor
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span><iframe src="about:blank" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" data-html=%3C%21DOCTYPE%20html%3E%0A%3Chead%3E%20%20%20%20%0A%20%20%20%20%3Cmeta%20http-equiv%3D%22content-type%22%20content%3D%22text/html%3B%20charset%3DUTF-8%22%20/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%3Cscript%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20L_NO_TOUCH%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20L_DISABLE_3D%20%3D%20false%3B%0A%20%20%20%20%20%20%20%20%3C/script%3E%0A%20%20%20%20%0A%20%20%20%20%3Cstyle%3Ehtml%2C%20body%20%7Bwidth%3A%20100%25%3Bheight%3A%20100%25%3Bmargin%3A%200%3Bpadding%3A%200%3B%7D%3C/style%3E%0A%20%20%20%20%3Cstyle%3E%23map%20%7Bposition%3Aabsolute%3Btop%3A0%3Bbottom%3A0%3Bright%3A0%3Bleft%3A0%3B%7D%3C/style%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//code.jquery.com/jquery-1.12.4.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/js/bootstrap.min.js%22%3E%3C/script%3E%0A%20%20%20%20%3Cscript%20src%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.js%22%3E%3C/script%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/npm/leaflet%401.6.0/dist/leaflet.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/bootstrap/3.2.0/css/bootstrap-theme.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//maxcdn.bootstrapcdn.com/font-awesome/4.6.3/css/font-awesome.min.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdnjs.cloudflare.com/ajax/libs/Leaflet.awesome-markers/2.0.2/leaflet.awesome-markers.css%22/%3E%0A%20%20%20%20%3Clink%20rel%3D%22stylesheet%22%20href%3D%22https%3A//cdn.jsdelivr.net/gh/python-visualization/folium/folium/templates/leaflet.awesome.rotate.min.css%22/%3E%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cmeta%20name%3D%22viewport%22%20content%3D%22width%3Ddevice-width%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20initial-scale%3D1.0%2C%20maximum-scale%3D1.0%2C%20user-scalable%3Dno%22%20/%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cstyle%3E%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%23map_fb9a021fdb164ead8f3640d4ceb465de%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20position%3A%20relative%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20width%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20height%3A%20100.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20left%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20top%3A%200.0%25%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%3C/style%3E%0A%20%20%20%20%20%20%20%20%0A%3C/head%3E%0A%3Cbody%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%3Cdiv%20class%3D%22folium-map%22%20id%3D%22map_fb9a021fdb164ead8f3640d4ceb465de%22%20%3E%3C/div%3E%0A%20%20%20%20%20%20%20%20%0A%3C/body%3E%0A%3Cscript%3E%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20map_fb9a021fdb164ead8f3640d4ceb465de%20%3D%20L.map%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22map_fb9a021fdb164ead8f3640d4ceb465de%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20center%3A%20%5B43.653226%2C%20-79.383184%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20crs%3A%20L.CRS.EPSG3857%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoom%3A%2011%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20zoomControl%3A%20true%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20preferCanvas%3A%20false%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29%3B%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20tile_layer_aaf8777e494f42d59190e0f4ef56721b%20%3D%20L.tileLayer%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%22https%3A//%7Bs%7D.tile.openstreetmap.org/%7Bz%7D/%7Bx%7D/%7By%7D.png%22%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22attribution%22%3A%20%22Data%20by%20%5Cu0026copy%3B%20%5Cu003ca%20href%3D%5C%22http%3A//openstreetmap.org%5C%22%5Cu003eOpenStreetMap%5Cu003c/a%5Cu003e%2C%20under%20%5Cu003ca%20href%3D%5C%22http%3A//www.openstreetmap.org/copyright%5C%22%5Cu003eODbL%5Cu003c/a%5Cu003e.%22%2C%20%22detectRetina%22%3A%20false%2C%20%22maxNativeZoom%22%3A%2018%2C%20%22maxZoom%22%3A%2018%2C%20%22minZoom%22%3A%200%2C%20%22noWrap%22%3A%20false%2C%20%22opacity%22%3A%201%2C%20%22subdomains%22%3A%20%22abc%22%2C%20%22tms%22%3A%20false%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_cc89e1eaf4a04f3ba12228bbcb2bb03b%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6555%2C%20-79.3626%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e2cd6f128a8b4f9eacf46f7e3cff8377%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_711633820be445a4812f132ef270a66f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_711633820be445a4812f132ef270a66f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERegent%20Park%2C%20Harbourfront%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e2cd6f128a8b4f9eacf46f7e3cff8377.setContent%28html_711633820be445a4812f132ef270a66f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_cc89e1eaf4a04f3ba12228bbcb2bb03b.bindPopup%28popup_e2cd6f128a8b4f9eacf46f7e3cff8377%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_15412a1a5b71460887c7ec23535a1d89%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6641%2C%20-79.3889%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_0b05677787494bd28e6c15b7e66967bf%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_fa285ee7c1284303953445a3590ea465%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_fa285ee7c1284303953445a3590ea465%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EQueen%26%2339%3Bs%20Park%2C%20Ontario%20Provincial%20Government%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_0b05677787494bd28e6c15b7e66967bf.setContent%28html_fa285ee7c1284303953445a3590ea465%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_15412a1a5b71460887c7ec23535a1d89.bindPopup%28popup_0b05677787494bd28e6c15b7e66967bf%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_e1593c153d7441649a96e7bbcde3f7d7%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6572%2C%20-79.3783%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9547e10013a845359f2b64927ceac8aa%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6dc70ef0086b446390b19602030eb8bc%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_6dc70ef0086b446390b19602030eb8bc%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EGarden%20District%2C%20Ryerson%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9547e10013a845359f2b64927ceac8aa.setContent%28html_6dc70ef0086b446390b19602030eb8bc%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_e1593c153d7441649a96e7bbcde3f7d7.bindPopup%28popup_9547e10013a845359f2b64927ceac8aa%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_565222a24a0e4cc3ac94d72087b8f2e9%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6513%2C%20-79.3756%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_ba4c267324a24c4599ed6f09aa612b2d%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1856b8201b9e4aaea5ae8fd8ac592ba2%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1856b8201b9e4aaea5ae8fd8ac592ba2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESt.%20James%20Town%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_ba4c267324a24c4599ed6f09aa612b2d.setContent%28html_1856b8201b9e4aaea5ae8fd8ac592ba2%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_565222a24a0e4cc3ac94d72087b8f2e9.bindPopup%28popup_ba4c267324a24c4599ed6f09aa612b2d%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_de2895b05f814d3696fe18d224b3cca8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6784%2C%20-79.2941%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d841fd3a7f5f46d38dcca85ce956bc89%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e3dacdb3824a44e4af81d0e2abdd72dc%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e3dacdb3824a44e4af81d0e2abdd72dc%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThe%20Beaches%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d841fd3a7f5f46d38dcca85ce956bc89.setContent%28html_e3dacdb3824a44e4af81d0e2abdd72dc%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_de2895b05f814d3696fe18d224b3cca8.bindPopup%28popup_d841fd3a7f5f46d38dcca85ce956bc89%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d0beb596ad8641c2b57644d573352400%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6456%2C%20-79.3754%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_88181b9ab72b4581bed7b97ab8ed7f79%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a7c182cd787742868c8431e26e146fb1%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_a7c182cd787742868c8431e26e146fb1%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBerczy%20Park%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_88181b9ab72b4581bed7b97ab8ed7f79.setContent%28html_a7c182cd787742868c8431e26e146fb1%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d0beb596ad8641c2b57644d573352400.bindPopup%28popup_88181b9ab72b4581bed7b97ab8ed7f79%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f4744311402944ad95c74bb53f7244a8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6564%2C%20-79.38600000000002%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_36d6c30c46414ad1bb433604b3749761%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e069d043257540129158a450a5528628%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e069d043257540129158a450a5528628%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECentral%20Bay%20Street%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_36d6c30c46414ad1bb433604b3749761.setContent%28html_e069d043257540129158a450a5528628%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f4744311402944ad95c74bb53f7244a8.bindPopup%28popup_36d6c30c46414ad1bb433604b3749761%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_588ff503205d4778830c2afe84e6758d%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6683%2C%20-79.4205%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6904629dbb40479fb340e96f37299ff1%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_85d20c6924784568b0234a5b84fe4113%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_85d20c6924784568b0234a5b84fe4113%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EChristie%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6904629dbb40479fb340e96f37299ff1.setContent%28html_85d20c6924784568b0234a5b84fe4113%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_588ff503205d4778830c2afe84e6758d.bindPopup%28popup_6904629dbb40479fb340e96f37299ff1%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f1843c43d8ac421e96468205d7600b77%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6496%2C%20-79.3833%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_dad6d9cf815748cdaae4081209702366%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c937cbedeec24006aca44eab703ba221%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c937cbedeec24006aca44eab703ba221%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERichmond%2C%20Adelaide%2C%20King%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_dad6d9cf815748cdaae4081209702366.setContent%28html_c937cbedeec24006aca44eab703ba221%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f1843c43d8ac421e96468205d7600b77.bindPopup%28popup_dad6d9cf815748cdaae4081209702366%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_3a56c5638d524aec89736890eb1545ff%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6655%2C%20-79.4378%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d39ae9f6486948b58797778836d6560e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_6e7e304f14664cc5b251013b9c4d0025%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_6e7e304f14664cc5b251013b9c4d0025%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDufferin%2C%20Dovercourt%20Village%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d39ae9f6486948b58797778836d6560e.setContent%28html_6e7e304f14664cc5b251013b9c4d0025%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_3a56c5638d524aec89736890eb1545ff.bindPopup%28popup_d39ae9f6486948b58797778836d6560e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_b11920a2ed664759bf4b389127e9c42e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.62300000000001%2C%20-79.3936%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_908ed5e7451b43c4888a08eb82a1d9c8%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ffe9c8e88f14452e82198b5348f795ba%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ffe9c8e88f14452e82198b5348f795ba%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHarbourfront%20East%2C%20Union%20Station%2C%20Toronto%20Islands%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_908ed5e7451b43c4888a08eb82a1d9c8.setContent%28html_ffe9c8e88f14452e82198b5348f795ba%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_b11920a2ed664759bf4b389127e9c42e.bindPopup%28popup_908ed5e7451b43c4888a08eb82a1d9c8%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_52816accc4044ac1bf15ab6e9f1f0356%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.648%2C%20-79.4177%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_cef60b92eb724b068cf8a56753b90a80%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_811a11d48a1c4f478e7be447ad0fb896%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_811a11d48a1c4f478e7be447ad0fb896%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELittle%20Portugal%2C%20Trinity%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_cef60b92eb724b068cf8a56753b90a80.setContent%28html_811a11d48a1c4f478e7be447ad0fb896%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_52816accc4044ac1bf15ab6e9f1f0356.bindPopup%28popup_cef60b92eb724b068cf8a56753b90a80%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f6a1a92c6ba340b9a3283f68f34d92df%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6803%2C%20-79.3538%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e749204e031a4e4f8ca793bfbf6481ca%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_eddf7067df614c19a8b175d75c959f68%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_eddf7067df614c19a8b175d75c959f68%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThe%20Danforth%20West%2C%20Riverdale%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e749204e031a4e4f8ca793bfbf6481ca.setContent%28html_eddf7067df614c19a8b175d75c959f68%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f6a1a92c6ba340b9a3283f68f34d92df.bindPopup%28popup_e749204e031a4e4f8ca793bfbf6481ca%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_aaec87f51caf48cd8c4f2774dca5f898%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6469%2C%20-79.3823%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_dfa455dc4428477da17c0c7148194cac%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_ad12ef912a6c4330a5d4d3f49a84b3b3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_ad12ef912a6c4330a5d4d3f49a84b3b3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EToronto%20Dominion%20Centre%2C%20Design%20Exchange%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_dfa455dc4428477da17c0c7148194cac.setContent%28html_ad12ef912a6c4330a5d4d3f49a84b3b3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_aaec87f51caf48cd8c4f2774dca5f898.bindPopup%28popup_dfa455dc4428477da17c0c7148194cac%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_4db834a161884ff2a20ac8a07b5c0ea8%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6383%2C%20-79.4301%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_62f230e50a32402f96a694c3dc5fc5ce%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_a5c69b8632a3452195cd65117b50ec81%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_a5c69b8632a3452195cd65117b50ec81%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBrockton%2C%20Parkdale%20Village%2C%20Exhibition%20Place%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_62f230e50a32402f96a694c3dc5fc5ce.setContent%28html_a5c69b8632a3452195cd65117b50ec81%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_4db834a161884ff2a20ac8a07b5c0ea8.bindPopup%28popup_62f230e50a32402f96a694c3dc5fc5ce%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_9f92d531a2cb4d46bc86a0d1af41e0d1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6693%2C%20-79.3155%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_c01151aa1cbc43989bc32ad9cef7e537%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_c8d05f49ffc444e3bab69c439e0b98c9%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_c8d05f49ffc444e3bab69c439e0b98c9%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EIndia%20Bazaar%2C%20The%20Beaches%20West%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_c01151aa1cbc43989bc32ad9cef7e537.setContent%28html_c8d05f49ffc444e3bab69c439e0b98c9%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_9f92d531a2cb4d46bc86a0d1af41e0d1.bindPopup%28popup_c01151aa1cbc43989bc32ad9cef7e537%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_35d1228cd1ed4e20bd88ad9a347b8bcc%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6492%2C%20-79.3823%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_e2100f31c4e44a06953e79f1416bb346%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d4b7140754d646ef96c2dbc464dad3c3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d4b7140754d646ef96c2dbc464dad3c3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECommerce%20Court%2C%20Victoria%20Hotel%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_e2100f31c4e44a06953e79f1416bb346.setContent%28html_d4b7140754d646ef96c2dbc464dad3c3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_35d1228cd1ed4e20bd88ad9a347b8bcc.bindPopup%28popup_e2100f31c4e44a06953e79f1416bb346%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_194127885c38452d9e6a5ed5c2810fd1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6561%2C%20-79.3406%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_97cb7dcd5ba8424691f270025e3abe41%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_eb40662d77ff44a7af7e0fb570d956ca%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_eb40662d77ff44a7af7e0fb570d956ca%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EStudio%20District%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_97cb7dcd5ba8424691f270025e3abe41.setContent%28html_eb40662d77ff44a7af7e0fb570d956ca%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_194127885c38452d9e6a5ed5c2810fd1.bindPopup%28popup_97cb7dcd5ba8424691f270025e3abe41%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_97552537487e4f668847b464a7e081b2%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7301%2C%20-79.3935%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ffb360%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ffb360%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_b5f3eebefea04af1a4d5d72f9fb24909%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_7a56cca23797430c81a6b9bc34d1a74d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_7a56cca23797430c81a6b9bc34d1a74d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ELawrence%20Park%20Cluster%204%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_b5f3eebefea04af1a4d5d72f9fb24909.setContent%28html_7a56cca23797430c81a6b9bc34d1a74d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_97552537487e4f668847b464a7e081b2.bindPopup%28popup_b5f3eebefea04af1a4d5d72f9fb24909%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5aed07b372f743eb9f5d4d32641c2fa6%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7113%2C%20-79.4195%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%2380ffb4%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%2380ffb4%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_2f7939763efb4af6aa89c57e062c13c1%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b2c56abe62864cf490557f38bc09cafa%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b2c56abe62864cf490557f38bc09cafa%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERoselawn%20Cluster%203%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_2f7939763efb4af6aa89c57e062c13c1.setContent%28html_b2c56abe62864cf490557f38bc09cafa%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_5aed07b372f743eb9f5d4d32641c2fa6.bindPopup%28popup_2f7939763efb4af6aa89c57e062c13c1%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_bb00be69e45a44059512c457fe907457%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7135%2C%20-79.3887%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_8e7e9d2f67344971919a4e51d087b601%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_8972f6514e434d6f8be2fea2cff90926%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_8972f6514e434d6f8be2fea2cff90926%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDavisville%20North%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_8e7e9d2f67344971919a4e51d087b601.setContent%28html_8972f6514e434d6f8be2fea2cff90926%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_bb00be69e45a44059512c457fe907457.bindPopup%28popup_8e7e9d2f67344971919a4e51d087b601%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_7910831fbd53445790d4c97fe8c9d437%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6966%2C%20-79.41199999999998%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ffb360%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ffb360%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_7e2e90e05d354640b6197469ccd716ea%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_00d7182894824cec817d38540497a123%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_00d7182894824cec817d38540497a123%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EForest%20Hill%20North%20%26amp%3B%20West%2C%20Forest%20Hill%20Road%20Park%20Cluster%204%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_7e2e90e05d354640b6197469ccd716ea.setContent%28html_00d7182894824cec817d38540497a123%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_7910831fbd53445790d4c97fe8c9d437.bindPopup%28popup_7e2e90e05d354640b6197469ccd716ea%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_210354591e0d40d98680a3d2d6cc7df5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6605%2C%20-79.4633%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%238000ff%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%238000ff%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d39fce1a9f7c482bb77290271ce1737f%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_06ca30d3318341999fb308effcb0b83d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_06ca30d3318341999fb308effcb0b83d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EHigh%20Park%2C%20The%20Junction%20South%20Cluster%201%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d39fce1a9f7c482bb77290271ce1737f.setContent%28html_06ca30d3318341999fb308effcb0b83d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_210354591e0d40d98680a3d2d6cc7df5.bindPopup%28popup_d39fce1a9f7c482bb77290271ce1737f%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_059fa4c31eb14e2bb94b165a012de788%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7143%2C%20-79.4065%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ffb360%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ffb360%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9de3a60d380e4077924fd02d6c1bd969%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_0f785cc0169b4bbe875d067d0e43c0b3%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_0f785cc0169b4bbe875d067d0e43c0b3%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ENorth%20Toronto%20West%2C%20%20Lawrence%20Park%20Cluster%204%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9de3a60d380e4077924fd02d6c1bd969.setContent%28html_0f785cc0169b4bbe875d067d0e43c0b3%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_059fa4c31eb14e2bb94b165a012de788.bindPopup%28popup_9de3a60d380e4077924fd02d6c1bd969%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_c7f7b21b7d2a432e94c5053d6786f6f6%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6736%2C%20-79.4035%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_cd0d57f3270b4ff9a64fff3fe94a7c62%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_01d3c230abf34020a5cbae9c8c5fc4f2%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_01d3c230abf34020a5cbae9c8c5fc4f2%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EThe%20Annex%2C%20North%20Midtown%2C%20Yorkville%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_cd0d57f3270b4ff9a64fff3fe94a7c62.setContent%28html_01d3c230abf34020a5cbae9c8c5fc4f2%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_c7f7b21b7d2a432e94c5053d6786f6f6.bindPopup%28popup_cd0d57f3270b4ff9a64fff3fe94a7c62%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_d248b2c594a1448db27d5b1c36ef8010%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6469%2C%20-79.4521%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_de0784bae968462eb97fc63d685adfb4%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_26e7d59e823549519d02d87205d72fb8%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_26e7d59e823549519d02d87205d72fb8%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EParkdale%2C%20Roncesvalles%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_de0784bae968462eb97fc63d685adfb4.setContent%28html_26e7d59e823549519d02d87205d72fb8%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_d248b2c594a1448db27d5b1c36ef8010.bindPopup%28popup_de0784bae968462eb97fc63d685adfb4%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_a00b2d8783d943b5b6223588accb4523%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.702%2C%20-79.3853%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_8c510cb979ad4bd0ad6e871c4095f142%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_208ed93ab3d94d51be091670f422cd6f%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_208ed93ab3d94d51be091670f422cd6f%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EDavisville%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_8c510cb979ad4bd0ad6e871c4095f142.setContent%28html_208ed93ab3d94d51be091670f422cd6f%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_a00b2d8783d943b5b6223588accb4523.bindPopup%28popup_8c510cb979ad4bd0ad6e871c4095f142%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_5b2a7e91342c4d55a6868f05fb715a84%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6629%2C%20-79.3987%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_28a5699cd33d48f8bcd4203c35a9fd6a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d5a1fb60926e4d40a058f5169fb62afd%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d5a1fb60926e4d40a058f5169fb62afd%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EUniversity%20of%20Toronto%2C%20Harbord%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_28a5699cd33d48f8bcd4203c35a9fd6a.setContent%28html_d5a1fb60926e4d40a058f5169fb62afd%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_5b2a7e91342c4d55a6868f05fb715a84.bindPopup%28popup_28a5699cd33d48f8bcd4203c35a9fd6a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f1b1cadeef7b4f359db543e575f87564%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6512%2C%20-79.4828%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_9c8e3026bb52444dadb71d367b8a61ab%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_1960e0a7f73b470c9ab5a8dde8c80324%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_1960e0a7f73b470c9ab5a8dde8c80324%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERunnymede%2C%20Swansea%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_9c8e3026bb52444dadb71d367b8a61ab.setContent%28html_1960e0a7f73b470c9ab5a8dde8c80324%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f1b1cadeef7b4f359db543e575f87564.bindPopup%28popup_9c8e3026bb52444dadb71d367b8a61ab%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_6a2a99522257438fa2549c6b709fd6ef%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6899%2C%20-79.3853%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ffb360%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ffb360%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_07d54755775346efb1a29dec739a39cb%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_db1f9ae50b364141b2ee8d14dc55aa30%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_db1f9ae50b364141b2ee8d14dc55aa30%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EMoore%20Park%2C%20Summerhill%20East%20Cluster%204%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_07d54755775346efb1a29dec739a39cb.setContent%28html_db1f9ae50b364141b2ee8d14dc55aa30%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_6a2a99522257438fa2549c6b709fd6ef.bindPopup%28popup_07d54755775346efb1a29dec739a39cb%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_f237729a687e40b4800a7bcbdc3c382e%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6541%2C%20-79.3978%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_6c07e6d0501d43eba9480d0a707ae76a%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_eb7fab5f5e7c4985aeb34137bc2c7e91%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_eb7fab5f5e7c4985aeb34137bc2c7e91%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EKensington%20Market%2C%20Chinatown%2C%20Grange%20Park%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_6c07e6d0501d43eba9480d0a707ae76a.setContent%28html_eb7fab5f5e7c4985aeb34137bc2c7e91%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_f237729a687e40b4800a7bcbdc3c382e.bindPopup%28popup_6c07e6d0501d43eba9480d0a707ae76a%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_49bb9c75b5094ec2916a29656739722f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6861%2C%20-79.4025%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%2300b5eb%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%2300b5eb%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d6f4a610c03149e2990b8d444b386d92%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_33eaac0990d94d25a9a4badd42ebbb1d%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_33eaac0990d94d25a9a4badd42ebbb1d%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESummerhill%20West%2C%20Rathnelly%2C%20South%20Hill%2C%20Forest%20Hill%20SE%2C%20Deer%20Park%20Cluster%202%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d6f4a610c03149e2990b8d444b386d92.setContent%28html_33eaac0990d94d25a9a4badd42ebbb1d%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_49bb9c75b5094ec2916a29656739722f.bindPopup%28popup_d6f4a610c03149e2990b8d444b386d92%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_661aafb8659b4f1d9f44e413d592e3ac%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6404%2C%20-79.3995%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_d5806d88dd38413dbbeb6281f3396f20%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_150fe506d0a749bfb982bc134c64a013%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_150fe506d0a749bfb982bc134c64a013%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ECN%20Tower%2C%20King%20and%20Spadina%2C%20Railway%20Lands%2C%20Harbourfront%20West%2C%20Bathurst%20Quay%2C%20South%20Niagara%2C%20Island%20airport%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_d5806d88dd38413dbbeb6281f3396f20.setContent%28html_150fe506d0a749bfb982bc134c64a013%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_661aafb8659b4f1d9f44e413d592e3ac.bindPopup%28popup_d5806d88dd38413dbbeb6281f3396f20%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_8fa29f8c3ef34f1990270326826de465%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6827%2C%20-79.373%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ffb360%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ffb360%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_bb7b08f313164ff1b18ae1dd3784c52e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d8cf4e92c5ee4733a86318c11a8aba78%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d8cf4e92c5ee4733a86318c11a8aba78%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ERosedale%20Cluster%204%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_bb7b08f313164ff1b18ae1dd3784c52e.setContent%28html_d8cf4e92c5ee4733a86318c11a8aba78%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_8fa29f8c3ef34f1990270326826de465.bindPopup%28popup_bb7b08f313164ff1b18ae1dd3784c52e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_952a1d4298424d35965f0d0ca412a8a5%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6437%2C%20-79.3787%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_f2358b06d50f4d3d956c1f4e44c5b645%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d95fdfd0f97a40e5b07ea7bd8f4b7d2a%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d95fdfd0f97a40e5b07ea7bd8f4b7d2a%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EStn%20A%20PO%20Boxes%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_f2358b06d50f4d3d956c1f4e44c5b645.setContent%28html_d95fdfd0f97a40e5b07ea7bd8f4b7d2a%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_952a1d4298424d35965f0d0ca412a8a5.bindPopup%28popup_f2358b06d50f4d3d956c1f4e44c5b645%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_41d7c801d46c4d898ed25f13993422e1%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6684%2C%20-79.3689%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_05c3ec53d88f403c8ea9b9c5b7b24a58%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_887dea9740a645488cc024370cb034df%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_887dea9740a645488cc024370cb034df%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3ESt.%20James%20Town%2C%20Cabbagetown%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_05c3ec53d88f403c8ea9b9c5b7b24a58.setContent%28html_887dea9740a645488cc024370cb034df%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_41d7c801d46c4d898ed25f13993422e1.bindPopup%28popup_05c3ec53d88f403c8ea9b9c5b7b24a58%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_2f6edf842f6e48bd907b41a4d4fb6baa%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6492%2C%20-79.3823%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_dd495f44065c47d88c53cc629f6400d7%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_d52cd56b4ec44b6ba7e2781185bafcfe%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_d52cd56b4ec44b6ba7e2781185bafcfe%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EFirst%20Canadian%20Place%2C%20Underground%20city%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_dd495f44065c47d88c53cc629f6400d7.setContent%28html_d52cd56b4ec44b6ba7e2781185bafcfe%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_2f6edf842f6e48bd907b41a4d4fb6baa.bindPopup%28popup_dd495f44065c47d88c53cc629f6400d7%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_02d4ef53e41e4cf697ec92cd781db688%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.6656%2C%20-79.383%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_3ea1624275a54e82a2126a6169a506e0%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_b6cff8b248094544873c6304514009c5%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_b6cff8b248094544873c6304514009c5%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EChurch%20and%20Wellesley%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_3ea1624275a54e82a2126a6169a506e0.setContent%28html_b6cff8b248094544873c6304514009c5%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_02d4ef53e41e4cf697ec92cd781db688.bindPopup%28popup_3ea1624275a54e82a2126a6169a506e0%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20circle_marker_60f7f74c800d4a72a3b0c21156af3e6f%20%3D%20L.circleMarker%28%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%5B43.7804%2C%20-79.2505%5D%2C%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%7B%22bubblingMouseEvents%22%3A%20true%2C%20%22color%22%3A%20%22%23ff0000%22%2C%20%22dashArray%22%3A%20null%2C%20%22dashOffset%22%3A%20null%2C%20%22fill%22%3A%20true%2C%20%22fillColor%22%3A%20%22%23ff0000%22%2C%20%22fillOpacity%22%3A%200.7%2C%20%22fillRule%22%3A%20%22evenodd%22%2C%20%22lineCap%22%3A%20%22round%22%2C%20%22lineJoin%22%3A%20%22round%22%2C%20%22opacity%22%3A%201.0%2C%20%22radius%22%3A%205%2C%20%22stroke%22%3A%20true%2C%20%22weight%22%3A%203%7D%0A%20%20%20%20%20%20%20%20%20%20%20%20%29.addTo%28map_fb9a021fdb164ead8f3640d4ceb465de%29%3B%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20var%20popup_03c68aacc5cb4b88b8a40e1e3bb8ef2e%20%3D%20L.popup%28%7B%22maxWidth%22%3A%20%22100%25%22%7D%29%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20var%20html_e8f4f77c50d645d7a6042a1cd97cc10b%20%3D%20%24%28%60%3Cdiv%20id%3D%22html_e8f4f77c50d645d7a6042a1cd97cc10b%22%20style%3D%22width%3A%20100.0%25%3B%20height%3A%20100.0%25%3B%22%3EBusiness%20reply%20mail%20Processing%20Centre%2C%20South%20Central%20Letter%20Processing%20Plant%20Toronto%20Cluster%200%3C/div%3E%60%29%5B0%5D%3B%0A%20%20%20%20%20%20%20%20%20%20%20%20popup_03c68aacc5cb4b88b8a40e1e3bb8ef2e.setContent%28html_e8f4f77c50d645d7a6042a1cd97cc10b%29%3B%0A%20%20%20%20%20%20%20%20%0A%0A%20%20%20%20%20%20%20%20circle_marker_60f7f74c800d4a72a3b0c21156af3e6f.bindPopup%28popup_03c68aacc5cb4b88b8a40e1e3bb8ef2e%29%0A%20%20%20%20%20%20%20%20%3B%0A%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%0A%3C/script%3E onload="this.contentDocument.open();this.contentDocument.write(    decodeURIComponent(this.getAttribute('data-html')));this.contentDocument.close();" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### Each cluster is examined to determine the discriminating venue categories. Based on the defining categories, we will assign a name to each cluster.

Cluster 1 = Neighborhoods with High Numbers of Dining Venues (Color=Red)


```python
df_merged.loc[df_merged['Cluster Labels'] == 0, df_merged.columns[[2] + list(range(5, df_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Regent Park, Harbourfront</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Breakfast Spot</td>
      <td>Yoga Studio</td>
      <td>Theater</td>
      <td>Distribution Center</td>
      <td>Pub</td>
      <td>Restaurant</td>
      <td>Electronics Store</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Queen's Park, Ontario Provincial Government</td>
      <td>0</td>
      <td>Gym</td>
      <td>Sushi Restaurant</td>
      <td>Coffee Shop</td>
      <td>Ethiopian Restaurant</td>
      <td>Bubble Tea Shop</td>
      <td>Café</td>
      <td>Martial Arts School</td>
      <td>Mexican Restaurant</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Garden District, Ryerson</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Clothing Store</td>
      <td>Cosmetics Shop</td>
      <td>Italian Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Middle Eastern Restaurant</td>
      <td>Café</td>
      <td>Movie Theater</td>
    </tr>
    <tr>
      <th>3</th>
      <td>St. James Town</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Seafood Restaurant</td>
      <td>Cocktail Bar</td>
      <td>Bakery</td>
      <td>American Restaurant</td>
      <td>Gastropub</td>
      <td>Italian Restaurant</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Beaches</td>
      <td>0</td>
      <td>Pub</td>
      <td>Gastropub</td>
      <td>Health Food Store</td>
      <td>Bakery</td>
      <td>Trail</td>
      <td>Donut Shop</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Berczy Park</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Hotel</td>
      <td>Café</td>
      <td>Seafood Restaurant</td>
      <td>Bakery</td>
      <td>Cocktail Bar</td>
      <td>Restaurant</td>
      <td>Japanese Restaurant</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Central Bay Street</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Middle Eastern Restaurant</td>
      <td>Sandwich Place</td>
      <td>Bubble Tea Shop</td>
      <td>Café</td>
      <td>Clothing Store</td>
      <td>Italian Restaurant</td>
      <td>Restaurant</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Christie</td>
      <td>0</td>
      <td>Café</td>
      <td>Grocery Store</td>
      <td>Playground</td>
      <td>Park</td>
      <td>Candy Store</td>
      <td>Baby Store</td>
      <td>Coffee Shop</td>
      <td>Yoga Studio</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Richmond, Adelaide, King</td>
      <td>0</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Gym</td>
      <td>Hotel</td>
      <td>Thai Restaurant</td>
      <td>American Restaurant</td>
      <td>Steakhouse</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Dufferin, Dovercourt Village</td>
      <td>0</td>
      <td>Bakery</td>
      <td>Grocery Store</td>
      <td>Park</td>
      <td>Bank</td>
      <td>Furniture / Home Store</td>
      <td>Middle Eastern Restaurant</td>
      <td>Pool</td>
      <td>Café</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Harbourfront East, Union Station, Toronto Islands</td>
      <td>0</td>
      <td>Music Venue</td>
      <td>Harbor / Marina</td>
      <td>Café</td>
      <td>Park</td>
      <td>Donut Shop</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Little Portugal, Trinity</td>
      <td>0</td>
      <td>Bar</td>
      <td>Restaurant</td>
      <td>Cocktail Bar</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Asian Restaurant</td>
      <td>Yoga Studio</td>
      <td>Cupcake Shop</td>
      <td>Seafood Restaurant</td>
    </tr>
    <tr>
      <th>12</th>
      <td>The Danforth West, Riverdale</td>
      <td>0</td>
      <td>Greek Restaurant</td>
      <td>Ice Cream Shop</td>
      <td>Café</td>
      <td>Italian Restaurant</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Yoga Studio</td>
      <td>Dessert Shop</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Toronto Dominion Centre, Design Exchange</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Hotel</td>
      <td>Café</td>
      <td>Japanese Restaurant</td>
      <td>American Restaurant</td>
      <td>Salad Place</td>
      <td>Seafood Restaurant</td>
      <td>Breakfast Spot</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Brockton, Parkdale Village, Exhibition Place</td>
      <td>0</td>
      <td>Café</td>
      <td>Coffee Shop</td>
      <td>Breakfast Spot</td>
      <td>Gift Shop</td>
      <td>Thrift / Vintage Store</td>
      <td>Pet Store</td>
      <td>Brewery</td>
      <td>Sandwich Place</td>
    </tr>
    <tr>
      <th>15</th>
      <td>India Bazaar, The Beaches West</td>
      <td>0</td>
      <td>Fast Food Restaurant</td>
      <td>Sandwich Place</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Ice Cream Shop</td>
      <td>Liquor Store</td>
      <td>Brewery</td>
      <td>Food &amp; Drink Shop</td>
      <td>Steakhouse</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Commerce Court, Victoria Hotel</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Hotel</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Gym</td>
      <td>American Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Asian Restaurant</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Studio District</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Coworking Space</td>
      <td>Park</td>
      <td>Gym</td>
      <td>Diner</td>
      <td>Garden Center</td>
      <td>Baseball Field</td>
      <td>Yoga Studio</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Davisville North</td>
      <td>0</td>
      <td>Breakfast Spot</td>
      <td>Department Store</td>
      <td>Gym / Fitness Center</td>
      <td>Park</td>
      <td>Dog Run</td>
      <td>Food &amp; Drink Shop</td>
      <td>Yoga Studio</td>
      <td>Eastern European Restaurant</td>
    </tr>
    <tr>
      <th>24</th>
      <td>The Annex, North Midtown, Yorkville</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Sandwich Place</td>
      <td>Café</td>
      <td>Mexican Restaurant</td>
      <td>Pub</td>
      <td>Donut Shop</td>
      <td>Italian Restaurant</td>
      <td>Furniture / Home Store</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Parkdale, Roncesvalles</td>
      <td>0</td>
      <td>Bakery</td>
      <td>Sushi Restaurant</td>
      <td>Coffee Shop</td>
      <td>Eastern European Restaurant</td>
      <td>Food &amp; Drink Shop</td>
      <td>Café</td>
      <td>Breakfast Spot</td>
      <td>Bookstore</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Davisville</td>
      <td>0</td>
      <td>Café</td>
      <td>Sandwich Place</td>
      <td>Coffee Shop</td>
      <td>Italian Restaurant</td>
      <td>Dessert Shop</td>
      <td>Thai Restaurant</td>
      <td>Gas Station</td>
      <td>Indian Restaurant</td>
    </tr>
    <tr>
      <th>27</th>
      <td>University of Toronto, Harbord</td>
      <td>0</td>
      <td>Café</td>
      <td>Bookstore</td>
      <td>Japanese Restaurant</td>
      <td>Bakery</td>
      <td>Yoga Studio</td>
      <td>Bar</td>
      <td>Noodle House</td>
      <td>Comfort Food Restaurant</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Runnymede, Swansea</td>
      <td>0</td>
      <td>Café</td>
      <td>Pizza Place</td>
      <td>Bank</td>
      <td>Restaurant</td>
      <td>Pub</td>
      <td>Sushi Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Kensington Market, Chinatown, Grange Park</td>
      <td>0</td>
      <td>Café</td>
      <td>Vegetarian / Vegan Restaurant</td>
      <td>Mexican Restaurant</td>
      <td>Coffee Shop</td>
      <td>Farmers Market</td>
      <td>Gaming Cafe</td>
      <td>Caribbean Restaurant</td>
      <td>Grocery Store</td>
    </tr>
    <tr>
      <th>32</th>
      <td>CN Tower, King and Spadina, Railway Lands, Har...</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Italian Restaurant</td>
      <td>Gym / Fitness Center</td>
      <td>Bar</td>
      <td>Speakeasy</td>
      <td>Restaurant</td>
      <td>Bank</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Stn A PO Boxes</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Restaurant</td>
      <td>Gym</td>
      <td>Hotel</td>
      <td>Japanese Restaurant</td>
      <td>Deli / Bodega</td>
      <td>Italian Restaurant</td>
      <td>Sporting Goods Shop</td>
    </tr>
    <tr>
      <th>35</th>
      <td>St. James Town, Cabbagetown</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Bakery</td>
      <td>Pizza Place</td>
      <td>Restaurant</td>
      <td>Italian Restaurant</td>
      <td>Jewelry Store</td>
      <td>Snack Place</td>
    </tr>
    <tr>
      <th>36</th>
      <td>First Canadian Place, Underground city</td>
      <td>0</td>
      <td>Coffee Shop</td>
      <td>Hotel</td>
      <td>Café</td>
      <td>Restaurant</td>
      <td>Gym</td>
      <td>American Restaurant</td>
      <td>Japanese Restaurant</td>
      <td>Asian Restaurant</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Church and Wellesley</td>
      <td>0</td>
      <td>Japanese Restaurant</td>
      <td>Coffee Shop</td>
      <td>Sushi Restaurant</td>
      <td>Gay Bar</td>
      <td>Restaurant</td>
      <td>Hotel</td>
      <td>Fast Food Restaurant</td>
      <td>Mediterranean Restaurant</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Business reply mail Processing Centre, South C...</td>
      <td>0</td>
      <td>Restaurant</td>
      <td>Coffee Shop</td>
      <td>Yoga Studio</td>
      <td>Bank</td>
      <td>Breakfast Spot</td>
      <td>Bookstore</td>
      <td>Sushi Restaurant</td>
      <td>Japanese Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



Cluster 2 = Neighborhoods with High Numbers of Outdoor Venues for Exercising (Color=Purple)


```python
df_merged.loc[df_merged['Cluster Labels'] == 1, df_merged.columns[[2] + list(range(5, df_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>High Park, The Junction South</td>
      <td>1</td>
      <td>Park</td>
      <td>Dive Bar</td>
      <td>Yoga Studio</td>
      <td>Donut Shop</td>
      <td>Fish Market</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
    </tr>
  </tbody>
</table>
</div>



Cluster 3 = Neighborhoods with High Numbers of Beauty and Health Focused Venues (Color=Cyan)


```python
df_merged.loc[df_merged['Cluster Labels'] == 2, df_merged.columns[[2] + list(range(5, df_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>31</th>
      <td>Summerhill West, Rathnelly, South Hill, Forest...</td>
      <td>2</td>
      <td>Coffee Shop</td>
      <td>Light Rail Station</td>
      <td>Liquor Store</td>
      <td>Supermarket</td>
      <td>Food &amp; Drink Shop</td>
      <td>Fish Market</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Fast Food Restaurant</td>
    </tr>
  </tbody>
</table>
</div>



Cluster 4 = Neighborhoods with High Numbers of Photography Studios (Color=Green)


```python
df_merged.loc[df_merged['Cluster Labels'] == 3, df_merged.columns[[2] + list(range(5, df_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19</th>
      <td>Roselawn</td>
      <td>3</td>
      <td>Home Service</td>
      <td>Fast Food Restaurant</td>
      <td>Yoga Studio</td>
      <td>Donut Shop</td>
      <td>Flower Shop</td>
      <td>Fish Market</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Farmers Market</td>
    </tr>
  </tbody>
</table>
</div>



Cluster 5 = Neighborhoods with High Numbers of Playground(Color=Light Brown)


```python
df_merged.loc[df_merged['Cluster Labels'] == 4, df_merged.columns[[2] + list(range(5, df_merged.shape[1]))]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Neighborhood</th>
      <th>Cluster Labels</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18</th>
      <td>Lawrence Park</td>
      <td>4</td>
      <td>Photography Studio</td>
      <td>Park</td>
      <td>Yoga Studio</td>
      <td>Doner Restaurant</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Falafel Restaurant</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Forest Hill North &amp; West, Forest Hill Road Park</td>
      <td>4</td>
      <td>French Restaurant</td>
      <td>Park</td>
      <td>Lawyer</td>
      <td>Trail</td>
      <td>Yoga Studio</td>
      <td>Donut Shop</td>
      <td>Fish &amp; Chips Shop</td>
      <td>Fast Food Restaurant</td>
    </tr>
    <tr>
      <th>23</th>
      <td>North Toronto West,  Lawrence Park</td>
      <td>4</td>
      <td>Playground</td>
      <td>Gym Pool</td>
      <td>Park</td>
      <td>Garden</td>
      <td>Doner Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
      <td>Falafel Restaurant</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Moore Park, Summerhill East</td>
      <td>4</td>
      <td>Park</td>
      <td>Thai Restaurant</td>
      <td>Trail</td>
      <td>Gym</td>
      <td>Grocery Store</td>
      <td>Donut Shop</td>
      <td>Dumpling Restaurant</td>
      <td>Eastern European Restaurant</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Rosedale</td>
      <td>4</td>
      <td>Playground</td>
      <td>Candy Store</td>
      <td>Grocery Store</td>
      <td>Park</td>
      <td>Yoga Studio</td>
      <td>Doner Restaurant</td>
      <td>Fast Food Restaurant</td>
      <td>Farmers Market</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```
