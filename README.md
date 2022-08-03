# K Nearest Neighbors
Attempts to classify an unknown document based on an existing tf_idf matrix 


<br/>

## USAGE
example usage:

`python3 knn.py data/unknown/unknown01.txt data/input_tf_idf_labels_headers.csv`

## About the inputs
1. Expects a text file as 1st argument of unknown class
2. Expects a csv file as 2nd argument containing a pre-created tf_idf matrix

_args can use either fullpath or relative path (both will work)_ 

<br/>

#### tf_idf file requirements:
- Must be a .CSV file 
- First column of the file must contain indices for each row (ie. 0,1,2...)
- Second column of the file must contain labels for each row (ie. Airline Safety, Hoof Mouth Disease, etc.)
- Must contain a header record indicating the names of the keywords or concepts (ie. `index`, `label`, `feature_1`, `feature_2`, ....)

_see below for example of expected format:_


<br/>


| Index  | Label | columbian airline  | civil aviation | flight safety  | hoof mouth |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| 0  | Airline Safety  | 0.01423 | 0.00 | 0.00 | 0.00202 |
| 1  | Airline Safety  | 0.00 | 0.00202 | 0.01423 | 0.00 |
| 2  | Hoof and Mouth Disease  | 0.01423 | 0.00 | 0.00202 | 0.00 |
| 3  | Hoof and Mouth Disease  | 0.00 | 0.00 | 0.00 | 0.00202 |
| 4  | Hoof and Mouth Disease  | 0.00 | 0.00 | 0.00202 | 0.01423 |
