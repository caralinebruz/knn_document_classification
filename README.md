# K Nearest Neighbors
Attempts to classify an unknown document based on an existing tf_idf matrix 



## REQUIREMENTS
- Requires python-3.9.7 or higher
- Create a new `virtualenv` and then run:
`pip3 install -r requirements.txt`



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


# About my model
If there is no distinguished category that got votes (in the case of a tie or a perpendicular document vector) then I simply pick the category of the closest document. This has worked well for me.

My model compares the results with varying values of k _(I chose K=3, K=6, and K=9)_ if they all agree on the classification, then my model rocks! 

For all files (aside from `unknown/unknown08.txt` for which there is a syntax error in the raw input file) my model continues to agree on the selected category across chosen values of K.


## RESULTS
For any given value for k, my code will print the scores and evaluation of the KNN iteration model (_below is for when k=3_)
```
2022-08-03 14:35:51 INFO     The closest 3 neighbors:
2022-08-03 14:35:51 INFO     	key 0, distance: 0.843123574683881, label: Airline Safety
2022-08-03 14:35:51 INFO     	key 1, distance: 0.3305893882835494, label: Airline Safety
2022-08-03 14:35:51 INFO     	key 4, distance: 0.024315114864623615, label: Airline Safety
2022-08-03 14:35:51 INFO     After voting, results:
2022-08-03 14:35:51 INFO     {'Airline Safety': 3, 'Hoof and Mouth Disease': 0, 'Mortgage Rates': 0}
2022-08-03 14:35:51 INFO     congrats you have a highest score and it's Airline Safety
2022-08-03 14:35:51 INFO     Nearest neighbor category : Airline Safety
```

Since I run the model for three different values of K and compare them, the final lines output by my code:
```
2022-08-03 21:37:19 INFO     Final Model Evaluation:
2022-08-03 21:37:19 INFO     your model rocks! all values of K agree on the winning category: Hoof and Mouth Disease
```

## MODEL PERFORMANCE

In 9/10 test cases my model has agreed 100% of the time for values of k=3, k=6, and k=9 :)


In just one file: `unknown_10.txt` I got different results for different values of k.

Thanks!
