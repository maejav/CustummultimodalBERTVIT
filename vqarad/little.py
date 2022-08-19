import json
from sklearn.model_selection import train_test_split
# Opening JSON file
f = open('data.json')
 
# returns JSON object as
# a dictionary
data = json.load(f)
 
# Iterating through the json
# list
for i in data:
    print(i)
    break
_, alltrainvalid = train_test_split(data, test_size=0.3)
alltrain, test = train_test_split(alltrainvalid, test_size=0.3)
train, validation = train_test_split(alltrain, test_size=0.2)
with open('littestset.json', 'w') as outfile:
    json.dump(test, outfile)
  
# Using a JSON string
with open('littrainset.json', 'w') as outfile:
    json.dump(train,outfile)
    

with open('litvalidationset.json', 'w') as outfile:
    json.dump(validation, outfile)
# Closing file
f.close()