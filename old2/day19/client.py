import requests
import json
import numpy as np

host = 'localhost'
port = 5000
task = 'restdemo'
url = 'http://{:s}:{:d}/{:s}/'.format(host, port, task)


data = {"one": "100", "two": "200"} # convert ndarray data to dict of list
j_data = json.dumps(data) # convert dict data to record of json
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'} # config header of REST API
res = requests.put(url, data=j_data, headers=headers) # send request
print(res.text)

j_data = json.dumps(["one", "two", "three"])
res = requests.post(url, data=j_data, headers=headers)
print(res.text)

# requests has no delete method, hence there is no del request here

res = requests.get(url, headers=headers)
print(res.text)