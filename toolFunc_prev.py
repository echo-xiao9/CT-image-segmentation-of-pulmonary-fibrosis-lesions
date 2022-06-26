import cv2 as cv
import numpy as np
import pickle

# read .pkl file under given path, get path[0] and path[1]
# path[0] stores all the data image. path[0][index] can get given index image
# path[1] stores all the image tags. path[1][index] can get given index tag.
# usage: data0,data1 = getSubImage('../Data/trainData1',1,19);
def getSubImage(subPath,case, size):
  path = subPath +'/trainData'+str(case)+'/trainData'+str(case)+'_'+str(size)+'.pkl';
  with open(path, 'rb') as f:
    data = pickle.load(f)
  return data[0], data[1]


def writeList(list_data):
  file=open('data.txt','w') 
  file.write(str(list_data)); 
  file.close() 

def writeNumpy(numpy_data, path):
  np.savetxt(path, numpy_data)

def arr1dTo2d(num, arr):
  _2dArr =[]
  for i in range((int)(len(arr)/num)):
    _tmp=[]
    for j in range(num):
      _tmp.append(arr[i+j])
    _2dArr.append(_tmp)
  return np.array(_2dArr)
