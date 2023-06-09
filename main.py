import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



data = pd.read_csv("city1.csv")
city = data[['城市', '经度', '纬度']]
city = np.array(city)
#path =[12,7,14,13,9,4,8,5,6,0,1,3,10,19,16,15,17,20,28,26,23,24,29,31,32,33,30,25,22,18,27,21,2,11]
#12,7,14,13,9,4,8,5,6,0,1,3,10,19,16,15,20,17,28,26,23,24,29,32,31,33,30,25,22,18,27,21,2,11
#25,18,22,27,21,2,11,12,7,14,13,9,4,8,5,6,0,1,3,10,16,15,20,17,28,26,23,19,24,29,32,31,33,30
#path =[25,18,22,27,21,2,11,12,7,14,13,9,4,8,5,6,0,1,3,10,16,15,20,17,28,26,23,19,24,29,32,31,33,30]
#12,7,14,13,9,4,8,5,6,0,1,3,10,16,15,17,20,28,26,23,19,24,29,31,32,33,30,25,22,18,27,21,2,11

#155.153
path = [12,7,14,13,9,4,8,5,6,0,1,3,10,16,15,17,20,28,26,23,19,24,29,31,32,33,30,25,22,18,27,21,2,11]

#155.64
#path=[33,30,25,22,18,27,21,2,11,12,7,14,13,9,4,8,5,6,0,1,3,10,16,15,20,17,28,26,23,19,24,29,32,31]
#33,30,25,22,18,27,21,2,11,12,7,14,13,9,4,8,5,6,0,1,3,10,16,15,20,17,28,26,23,19,24,29,32,31

#10->19->16 ->.... ->23->24
#10->skip->16 ->

def drawPath(city, path):
    #print('hi1')
    plt.figure(1, figsize=(20, 13))

    #plt.figure(1,figsize=(100,100))
    #print('hi2')
    X = []
    Y = []
    cityName = []
    for i in range(len(path)):
        X.append(city[path[i]][1])
        Y.append(city[path[i]][2])
        cityName.append(city[path[i]][0])
    X.append(city[path[0]][1])
    Y.append(city[path[0]][2])
    for i in range(len(X) - 1):
        plt.plot((X[i], X[i + 1]), (Y[i], Y[i + 1]), 'k^-')
        print(cityName[i] + '->', end='')
        plt.text(X[i], Y[i] + 0.05, str(cityName[i]), color='red')
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Route')
    plt.show()

if __name__ == '__main__':
    drawPath(city, path)