import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 



def main() :
    #Training and Testing Data
    trainData = pd.read_csv('trainRegression.csv')
    TrD = trainData.head() # returns by default first 5 rows of dataSet
    testData = pd.read_csv('trainRegression.csv')
    
    # print(TrD)
    x_Train = np.array(trainData['X'])
    y_Train = np.array(trainData['R'])
    x_Test = np.array(testData['X'])
    y_Test = np.array(testData['R'])
    
    print(x_Test)
    # print(x_Train)
    # print(y_Train)
    plt.plot(x_Train,y_Train)
    # plt.show()
    
    #Calculating the values for the linear model
    m = len(x_Train)
    sum_x = np.sum(x_Train)
    sum_y =np.sum(y_Train)
    sum_x_square = np.sum(np.square(x_Train))
    sum_xy = np.sum(x_Train*y_Train)
    
    #Making the matrices
    matrix_A = np.array([[m,sum_x],[sum_x,sum_x_square]])
    matrix_B = np.array([[sum_y],[sum_xy]])
    # print(matrix_A)
    # print(matrix_B)
    
    #Calculating the inverse of matrix A and then multiplying with the B
    inverse_matrixA = np.linalg.inv(matrix_A)
    inMatA_dot_matB = np.dot(inverse_matrixA, matrix_B)
    # print(inverse_matrixA_dot_matrix_B)

    # Calculating the  y predictions for the test data   
    # x_Test2D = np.array(x_Test).reshape(-1, 1) # To set rows automatically and make 1 column of 2D
    x_Test2D = np.array(x_Test) # To set rows automatically and make 1 column of 2D
    print(x_Test2D)
    y_pred = inMatA_dot_matB[0] +(inMatA_dot_matB[1]*x_Test2D) 
    
    #Calculating the mean square error for the test_x
    # mse_Test = (1/2*len(x_Test))*np.sum(np.square(y_Test-y_pred))
    # print(mse_Test)  

if __name__ =='__main__' :
    main()