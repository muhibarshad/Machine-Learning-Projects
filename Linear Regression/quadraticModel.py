import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 



def main() :
    #Training and Testing Data
    trainData = pd.read_csv('trainRegression.csv')
    TrD = trainData.head() 
    testData = pd.read_csv('testRegression.csv')
    
    x_Train = np.array(trainData['X'])
    y_Train = np.array(trainData['R'])
    x_Test = np.array(testData['X'])
    y_Test = np.array(testData['R'])
    
    
    #Calculating the values for the quadratic model
    m = len(x_Train)
    sum_x = np.sum(x_Train)
    sum_y =np.sum(y_Train)
    sum_x_square = np.sum(np.square(x_Train))
    sum_x_cube = np.sum(np.power(x_Train, 3))
    sum_x_four = np.sum(np.power(x_Train, 4))
    sum_xy = np.sum(x_Train*y_Train)
    sum_xSqy = np.sum((np.square(x_Train)*y_Train))
    
    #Making the matrices
    matrix_A = np.array([[m,sum_x,sum_x_square],
                        [sum_x,sum_x_square, sum_x_cube],
                        [sum_x_square, sum_x_cube, sum_x_four]
                        ])
    matrix_B = np.array([[sum_y],[sum_xy],[sum_xSqy]])
    print(matrix_A)
    print(matrix_B)
    
    #Calculating the inverse of matrix A and then multiplying with the B
    inverse_matrixA = np.linalg.inv(matrix_A)
    inMatA_dot_matB = np.dot(inverse_matrixA, matrix_B)
    print(inMatA_dot_matB)

    #Calculating the  y predictions for the test data   
    x_TestPred = np.array(x_Test).reshape(-1, 1) # To set rows automatically and make 1 column of 2D
    y_TestPred = inMatA_dot_matB[0] + np.dot(x_TestPred, inMatA_dot_matB[1]) + np.dot(np.square(x_TestPred), inMatA_dot_matB[2])
    
    # Calculating the  y predictions for the train data   
    x_TrainPred = np.array(x_Train).reshape(-1, 1) # To set rows automatically and make 1 column of 2D
    y_TrainPred = inMatA_dot_matB[0] + np.dot(x_TrainPred, inMatA_dot_matB[1])+np.dot(np.square(x_TrainPred), inMatA_dot_matB[2])
    
    #Calculating the mean square error for the test_x
    mse_Train = (1/len(x_Train))*np.sum(np.square(y_Train-y_TrainPred))
    mse_Test = (1/len(x_Test))*np.sum(np.square(y_Test-y_TestPred))
    print(mse_Train)  
    print(mse_Test)  
    
    # plotting the graph for training data and predicted data on training 
    plt.plot(x_Train,y_Train,'o') # for getting dots 
    plt.plot(x_TrainPred,y_TrainPred)
    plt.show()
    
    # plotting the graph for testing data and predicted data on testing 
    plt.plot(x_Test,y_Test,'o') # for getting dots 
    plt.plot(x_TestPred,y_TestPred)
    plt.show()
    
if __name__ =='__main__' :
    main()