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
    
    
    #Calculating the values for the sixDegree model
    m = len(x_Train)
    sum_x = np.sum(x_Train)
    sum_y =np.sum(y_Train)
    sum_x_square = np.sum(np.square(x_Train))
    sum_x_cube = np.sum(np.power(x_Train, 3))
    sum_x_four = np.sum(np.power(x_Train, 4))
    sum_x_five = np.sum(np.power(x_Train, 5))
    sum_x_six = np.sum(np.power(x_Train, 6))
    sum_x_seven = np.sum(np.power(x_Train, 7))
    sum_x_eight = np.sum(np.power(x_Train, 8))
    sum_x_nine = np.sum(np.power(x_Train, 9))
    sum_x_ten = np.sum(np.power(x_Train, 10))
    sum_x_eleven = np.sum(np.power(x_Train, 11))
    sum_x_twelve = np.sum(np.power(x_Train, 12))
    
    sum_xy = np.sum(x_Train*y_Train)
    sum_xSqy = np.sum((np.square(x_Train)*y_Train))
    sum_xCub = np.sum((np.power(x_Train,3)*y_Train))
    sum_xFou = np.sum((np.power(x_Train,4)*y_Train))
    sum_xFiv = np.sum((np.power(x_Train,5)*y_Train))
    sum_xSix = np.sum((np.power(x_Train,6)*y_Train))
    
    #Making the matrices
    matrix_A = np.array([[m,sum_x,sum_x_square,sum_x_cube,sum_x_four,sum_x_five,sum_x_six],
                        [sum_x,sum_x_square, sum_x_cube,sum_x_four,sum_x_five,sum_x_six,sum_x_seven],
                        [sum_x_square, sum_x_cube, sum_x_four,sum_x_five,sum_x_six,sum_x_seven,sum_x_eight],
                        [sum_x_cube, sum_x_four,sum_x_five,sum_x_six,sum_x_seven,sum_x_eight,sum_x_nine],
                        [sum_x_four, sum_x_five,sum_x_six,sum_x_seven,sum_x_eight,sum_x_nine,sum_x_ten],
                        [sum_x_five, sum_x_six,sum_x_seven,sum_x_eight,sum_x_nine,sum_x_ten,sum_x_eleven],
                        [sum_x_six, sum_x_seven,sum_x_eight,sum_x_nine,sum_x_ten,sum_x_eleven,sum_x_twelve]
                        ])
    matrix_B = np.array([[sum_y],
                        [sum_xy],
                        [sum_xSqy],
                        [sum_xCub],
                        [sum_xFou],
                        [sum_xFiv],
                        [sum_xSix]
                        ])
    print(matrix_A)
    print(matrix_B)
    
    #Calculating the inverse of matrix A and then multiplying with the B
    inverse_matrixA = np.linalg.inv(matrix_A)
    inMatA_dot_matB = np.dot(inverse_matrixA, matrix_B)
    print(inMatA_dot_matB)

    #Calculating the  y predictions for the test data   
    x_TestPred = np.array(x_Test).reshape(-1, 1) # To set rows automatically and make 1 column of 2D
    y_TestPred = inMatA_dot_matB[0] + np.dot(x_TestPred, inMatA_dot_matB[1]) + np.dot(np.square(x_TestPred), inMatA_dot_matB[2]) + np.dot(np.power(x_TestPred,3), inMatA_dot_matB[3])+ np.dot(np.power(x_TestPred,4), inMatA_dot_matB[4])+ np.dot(np.power(x_TestPred,5), inMatA_dot_matB[5])+ np.dot(np.power(x_TestPred,6), inMatA_dot_matB[6])
    
    # Calculating the  y predictions for the train data   
    x_TrainPred = np.array(x_Train).reshape(-1, 1) # To set rows automatically and make 1 column of 2D
    y_TrainPred = inMatA_dot_matB[0] + np.dot(x_TrainPred, inMatA_dot_matB[1])+np.dot(np.square(x_TrainPred), inMatA_dot_matB[2])+np.dot(np.power(x_TrainPred,3), inMatA_dot_matB[3])+np.dot(np.power(x_TrainPred,4), inMatA_dot_matB[4])+np.dot(np.power(x_TrainPred,5), inMatA_dot_matB[5])+np.dot(np.power(x_TrainPred,6), inMatA_dot_matB[6])
    
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