import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

class Regression:
    def __init__(self, trainData='trainRegression.csv', testData='testRegression.csv'):
        self.trainData = pd.read_csv(trainData)
        self.testData = pd.read_csv(testData)
        self.x_Train = np.array(self.trainData['X'])
        self.y_Train = np.array(self.trainData['R'])
        self.x_Test = np.array(self.testData['X'])
        self.y_Test = np.array(self.testData['R'])    
         
    def get_m_sumX_sumY(self):
        self.m = len(self.x_Train)
        self.sum_x = np.sum(self.x_Train)
        self.sum_y = np.sum(self.y_Train)
        self.sum_x_square = np.sum(np.square(self.x_Train))
        self.sum_xy = np.sum(self.x_Train * self.y_Train)
        
    def thetasMatrix(self,matrix_A,matrix_B):
        self.get_m_sumX_sumY()
        inverse_matrixA = np.linalg.inv(matrix_A)
        inMatA_dot_matB = np.dot(inverse_matrixA, matrix_B)
        return inMatA_dot_matB
    
    def x_pred(self, data):
        return np.array(data).reshape(-1, 1)
    
    def lineEquationCal(self, x,matrix_A,matrix_B):
        thetas = self.thetasMatrix(matrix_A,matrix_B)
        y_pred = 0
        for i in range(len(thetas)):
            y_pred += np.dot(np.power(x.reshape(-1, 1), i), thetas[i]) 
        return y_pred
    
    def mean_square_error(self, x, y, y_Pred):
        return (1/len(x)) * np.sum(np.square(y - y_Pred))
    
    def plotGraph(self, x, y, x_pred, y_pred):
        plt.plot(x, y, 'o')
        plt.plot(x_pred, y_pred)
        plt.show()
        
    def display_results_TrainData(self, matrix_A,matrix_B):
        y = self.lineEquationCal(self.x_Train, matrix_A,matrix_B) 
        x = self.x_pred(self.x_Train)
        mse = self.mean_square_error(self.x_Train, self.y_Train, y)
        print(f"Mean Square Error for train data : {mse}")
        self.plotGraph(self.x_Train, self.y_Train, x, y)
        
    def display_results_TestData(self, matrix_A,matrix_B):
        y = self.lineEquationCal(self.x_Test,matrix_A,matrix_B) 
        x = self.x_pred(self.x_Test)
        mse = self.mean_square_error(self.x_Test, self.y_Test, y)
        print(f"Mean Square Error for test data : {mse}")
        self.plotGraph(self.x_Test, self.y_Test, x, y)
        

class Linear(Regression):
    def __init__(self, trainData='trainRegression.csv', testData='testRegression.csv'):
        super().__init__(trainData, testData)
    def variables(self):
        self.get_m_sumX_sumY()
        
    def linearMatrix(self):
        self.variables()
        self.matrix_A = np.array([[self.m, self.sum_x], [self.sum_x, self.sum_x_square]])
        self.matrix_B = np.array([[self.sum_y], [self.sum_xy]])
        
    def showPred_test(self) :
        self.linearMatrix()
        self.display_results_TestData(self.matrix_A, self.matrix_B)
        
    def showPred_train(self) :
        self.linearMatrix()
        self.display_results_TrainData(self.matrix_A, self.matrix_B)
        
    def predicted_values_test(self):
        self.linearMatrix()
        return self.lineEquationCal(self.x_Test,self.matrix_A,self.matrix_B) 
    
    def predicted_values_train(self):
        self.linearMatrix()
        return self.lineEquationCal(self.x_Test,self.matrix_A,self.matrix_B) 

class Quadratic(Linear) :
    def __init__(self, trainData='trainRegression.csv', testData='testRegression.csv'):
        super().__init__(trainData, testData)
    def variables(self):
        super().variables()
        self.sum_x_cube = np.sum(np.power(self.x_Train, 3))
        self.sum_x_four = np.sum(np.power(self.x_Train, 4))
        self.sum_xSqy = np.sum((np.square(self.x_Train)*self.y_Train))
    def linearMatrix(self):
        self.variables()
        self.matrix_A = np.array([[self.m, self.sum_x, self.sum_x_square],
                                  [self.sum_x, self.sum_x_square, self.sum_x_cube],
                                  [self.sum_x_square, self.sum_x_cube, self.sum_x_four]])
        self.matrix_B = np.array([[self.sum_y], [self.sum_xy], [self.sum_xSqy]])
    def showPred_test(self) :
        self.linearMatrix()
        self.display_results_TestData(self.matrix_A, self.matrix_B)
    def showPred_train(self) :
        self.linearMatrix()
        self.display_results_TrainData(self.matrix_A, self.matrix_B)
        
    def predicted_values_test(self):
        self.linearMatrix()
        return self.lineEquationCal(self.x_Test,self.matrix_A,self.matrix_B) 
    
    def predicted_values_train(self):
        self.linearMatrix()
        return self.lineEquationCal(self.x_Test,self.matrix_A,self.matrix_B) 

class Cubic(Quadratic) :
    def __init__(self, trainData='trainRegression.csv', testData='testRegression.csv'):
        super().__init__(trainData, testData)
    def variables(self):
        super().variables()
        self.sum_x_five = np.sum(np.power(self.x_Train, 5))
        self.sum_x_six = np.sum(np.power(self.x_Train, 6))
        self.sum_xCub = np.sum((np.power(self.x_Train,3)*self.y_Train))
    def linearMatrix(self):
        self.variables()
        self.matrix_A = np.array([[self.m, self.sum_x, self.sum_x_square, self.sum_x_cube],
                                        [self.sum_x, self.sum_x_square, self.sum_x_cube, self.sum_x_four],
                                        [self.sum_x_square, self.sum_x_cube, self.sum_x_four, self.sum_x_five],
                                        [self.sum_x_cube, self.sum_x_four, self.sum_x_five, self.sum_x_six]])
        self.matrix_B = np.array([[self.sum_y], [self.sum_xy], [self.sum_xSqy], [self.sum_xCub]])
    def showPred_test(self) :
        self.linearMatrix()
        self.display_results_TestData(self.matrix_A, self.matrix_B)
    def showPred_train(self) :
        self.linearMatrix()
        self.display_results_TrainData(self.matrix_A, self.matrix_B)
    def predicted_values_test(self):
        self.linearMatrix()
        return self.lineEquationCal(self.x_Test,self.matrix_A,self.matrix_B) 
    
    def predicted_values_train(self):
        self.linearMatrix()
        return self.lineEquationCal(self.x_Test,self.matrix_A,self.matrix_B) 

class FourDegree(Cubic) :
    def __init__(self, trainData='trainRegression.csv', testData='testRegression.csv'):
        super().__init__(trainData, testData)
    def variables(self):
        super().variables()
        self.sum_x_seven = np.sum(np.power(self.x_Train, 7))
        self.sum_x_eight = np.sum(np.power(self.x_Train, 8))
        self.sum_xFou = np.sum((np.power(self.x_Train,4)*self.y_Train))

    def linearMatrix(self):
        self.variables()
        self.matrix_A = np.array([[self.m, self.sum_x, self.sum_x_square, self.sum_x_cube, self.sum_x_four],
                                  [self.sum_x, self.sum_x_square, self.sum_x_cube, self.sum_x_four, self.sum_x_five],
                                  [self.sum_x_square, self.sum_x_cube, self.sum_x_four, self.sum_x_five, self.sum_x_six],
                                  [self.sum_x_cube, self.sum_x_four, self.sum_x_five, self.sum_x_six, self.sum_x_seven],
                                  [self.sum_x_four, self.sum_x_five, self.sum_x_six, self.sum_x_seven, self.sum_x_eight]])
        
        self.matrix_B = np.array([[self.sum_y], [self.sum_xy], [self.sum_xSqy], [self.sum_xCub], [self.sum_xFou]])
    def showPred_test(self) :
        self.linearMatrix()
        self.display_results_TestData(self.matrix_A, self.matrix_B)
    def showPred_train(self) :
        self.linearMatrix()
        self.display_results_TrainData(self.matrix_A, self.matrix_B)
 
    def predicted_values_test(self):
        self.linearMatrix()
        return self.lineEquationCal(self.x_Test,self.matrix_A,self.matrix_B) 
    
    def predicted_values_train(self):
        self.linearMatrix()
        return self.lineEquationCal(self.x_Test,self.matrix_A,self.matrix_B)        

class FiveDegree(FourDegree) :
    def __init__(self, trainData='trainRegression.csv', testData='testRegression.csv'):
        super().__init__(trainData, testData)
    def variables(self):
        super().variables()
        self.sum_x_nine = np.sum(np.power(self.x_Train, 9))
        self.sum_x_ten = np.sum(np.power(self.x_Train, 10))
        self.sum_xFiv = np.sum((np.power(self.x_Train,5)*self.y_Train))
    def linearMatrix(self):
        self.variables()
        self.matrix_A = np.array([[self.m, self.sum_x, self.sum_x_square, self.sum_x_cube, self.sum_x_four, self.sum_x_five],
                                  [self.sum_x, self.sum_x_square, self.sum_x_cube, self.sum_x_four, self.sum_x_five, self.sum_x_six],
                                  [self.sum_x_square, self.sum_x_cube, self.sum_x_four, self.sum_x_five, self.sum_x_six, self.sum_x_seven],
                                  [self.sum_x_cube, self.sum_x_four, self.sum_x_five, self.sum_x_six, self.sum_x_seven, self.sum_x_eight],
                                  [self.sum_x_four, self.sum_x_five, self.sum_x_six, self.sum_x_seven, self.sum_x_eight, self.sum_x_nine],
                                  [self.sum_x_five, self.sum_x_six, self.sum_x_seven, self.sum_x_eight, self.sum_x_nine, self.sum_x_ten]])
        
        self.matrix_B = np.array([[self.sum_y], [self.sum_xy], [self.sum_xSqy], [self.sum_xCub], [self.sum_xFou], [self.sum_xFiv]])
    def showPred_test(self) :
        self.linearMatrix()
        self.display_results_TestData(self.matrix_A, self.matrix_B)
    def showPred_train(self) :
        self.linearMatrix()
        self.display_results_TrainData(self.matrix_A, self.matrix_B)
    def predicted_values_test(self):
        self.linearMatrix()
        return self.lineEquationCal(self.x_Test,self.matrix_A,self.matrix_B) 
    
    def predicted_values_train(self):
        self.linearMatrix()
        return self.lineEquationCal(self.x_Test,self.matrix_A,self.matrix_B)       
class SixDegree(FiveDegree) :
    def __init__(self, trainData='trainRegression.csv', testData='testRegression.csv'):
        super().__init__(trainData, testData)
    def variables(self):
        super().variables()
        self.sum_x_eleven = np.sum(np.power(self.x_Train, 11))
        self.sum_x_twelve = np.sum(np.power(self.x_Train, 12))
        self.sum_xSix = np.sum((np.power(self.x_Train,6)*self.y_Train))
    def linearMatrix(self):
        self.variables()
        self.matrix_A = np.array([[self.m, self.sum_x, self.sum_x_square, self.sum_x_cube, self.sum_x_four, self.sum_x_five, self.sum_x_six],
                                  [self.sum_x, self.sum_x_square, self.sum_x_cube, self.sum_x_four, self.sum_x_five, self.sum_x_six, self.sum_x_seven],
                                  [self.sum_x_square, self.sum_x_cube, self.sum_x_four, self.sum_x_five, self.sum_x_six, self.sum_x_seven, self.sum_x_eight],
                                  [self.sum_x_cube, self.sum_x_four, self.sum_x_five, self.sum_x_six, self.sum_x_seven, self.sum_x_eight, self.sum_x_nine],
                                  [self.sum_x_four, self.sum_x_five, self.sum_x_six, self.sum_x_seven, self.sum_x_eight, self.sum_x_nine, self.sum_x_ten],
                                  [self.sum_x_five, self.sum_x_six, self.sum_x_seven, self.sum_x_eight, self.sum_x_nine, self.sum_x_ten, self.sum_x_eleven],
                                  [self.sum_x_six, self.sum_x_seven, self.sum_x_eight, self.sum_x_nine, self.sum_x_ten, self.sum_x_eleven, self.sum_x_twelve]])
        
        self.matrix_B = np.array([[self.sum_y], [self.sum_xy], [self.sum_xSqy], [self.sum_xCub], [self.sum_xFou], [self.sum_xFiv], [self.sum_xSix]])
    def showPred_test(self) :
        self.linearMatrix()
        self.display_results_TestData(self.matrix_A, self.matrix_B)
    def showPred_train(self) :
        self.linearMatrix()
        self.display_results_TrainData(self.matrix_A, self.matrix_B)

    def predicted_values_test(self):
        self.linearMatrix()
        return self.lineEquationCal(self.x_Test,self.matrix_A,self.matrix_B) 
    
    def predicted_values_train(self):
        self.linearMatrix()
        return self.lineEquationCal(self.x_Test,self.matrix_A,self.matrix_B)        

def main():  
    trainData = input("Enter the name of the training Data csv file :\n")
    testData = input("Enter the name of the testing Data csv file :\n")
    models = {1: {'Linear Regression Model' :
                Linear(trainData, testData)}, 
              2: {'Quadractic regression model' :
                  Quadratic(trainData, testData)}, 
              3: {'Cubic regression model' :
                Cubic(trainData, testData)}, 
              4: {'Four Degree regression model' :
                FourDegree(trainData, testData)}, 
              5: {'Five Degree regression model' :
                FiveDegree(trainData, testData)}, 
              6: {'Six Degree regression model' :
                SixDegree(trainData, testData)}, 
              }
    degree = 1
    while True :
        print("Results :\n")
        model_name = list(models[degree].keys())[0]
        print(f"___________{model_name}____________\n")
        print("____________Training___________\n")
        models[degree][model_name].showPred_train()
        print("____________Testing___________\n")
        models[degree][model_name].showPred_test()
        print("Are you want to increase the degree power to decrease the mse (upto 6th degree) :\n")
        x = input("If yes ? enter 1, for stop enter 0 and get predicted vlaues:\n")
        if int(x) == 0 or degree == 7:
            print("Predicted values against training data:_____\n")
            print(models[degree][model_name].predicted_values_train(), "\n")   
            print("Predicted values against testing data:_____\n")
            print(models[degree][model_name].predicted_values_test(),"\n")    
            return
        degree+=1
if __name__ == '__main__':
    main()
