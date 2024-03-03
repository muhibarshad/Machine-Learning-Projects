
import numpy as np
import fnmatch
import os


def main():
    
    path = input("Enter the Files path like : ./files/ :\n")
    pattern=input("Enter the pattern of files , like : *.txt :\n")
    files_dict = {}  
    file_count = 0 
    unique_word_set = set()
    files_content=list()
    
    #Getting name of files in a folder and count 
    for root, _, files in os.walk(path) :
        for key, file in enumerate(files) :
            if fnmatch.fnmatch(file, pattern) :
                files_dict[os.path.join(root, file)] = key
                file_count+=1
    
    # getting unique words in the files and storing in a set
    with open('all_data.txt', 'a') as all_data :
        for key, _ in files_dict.items() :
            with open(key, 'r') as f:
                text = f.read()
                files_content.append(text)
            all_data.write(text+" ")
    with open('all_data.txt', 'r') as f :
        text = f.read()
    line = text.lower().split()
    os.remove('all_data.txt')
    for word in line :
        unique_word_set.add(word)
        
    # Creating the TDM using bagOfWord approach
    tdm = np.zeros((file_count,len( unique_word_set)), dtype=int)
    # print(tdm)
    for key, i in files_dict.items() :
        with open(key, 'r') as f:
            l = f.read().lower()
            words = l.split()
        for j, word in enumerate(unique_word_set):
            if word in words:
                tdm[i][j]+=1
    # print(tdm)
    
    # making the colvector for user query 
    colVector = np.zeros((len(unique_word_set),1), dtype=int)
    query = input("\nWrite something for searching:\n")
    
    # Making the col vector TDM  as user query
    queryWords = query.split()
    for key, value in enumerate(unique_word_set):
        if value in queryWords :
            colVector[key]+=1
            
    # Calculatying the dot product of colvector and queryVector to calculate the simlarties between them 
    resultantVector  = np.dot(tdm, colVector)
    max_index = np.argmax(resultantVector)
    max_value = resultantVector[max_index]
    if max_value == 0 :
        print("Such words not found in any of the files in a directory ", path, "With pattern", pattern," !")
        return
    #Printing the content of the file conating such words 
    max_file_name = list(files_dict.keys())[max_index]
    with open(max_file_name, 'r') as f :
        text = f.read()
        print(f"File name : {max_file_name}\nContent : {text}")   
    
if __name__ =='__main__' :
    main()