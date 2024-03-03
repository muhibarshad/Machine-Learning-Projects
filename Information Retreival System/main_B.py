
import numpy as np
import os

def main():
    path = "./files"
    files_dict = {}  
    file_count = 0 
    unique_word_set = set()
    files_content=list()
    
    #Getting name of files in a folder and count 
    for root, _, files in os.walk(path) :
        for key, file in enumerate(files) :
            file_path = os.path.join(root, file)
            files_dict[file_path] = key
            file_count+=1
    # print("\nTotal Number  of files\n", file_count)
    # print("\nDictionary containing  files\n", files_dict)
    
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
    unique_word_dict={}
    for key, word in enumerate(unique_word_set) :
        unique_word_dict[word]=key
    tdm = np.zeros((file_count,len(unique_word_set)), dtype=int)
    # print(tdm)
    for key, i in files_dict.items() :
        with open(key, 'r') as f:
            l = f.read().lower()
        for word, j in unique_word_dict.items():
            if word in l:
                tdm[i][j]+=1
    # print(tdm)
    
    # making the colvector for user query 
    colVector = np.zeros((len(unique_word_dict),1), dtype=int)
    query = input("\nWrite something for searching:\n")
    uniqueWordsList = list(unique_word_dict.keys())
    # print(uniqueWordsList)
    
    # Making the colvector of relatble user query frequencus of words
    for word in query.lower().split() :
        if word in uniqueWordsList :
            index = uniqueWordsList.index(word)
            colVector[index]+=1
    # print(colVector)
    
    '''Getting the highest frequency user query word and its index and then the word and then find the
    relatble files conating that highest frequency word'''
    maximumFreq = np.max(colVector)
    if maximumFreq != 0 :
        indexMax= np.where(colVector==maximumFreq)
        final = uniqueWordsList[indexMax[0][0]]
        filesName = list(files_dict.keys())
        for key, file in enumerate(files_content):
            if final in file :
                print(f"File name : {filesName[key]}\n Content : {file}")
    else:
        print("Not found in any of the document")           
            
    
if __name__ =='__main__' :
    main()