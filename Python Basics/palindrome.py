def Palindrome(str):
    start=0
    end=len(str)-1
    flag=False
    for ch in str :
        if flag == False :
            if str[start] != str[end] :
                flag=True
        start+=1
        end-=1
    if flag==False :
        print(str, "is a Palindrome")
    else :
        print(str, "is not a palindrome")
    
str = input("Enter a string:\n")
Palindrome(str)
