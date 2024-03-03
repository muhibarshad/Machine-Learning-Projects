try:
    def Password_checker(username,password) :
        username=username.lower()
        if username == 'aliabdullah' and password == '123' :
            print("Logged in")
            return True
        elif username != 'aliabdullah' and password == '123' :
            print("username is incorrect")
            return False
        elif username == 'aliabdullah' and password != '123' :
            print("Password is incorrect")
            return False
        else :
            print("Both username and password is incorrect")
            return False
    while True :
        username = input("Enter username : ")
        password = input("Enter password : ")
        if Password_checker(username, password) :
            break 
except:
    print("Wrong input")

