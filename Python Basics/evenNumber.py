def Even_number_calculate():
    n=5
    while n >0 :
        num = input("Enter a number:")
        if int(num)%2==0 :
            print(num," is a even number")
        else :
            print(num," is a odd number")
        n-=1
Even_number_calculate()