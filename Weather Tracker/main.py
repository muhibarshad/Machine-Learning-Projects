import requests as rq
import json

def main() :
    city = input("Enter the name of city : \n")
    API_KEY_= "b8cda7d309a14d33925155103240103" # ends on 14 MARCH
    url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY_}&q={city}"
    str = rq.get(url)
    dict = json.loads(str.text)
    temp = dict['current']['temp_c']
    print(f"The temperature of {city} = {temp} Celsius")   
    
if __name__ == '__main__' :
    main()