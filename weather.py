import requests, json, time
from datetime import datetime
import sys

DATASET_FILE = "dataset2.txt"

def evaluate_description(description):
	if description == "Thunderstorm":
		return 0
	elif description == "Drizzle":
		return 1
	elif description == "Rain":
		return 2
	elif description == "Snow":
		return 3
	elif description == "Clear":
		return 4
	elif description == "Clouds":
		return 5
	else:
		return 6

def evaluate_temperature(temperature):
	if temperature <= 5:
		return 0
	elif temperature <= 20:
		return 1
	else:
		return 2

def evaluate_humidity(humidity):
	if humidity <= 20:
		return 0
	elif humidity <= 60:
		return 1
	else:
		return 2

def evaluate_visibility(visibility):
	if visibility <= 4000:
		return 0
	elif visibility <= 7000:
		return 1
	else:
		return 2

def evaluate_wind_speed(wind_speed):
	if wind_speed <= 6:
		return 0
	elif wind_speed <= 14:
		return 1
	else:
		return 2

def evaluate_cloudiness(cloudiness):
	if cloudiness <= 20:
		return 0
	elif cloudiness <= 70:
		return 1
	else:
		return 2

def to_dataset(desc, temp, feels, hum, vis, wind, cloud):
	file = open(DATASET_FILE, "a")
	file.write(str(desc) + "," + str(temp) + "," + str(feels) + "," + str(hum) + "," + str(vis) + "," + str(wind) + "," + str(cloud) + "\n")
	file.close()

def get_weather_report():
	api_key = "7edf84dd10acde11474dde25976d17a1"
	base_url = "http://api.openweathermap.org/data/2.5/weather?"
	
	city_name = requests.get("https://ipinfo.io/").json()["city"]
	
	complete_url = base_url + "appid=" + api_key + "&q=" + city_name + "&units=metric"
	
	response = requests.get(complete_url)
	
	x = response.json()
	
	if x["cod"] != "404":
		description = str(x["weather"][0]["main"])
		temperature = str(int(x["main"]["temp"]))
		feels_like = str(int(x["main"]["feels_like"]))
		
		print(description + ", Temperature: " + temperature + "°C, Feels like: " + feels_like + "°C")
	else:
		print("Couldn't get data")

def get_weather():
	api_key = "7edf84dd10acde11474dde25976d17a1"
	base_url = "http://api.openweathermap.org/data/2.5/weather?"
	
	city_name = requests.get("https://ipinfo.io/").json()["city"]
	
	complete_url = base_url + "appid=" + api_key + "&q=" + city_name + "&units=metric"
	
	response = requests.get(complete_url)
	
	x = response.json()
	
	if x["cod"] != "404":
		description = str(x["weather"][0]["main"])
		temperature = int(x["main"]["temp"])
		feels_like = int(x["main"]["feels_like"])
		humidity = x["main"]["humidity"]
		visibility = x["visibility"]
		wind_speed = x["wind"]["speed"]
		cloudiness = x["clouds"]["all"]
		
		evaluated_description = evaluate_description(description)
		evaluated_temperature = evaluate_temperature(temperature)
		evaluated_feels_like = evaluate_temperature(feels_like)
		evaluated_humidity = evaluate_humidity(humidity)
		evaluated_visibility = evaluate_visibility(visibility)
		evaluated_wind_speed = evaluate_wind_speed(wind_speed)
		evaluated_cloudiness = evaluate_cloudiness(cloudiness)
		
		return [evaluated_description, evaluated_temperature, evaluated_feels_like, evaluated_humidity, evaluated_visibility, evaluated_wind_speed, evaluated_cloudiness]
	else:
		print("Couldn't get data")