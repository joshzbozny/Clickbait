import Decider
import requests
from urllib import request, response, error, parse
from urllib.request import urlopen
from bs4 import BeautifulSoup

from selenium import webdriver
import time

def highlight(element):
 """Highlights (blinks) a Selenium Webdriver element"""
 driver = element._parent
 def apply_style(s):
   driver.execute_script("arguments[0].setAttribute('style', arguments[1]);",element, s)
 original_style = element.get_attribute('style')
 apply_style("background: yellow; border: 2px solid red;")
 time.sleep(.3)
 # apply_style(original_style)

# driver = webdriver.Chrome()
# choose = Decider
# url = "https://www.foxnews.com"
# driver.get(url)
# if(url== "https://www.foxnews.com"):
#     articles = driver.find_elements_by_class_name("info-header")
# else:
#     articles = driver.find_elements_by_class_name("cd__headline-text")
# for article in articles:
#     title = article.text
#     print(title)
#     if(choose.PredictClass(title)):
#         print("Clickbait")
#         highlight(article)
#     else:
#         print("Non-Clickbait")


def highlightWebsite(url):
    driver = webdriver.Chrome()
    choose = Decider
    if url.find("www.") == -1:
        url = "www." + url
    if url.find("https://") == -1:
        url = "https://" + url
    driver.get(url)
    if(url== "https://www.foxnews.com/"):
        articles = driver.find_elements_by_class_name("info-header")
    else:
        articles = driver.find_elements_by_class_name("cd__headline-text")
    for article in articles:
        title = article.text
        print(title)
        if(choose.PredictClass(title)):
            print("Clickbait")
            highlight(article)
        else:
            print("Non-Clickbait")










# response = requests.get(url)
# soup = BeautifulSoup(response.text, "html.parser")
#
# title = soup.title.get_text()
# print(title)
# isClickbait = choose.PredictClass(title)
# if isClickbait:
#     print("Clickbait")
# else:
#     print("Not-Clickbait")
