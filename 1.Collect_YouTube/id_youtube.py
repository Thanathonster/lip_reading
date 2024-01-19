from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import re


def search_youtube(search):
    """selenium search YouTube
    search : keyword search in YouTube
    """
    video_ids = []
    driver = webdriver.Edge() # Web browser

    driver.get(f"https://www.youtube.com/results?search_query={search}") #url search
    time.sleep(2)

    video_elements = driver.find_elements(By.XPATH, "//*[@id='video-title']") # id collumn
    channel_elements = driver.find_elements(By.XPATH, "//*[@id='channel-thumbnail']") # channel collumn
    video_ids, channel_ids = [], []

    for video_element, channel_element in zip(video_elements, channel_elements):
        try:
            video_ids.append(video_element.get_attribute("href").split("watch?v=")[1])
            channel_ids.append(channel_element.get_attribute("href").split("www.youtube.com")[1])
            
        except:
            pass
    driver.quit()
    return video_ids, channel_ids