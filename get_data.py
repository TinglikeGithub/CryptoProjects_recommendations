from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
import time
import pickle
import pandas as pd

data = []

# Extract link and project name
i = 1
while True:
    try:
        website = "https://cryptorank.io/upcoming-ico?page={0}".format(i)
        
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        driver.maximize_window()
        driver.get(website)
        time.sleep(2)
        projects = driver.find_elements(By.XPATH,'//tr[@class="sc-7ff8d1ea-0 kuuWTw init-scroll"]')
        
        for project in projects:    
            name = project.find_element(By.TAG_NAME, "p").text
            token_sale_link = project.find_element(By.TAG_NAME, "a").get_attribute("href")
            overview_link = token_sale_link.replace("/ico/","/price/")
            
            data.append([name, overview_link, token_sale_link])
        driver.quit()
        i += 1
    except:
        break

df = pd.DataFrame(data, columns=['Project Name', 'Overview', 'Token Sale'])
# print(df)

# Extraction of link and project name ends here

services = []
revenue = []

for website in df['Overview']:
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        driver.maximize_window()
        driver.get(website)

        # Write your code here
        
        # Find the service element
        services_ = driver.find_element(By.XPATH,'//a[starts-with(@href, "/categories/")]/p')
        services.append(services_.text)  # appending the text inside a list

        # Finding the revenue amount element
        # revenue_ = driver............
        # revenue.append(revenue_.text)
        
        # Code ends here
        
        driver.quit()

    # By any chance, there's something wrong with the link,
    # we don't want to append same data to the next one.
    except:
        services.append("")
        revenue.append("")

# Adding everything in the original dataFrame(df)

df['Service'] = services
# df['Revenue'] = revenue

# Saving the df