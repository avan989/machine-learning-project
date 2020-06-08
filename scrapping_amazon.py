#
# COSC 880 - Project - Web scrapping amazon top 100 ebook.
#
# Based on https://www.datacamp.com/community/tutorials/amazon-web-scraping-using-beautifulsoup
# and https://towardsdatascience.com/top-5-beautiful-soup-functions-7bfe5a693482
# Web crawl amazon top ebook

#import
import requests, re, time, csv
from bs4 import BeautifulSoup 
# https://www.amazon.com/Best-Sellers-Kindle-Store-eBooks/zgbs/digital-text/154606011
# top 100 ebook: https://www.amazon.com/Best-Sellers-Kindle-Store/zgbs/digital-text/ref=zg_bs_pg_1?_encoding=UTF8&pg=1

urls = [
    "https://www.amazon.com/Best-Sellers-Kindle-Store-Salad-Cooking/zgbs/digital-text/156276011/ref=zg_bs_pg_1?_encoding=UTF8&pg=",
    "https://www.amazon.com/Best-Sellers-Kindle-Store-Vegan-Cooking/zgbs/digital-text/14530437011/ref=zg_bs_pg_1?_encoding=UTF8&pg=",
    "https://www.amazon.com/Best-Sellers-Kindle-Store-Vegetable-Cooking/zgbs/digital-text/156277011/ref=zg_bs_pg_1?_encoding=UTF8&pg="   
]
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.102 Safari/537.36 Edge/18.18363"}


# get individual product information data from individual product page
def get_website_data(website_url):
   
    try:
        response = requests.get(website_url, headers=headers)
    except requests.exceptions.ConnectionError:
        time.sleep(10) # if connection error, sleep and try again
        
        try: 
            response = requests.get(website_url, headers=headers)
        except requests.exceptions.ConnectionError:
            print("Error connection to %s" %(website_url))
    
    
    amazon_soup = BeautifulSoup(response.content, features='lxml')
    
    if amazon_soup.find(id='productTitle') is not None:

        product_title = amazon_soup.find(id='productTitle').get_text().strip()
        sales_rank_info = amazon_soup.find(id='SalesRank').get_text().strip()
        sales_rank_info_strip_comma = sales_rank_info.replace(',', '')
        sales_rank_list = re.findall('\d+', sales_rank_info_strip_comma ) # https://stackoverflow.com/questions/26825729/extract-number-from-string-in-python, answer from Irshad Bhat
        overall_sales_rank = int(sales_rank_list[0])
     
        return_data = [product_title, overall_sales_rank]
    else: 
        return_data = ["None", "None"]
    
    return return_data;


# iterate through each product_listing and get link to the product page
def website_list_view(website_url):

    final_product_scrap = []
    
    response = requests.get(website_url, headers=headers)
    amazon_soup = BeautifulSoup(response.content, features='lxml')
    
    # get individual product and construct url
    product_list = amazon_soup.select('li.zg-item-immersion span.a-list-item span.aok-inline-block.zg-item a.a-link-normal')  
    amazon_url = ['https://amazon.com' + link['href'] for link in product_list]
    
    # remove duplicate
    url_list = list(set(amazon_url))
    
    # iterate though to get ebook
    for index, url in enumerate(url_list):
    
        if url.find('product-reviews') >= 0:          
            continue
            
        respond = get_website_data(url);
        final_product_scrap.append(respond)  
        
        time_to_pause = 1 * index % 5 
        time.sleep(time_to_pause)
        
        
    return final_product_scrap


# iterate through url and check if there is pagnation. If there is iterate through each pagnation. 
# only 2 since it is the top 100 - 50 per pages
def iterate_through_amazon(urls):

    total_product_info = []
    for url in urls:
        for i in range(1,3):
            website_url = url + str(i)
            print(website_url)
            products_info = website_list_view(website_url)
            
            total_product_info += products_info
    

    # wrtie file from data  
    with open('total_product_info.csv', mode='w') as product_info:
        
        writer = csv.writer(product_info, delimiter=',')
        
        for row in total_product_info:
            writer.writerow(row)
    
if __name__ == '__main__':
    
    iterate_through_amazon(urls)
    print("finish")