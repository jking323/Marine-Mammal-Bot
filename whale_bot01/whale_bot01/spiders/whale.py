import scrapy
import logging
from scrapy.item import WhaleBot01Item
from scrapy.linkextractors import LinkExtractor

class orcaspider(scrapy.Spider)
    name = "orca"

    def start_requests(self):
        urls = [
        'https://old.reddit.com/r/orcas/'
        ]

    def parse(self, response):
        for elem in response.xpath('//html/body'):
            img_url = sel.xpath()
            yield {'image_urls':[img_url]}
