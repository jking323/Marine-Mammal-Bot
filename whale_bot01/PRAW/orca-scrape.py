import logging
import praw
"""
use praw to pull image URLs then youe scrapy to scrape images from that list
"""
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
for logger_name in ("praw", "prawcore"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(Handler)


reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     refresh_token=refresh_token,
                     user_agent="orca script by /u/goose323")

print("it works")
