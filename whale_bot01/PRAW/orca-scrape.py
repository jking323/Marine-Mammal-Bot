#! python3.8.5
import logging
import praw
import logging.handlers
import random
import pandas as pd
import datetime as dt
import csv
import numpy as np
import os


"""
use praw to pull image URLs then youe scrapy to scrape images from that list
"""


reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     refresh_token=refresh_token,
                     user_agent="orca script by /u/goose323")

def scrape(sub_scrape):
    url = []
    sub_id = []
    sub = reddit.subreddit(sub_scrape)
    top_sub = sub.top(limit=500)
    for submission in top_sub:
        u = submission.url
        i = submission.id
        url.append(u)
        sub_id.append(i)
    np.savetxt('C:\Temp\orca.csv', [p for p in zip(url, sub_id)], delimiter=',', fmt ='%s')
    print("success")

scrape('orcas')
