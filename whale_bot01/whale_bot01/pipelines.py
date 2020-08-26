# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import boto3

class WhaleBot01Pipeline:
    def process_item(self, item, spider):
        return item

ITEM_PIPELINES = {'scrapy.pipelines.images.ImagesPipeline': 1}
s3 = boto3.resource('s3')

for bucket in s3.buckets.all():
    print(bucket.name)
