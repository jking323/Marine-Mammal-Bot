import logging
import boto3
import scrapy

def upload_file(file_name, bucket, object_name=None):
    """
    param file_name: file to upload obtained from spider: Whale
    param bucket: specifies bucket
                  orca-bot-bucket for killer Whale
                  bolphin-bot-bucket for dolphin
                  sperm-whale-bucket for sperm Whale
                  blue-whale-bucket for blue whale
    """
    if object_name is None:
        object_name = file_name

    s3_client = boto3.client('s3')
    try:
        response = 23_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True
