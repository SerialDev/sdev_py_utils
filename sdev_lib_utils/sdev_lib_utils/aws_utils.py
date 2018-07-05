
import json
import boto3
import dill as pickle
import zlib

class aws_utils(object):
    def __init__(self, key, secret):

        self.key = key
        self.secret = secret


    def get_s3_client(self):
        self.s3_client =  boto3.client("s3",
                            aws_access_key_id=self.key,
                            aws_secret_access_key=self.secret)
        return self.s3_client


    def get_s3_res(self):
        self.s3_res =  boto3.resource("s3",
                            aws_access_key_id=self.key,
                            aws_secret_access_key=self.secret)
        return self.s3_res


    def get_bucket(self, bucket_name):
        self.current_bucket=  self.get_s3_res().Bucket(bucket_name)
        return self.current_bucket

    def get_bucket_key(self, bucket_name, key_name):
        self.current_bucket_key=  self.get_s3_client().get_object(Bucket=bucket_name, Key=key_name)
        return self.current_bucket_key

    def get_bucket_key_json(self):
        if hasattr(self, 'current_bucket_key'):
            for i in self.current_bucket_key['Body'].iter_lines():
                yield json.loads(i.decode())
        else:
            print("No current bucket object initialized")


    def get_bucket_info(self, prefix):
        if hasattr(self, 'current_bucket'):
            for obj in bucket.objects.filter(Prefix=prefix):
                print(obj.key, obj.size)
                o = obj
            return o
        else:
            print("Bucket has not yet been initialized")

    def to_s3(self, bucket_name, data, key_name):
        response = self.get_s3_client().put_object(Bucket=bucket_name,
                                                   Body=zlib.compress(pickle.dumps(data)),
                                                   Key=key_name)
        return response

    def from_s3(self, bucket_name, key_name):
        return pickle.loads(zlib.decompress(self.get_s3_client().get_object(Bucket=bucket_name, Key=key_name)['Body'].read()))


    def iter_to_s3(self, bucket_name, iterable, key_name, increments=50):

        temp_list = []
        item_num = 0
        file_num = 0
        for item in iterable:
            item_num +=1
            print_iter(item_num)
            temp_list.append(item)
            if item_num != 0 and item_num % increments == 0:
                file_num += 1
                self.to_s3(bucket_name, temp_list, key_name + "_{}".format(file_num))
                temp_list = []

        if temp_list != []:
            self.to_s3(bucket_name, temp_list, key_name + "_{}".format(file_num + 1))
