
import json
import boto3
import dill as pickle
import zlib
import io

class aws_utils(object):
    """
    Aws Utility class for interfacing with S3

    Parameters
    ----------

    __init__ : key|string secret|string
       Initialize the class with AWS key and secret

    get_s3_client : None
       Returns an S3 Client object for the low level API

    get_s3_res : None
       Returns a S3 Resource object for the high level API

    get_bucket : bucket_name|string
       Returns a bucket from S3

    get_bucket_key : bucket_name|string key_name|string
       Returns a key object from inside a bucket [file]

    get_bucket_key_json : None
       Use initialised bucket/key and creates an iterator to stream Json results line by line to not consume memory

    get_bucket_info : prefix|string
       Returns info of all objects with prefix from the current bucket

    to_s3 : bucket_name|string key_name|string
       Stores a python object as a compressed pickle

    from_s3 : bucket_name|str key_name|str
       Retrieves a pickle object from S3 uncompresses it and loads it as a python object

    from_bin_file_streaming : name|str[file_path] bucket_name|str key_name|str full_path|bool
       Takes a binary object from S3 and saves it on the filesystem under <name>

    from_bin_streaming : bucket_name|str key_name|str
       Returns a File buffer from a binary file in S3

    to_bin_streaming : data|Binary bucket_name|str key_name|str
       Binary Streaming data into S3 uses low memory

    read_bin : name|str[path] full_path|bool
       reads a binary file from filesystem using the same logic as from_bin_file_streaming

    iter_to_s3 : bucket_name|str iterable|iter key_name|str increments|int
       Stores any iterable into s3 in different increments to allow for reconstruction/function application

    """
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
            for obj in self.current_bucket.objects.filter(Prefix=prefix):
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



    def from_bin_file_streaming(self, name, bucket_name, key_name, full_path=False):
        if full_path:
            path = name
        else:
            path = os.path.join(os.getcwd(), name)

        with open(path, 'ab') as f:
            obj = self.get_s3_client().get_object(Bucket=bucket_name, Key=key_name)['Body'].iter_lines()
            for i in obj:
                f.write(i)


    def from_bin_streaming(self, bucket_name, key_name):
        out_buffer = io.BytesIO()
        obj = self.get_s3_client().get_object(Bucket=bucket_name, Key=key_name)['Body'].iter_lines()
        for i in obj:
            out_buffer.write(i)
        out_buffer.seek(0)
        return out_buffer

    def to_bin_streaming(self, data, bucket_name, key_name):
        out_buffer = io.BytesIO()
        out_buffer.write(data)
        out_buffer.seek(0)
        result = self.get_s3_client().upload_fileobj(out_buffer, bucket_name, key_name)
        return result


    def read_bin(self, name, full_path=False):
        if full_path:
            path = name
        else:
            path = os.path.join(os.getcwd(), name)

        with open(path, 'rb') as f:
            obj = f.read()
        return obj

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


    def iter_bucket(self, bucket_name):
        for i in self.get_s3_res().Bucket(bucket_name).objects.all():
            yield i

    def get_s3_objects_containing(self, bucket_name, containing_string):
        buckets = self.iter_bucket(bucket_name)

        buckets = [i for i in buckets if containing_string in i.key]
        return buckets




def iter_bucket_folder(aws_personal, bucket_name, folder):
    u = aws_personal.iter_bucket(bucket_name)
    for i in u:
        if i.key.split("/")[0] == folder:
            yield i


def get_folders_bucket(aws_personal, bucket_name):
    u = aws_personal.iter_bucket(bucket_name)
    folders = {}
    for i in u:
        folders[i.key.split("/")[0]] = ""
    return list(folders.keys())

def iter_bucket(resource, bucket_name):
    for i in resource.Bucket(bucket_name).objects.all():
        yield i
        

def iter_bucket_folder(resource, bucket_name, folder):
    u = iter_bucket(resource, bucket_name)
    for i in u:
        if i.key.split("/")[0] == folder:
            yield i


def get_folders_bucket(resource, bucket_name):
    u = iter_bucket(resource, bucket_name)
    folders = {}
    for i in u:
        folders[i.key.split("/")[0]] = ""
    return list(folders.keys())

def from_s3(object_summary, client):
    result = client.get_object(Bucket=object_summary.bucket_name,
                                Key=object_summary.key)['Body'].read()
    return result

def download_from_s3(bucket_name, key_name, client):
    """
    Downloads s3 file and returns the path its stored in
    """
    try:
        name = key_name.split('/')[1]
    except Exception as e:
        name = key_name
        
    if os.path.exists(os.path.join(os.getcwd(), name)):
        return os.path.join(os.getcwd(), name)

    result = client.get_object(Bucket=bucket_name, Key=key_name)['Body'].read()
    with open(name, 'wb') as f:
        f.write(result)
    return os.path.join(os.getcwd(), name)

def joblib_load_bytes(file_name, file_bytes):
    try:
        name = file_name.split('/')[1]
    except Exception as e:
        name = file_name
    try:
        with open(name, 'wb') as f:
            f.write(file_bytes)
        return joblib.load(os.path.join(os.getcwd(), name))
    except Exception as e:
        print(f'error with {name} with content {file_bytes} \n with traceback: {traceback.format_exc()}')


def to_s3_pickles(client, bucket_name, data, key_name):
    import pickle
    import zlib
    response = client().put_object(
        Bucket=bucket_name, Body=zlib.compress(pickle.dumps(data)), Key=key_name
    )
    return response


def to_s3(client, bucket_name, data, key_name):
        response = client().put_object(
            Bucket=bucket_name, Body=data, Key=key_name
        )
        return response
