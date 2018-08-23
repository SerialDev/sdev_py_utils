
import boto3
import datetime



class sqs_utils(object):
    def __init__(self, key, secret, region_name="eu-west-1"):

        self.key = key
        self.secret = secret
        self.session = boto3.Session(region_name=region_name,
                                     aws_access_key_id = self.key,
                                     aws_secret_access_key = self.secret,
                                     # aws_session_token=response['Credentials']['SessionToken']
        )

        self.client = self.session.client("sqs")
        self.resource = self.session.resource("sqs")
        self.current_id = self.session.client("sts").get_caller_identity()["Account"]


    def create_queue(self, queue_name, fifo_queue, delay_seconds=0,
                     message_size=262_144, message_retention =
                     345_600, content_deduplication=False, ):
        if fifo_queue == True:
             attributes = {
                 "DelaySeconds" : str(delay_seconds),
                 "MaximumMessageSize": str(message_size),
                 "MessageRetentionPeriod" : str(message_retention),
                 "FifoQueue" : str(fifo_queue),
                 "ContentBasedDeduplication" : str(content_deduplication),
             }
        else:

             attributes = {
                 "DelaySeconds" : str(delay_seconds),
                 "MaximumMessageSize": str(message_size),
                 "MessageRetentionPeriod" : str(message_retention),
             }

        response = self.client.create_queue( QueueName = queue_name, Attributes = attributes)
        return response


    def delete_message(self, queue_url, receipt_handle):
        response = self.client.delete_message( QueueUrl = queue_url, ReceiptHandle= receipt_handle)
        return response


    def gen_delete_entry(self, id, receipt_handle):
        return {"Id": id, "ReceiptHandle": receipt_handle}


    def delete_message_batch(self, queue_url, entry_list):
        response = self.client.delete_message_batch(QueueUrl = queue_url, Entries = entry_list)
        return response


    def delete_queue(self, queue_url):
        response = self.client.delete_queue(QueueUrl = queue_url)
        return response

    def queue_attributes(self, queue_url, attributes_list):
        response = self.client.get_queue_attributes(QueueUrl, AttributeNames = attributes_list)
        return response


    def queue_url(self, queue_name, queue_owner_id=None):
        if queue_owner_id is None :
            queue_owner_id = self.current_id
        response = self.client.get_queue_url(QueueName = queue_name, QueueOwnerAWSAccountId = queue_owner_id)
        return response


    def list_queues(self, queue_name_prefix):
        response = self.client.list_queues(QueueNamePrefix = queue_name_prefix)
        return response


    def purge_queue(self, queue_url):
        response = self.client.purge_queue(QueueUrl = queue_url)
        return response


    def receive_message(self, queue_url, wait_time=0, attribute_name_list=['All'],
                        message_attribute_names=["All"], max_n_messages=1):

        response = self.client.receive_message(QueueUrl = queue_url,
                                               AttributeNames = attribute_name_list,
                                               MessageAttributeNames = message_attribute_names,
                                               WaitTimeSeconds = wait_time,
                                               MaxNumberOfMessages = max_n_messages, )
        return response


    def send_message_fifo(self, queue_url, body, delay_seconds, dedup_id, group_id, attributes=None,  ):
        if attributes is None:
            response = self.client.send_message(QueueUrl = queue_url,
                                            MessageBody = body,
                                            DelaySeconds=delay_seconds,
                                            MessageDeduplicationId = str(dedup_id),
                                            MessageGroupId= group_id )

        else:
            response = self.client.send_message(QueueUrl = queue_url,
                                                MessageBody = body,
                                                DelaySeconds=delay_seconds,
                                                MessageAttributes = attributes,
                                                MessageDeduplicationId = str(dedup_id),
                                                MessageGroupId= group_id )

        return response


    def send_message_standard(self, queue_url, body, delay_seconds, attributes=None, ):
        if attributes is None:
            response = self.client.send_message(QueueUrl = queue_url,
                                                MessageBody = body,
                                                DelaySeconds=delay_seconds, )

        else:
            response = self.client.send_message(QueueUrl = queue_url,
                                                MessageBody = body,
                                                DelaySeconds=delay_seconds,
                                                MessageAttributes = attributes, )


        return response


    def gen_send_fifo_entry(self, queue_url, body, delay_seconds, attributes, dedup_id, group_id):
        entry = {"QueueUrl" : queue_url,
                   "DelaySeconds" : delay_seconds,
                   "MessageAttributes" : attributes,
                   "MessageDeduplicationId" : dedup_id,
                   "MessageGroupId" : group_id }
        return entry


    def gen_send_std_entry(self, queue_url, body, delay_seconds, attributes,):
        entry = {"QueueUrl" : queue_url,
                   "DelaySeconds" : delay_seconds,
                   "MessageAttributes" : attributes,
                    }
        return entry


    def send_message_batch(self, queue_url, entries ):

        response = self.client.send_message_batch(QueueUrl = queue_url,
                                                  Entries= entries )

        return response
