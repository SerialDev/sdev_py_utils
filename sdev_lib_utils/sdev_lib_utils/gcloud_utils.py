from google.cloud import storage
from google.cloud import bigquery
import pprint

import os

from google.oauth2 import service_account
import googleapiclient.discovery


def extract_gsutil_url_components(url):
    import re

    # Define the regex pattern for the gsutil URL
    pattern = r"gs://([^/]+)/(.+)$"
    match = re.match(pattern, url)

    if not match:
        raise ValueError("Invalid gsutil URL")

    bucket_name, blob_name = match.groups()

    # For this example, we'll assume the project name is part of the bucket name
    project = None

    return bucket_name, blob_name


def write_into_bigquery(
    bq_write_client,
    input_path: str,
    table_info: dict,
    table_name: str,
    file_format: str,
    mode: str,
    partition_value: str,
) -> None:
    """
    This function writes data to BigQuery table
    [ Arguments ]
        bq_write_client - intialized BigQuery client to write data
        input_path - GCS path
        table_info - Pass table description, labels & schema of table with field types & column descriptions in a dictionary
        table_name - BigQuery table path. eg: ida.contract_churn_scores
        file_format - File format to read. Default format is Parquet
        mode - Mode to save. Append or Overwrite
        partition_value - Partition column value
    [ Returns ]
        None
    """
    if file_format.lower() == "parquet":
        format_type = bigquery.SourceFormat.PARQUET
    elif file_format.lower() == "csv":
        format_type = bigquery.SourceFormat.CSV
    elif file_format.lower() == "avro":
        format_type = bigquery.SourceFormat.AVRO
    elif file_format.lower() == "orc":
        format_type = bigquery.SourceFormat.ORC
    elif file_format.lower() == "json":
        format_type = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    else:
        raise f"unsupported file format: {file_format.lower()}"

    if mode.lower() == "append":
        write_type = bigquery.WriteDisposition.WRITE_APPEND
    elif mode.lower() == "overwrite":
        write_type = bigquery.WriteDisposition.WRITE_TRUNCATE
    #         print(f"Write Type = {write_type}")
    else:
        write_type = bigquery.WriteDisposition.WRITE_EMPTY

    if partition_value != "":
        time_partition_config = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="model_run_month",  # field to use for partitioning
            require_partition_filter=True,
        )
    else:
        time_partition_config = bigquery.TimePartitioning()

    job_config = bigquery.LoadJobConfig(
        schema_update_options=bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION,
        schema=table_info["table_schema"],
        source_format=format_type,
        write_disposition=write_type,
        time_partitioning=time_partition_config,
    )

    _logger.info("Updating description & labels to the %s table", table_name)
    table_id = table_name + "$" + partition_value.replace("-", "")
    bq_dataset, bq_table = table_name.split(".")
    job = bq_write_client.load_table_from_uri(
        input_path, table_id, job_config=job_config
    )

    # Wait to ensure the table is created
    time.sleep(5)

    # Request results from the job
    result = job.result()
    table_ref = bq_write_client.dataset(bq_dataset).table(bq_table)

    try:
        table = bq_write_client.get_table(table_ref)
    except Exception:  # pylint: disable=broad-except
        _logger.exception("An exception occurred from getting table: %s", table_ref)

    # Description
    table.description = table_info["table_description"]
    table = bq_write_client.update_table(table, ["description"])
    assert table.description == table_info["table_description"]

    # Labels
    table.labels = table_info["table_labels"]
    table = bq_write_client.update_table(table, ["labels"])
    assert table.labels == table_info["table_labels"]
    _logger.info("Written %s rows to %s", result.output_rows, result.destination)


def bigquery_column_dtypes(project_name, dataset_name):
    query = f"""
select column_name, data_type
from `{project_name}.{dataset_name}.INFORMATION_SCHEMA.COLUMNS`
where table_name = '{dataset_name}'
order by ordinal_position
    """
    return query


def bigquery_aos(client, query):
    """
    client: google.cloud.bigquery.client.Client, query: str
    """
    result = []
    query_job = client.query(query)  # Make an API request.
    for row in query_job:
        result.append(dict(row.items()))
    return result


def time_delta_days(day_count):
    import datetime

    return datetime.datetime.now() - datetime.timedelta(days=day_count)


def bucket_metadata(bucket_name):
    """Prints out a bucket's metadata."""
    # bucket_name = 'your-bucket-name'

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    print("ID: {}".format(bucket.id))
    print("Name: {}".format(bucket.name))
    print("Storage Class: {}".format(bucket.storage_class))
    print("Location: {}".format(bucket.location))
    print("Location Type: {}".format(bucket.location_type))
    print("Cors: {}".format(bucket.cors))
    print("Default Event Based Hold: {}".format(bucket.default_event_based_hold))
    print("Default KMS Key Name: {}".format(bucket.default_kms_key_name))
    print("Metageneration: {}".format(bucket.metageneration))
    print("Retention Effective Time: {}".format(bucket.retention_policy_effective_time))
    print("Retention Period: {}".format(bucket.retention_period))
    print("Retention Policy Locked: {}".format(bucket.retention_policy_locked))
    print("Requester Pays: {}".format(bucket.requester_pays))
    print("Self Link: {}".format(bucket.self_link))
    print("Time Created: {}".format(bucket.time_created))
    print("Versioning Enabled: {}".format(bucket.versioning_enabled))
    print("Labels:")
    pprint.pprint(bucket.labels)


def create_bucket(bucket_name):
    """Creates a new bucket."""
    # bucket_name = "your-new-bucket-name"

    storage_client = storage.Client()

    bucket = storage_client.create_bucket(bucket_name)

    print("Bucket {} created".format(bucket.name))


def list_buckets():
    """Lists all buckets."""

    storage_client = storage.Client()
    buckets = storage_client.list_buckets()

    for bucket in buckets:
        print(bucket.name)


def upload_blob_file(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print("File {} uploaded to {}.".format(source_file_name, destination_blob_name))


def list_blobs(bucket_name, show=False):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"
    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)

    blobs_list = []
    for blob in blobs:
        if show:
            print(blob.name)
            blobs_list.append(blob)
        else:
            blobs_list.append(blob)
    return blobs_list


def list_blobs_with_prefix(bucket_name, prefix, delimiter=None):
    """Lists all the blobs in the bucket that begin with the prefix.

    This can be used to list all blobs in a "folder", e.g. "public/".

    The delimiter argument can be used to restrict the results to only the
    "files" in the given "folder". Without the delimiter, the entire tree under
    the prefix is returned. For example, given these blobs:

        a/1.txt
        a/b/2.txt

    If you just specify prefix = 'a', you'll get back:

        a/1.txt
        a/b/2.txt

    However, if you specify prefix='a' and delimiter='/', you'll get back:

        a/1.txt

    Additionally, the same request will return blobs.prefixes populated with:

        a/b/
    """

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)

    print("Blobs:")
    for blob in blobs:
        print(blob.name)

    if delimiter:
        print("Prefixes:")
        for prefix in blobs.prefixes:
            print(prefix)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))


def rename_blob(bucket_name, blob_name, new_name):
    """Renames a blob."""
    # bucket_name = "your-bucket-name"
    # blob_name = "your-object-name"
    # new_name = "new-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    new_blob = bucket.rename_blob(blob, new_name)

    print("Blob {} has been renamed to {}".format(blob.name, new_blob.name))


def copy_blob(bucket_name, blob_name, destination_bucket_name, destination_blob_name):
    """Copies a blob from one bucket to another with a new name."""
    # bucket_name = "your-bucket-name"
    # blob_name = "your-object-name"
    # destination_bucket_name = "destination-bucket-name"
    # destination_blob_name = "destination-object-name"

    storage_client = storage.Client()

    source_bucket = storage_client.bucket(bucket_name)
    source_blob = source_bucket.blob(blob_name)
    destination_bucket = storage_client.bucket(destination_bucket_name)

    blob_copy = source_bucket.copy_blob(
        source_blob, destination_bucket, destination_blob_name
    )

    print(
        "Blob {} in bucket {} copied to blob {} in bucket {}.".format(
            source_blob.name,
            source_bucket.name,
            blob_copy.name,
            destination_bucket.name,
        )
    )


def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # blob_name = "your-object-name"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()

    print("Blob {} deleted.".format(blob_name))


def add_bucket_label(bucket_name):
    """Add a label to a bucket."""
    # bucket_name = "your-bucket-name"

    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)
    labels = bucket.labels
    labels["example"] = "label"
    bucket.labels = labels
    bucket.patch()

    print("Updated labels on {}.".format(bucket.name))
    pprint.pprint(bucket.labels)


def get_bucket_labels(bucket_name):
    """Prints out a bucket's labels."""
    # bucket_name = 'your-bucket-name'
    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)

    labels = bucket.labels
    pprint.pprint(labels)


def remove_bucket_label(bucket_name):
    """Remove a label from a bucket."""
    # bucket_name = "your-bucket-name"

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    labels = bucket.labels

    if "example" in labels:
        del labels["example"]

    bucket.labels = labels
    bucket.patch()

    print("Removed labels on {}.".format(bucket.name))
    pprint.pprint(bucket.labels)


def enable_requester_pays(bucket_name):
    """Enable a bucket's requesterpays metadata"""
    # bucket_name = "my-bucket"

    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)
    bucket.requester_pays = True
    bucket.patch()

    print("Requester Pays has been enabled for {}".format(bucket_name))


def disable_requester_pays(bucket_name):
    """Disable a bucket's requesterpays metadata"""
    # bucket_name = "my-bucket"

    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)
    bucket.requester_pays = False
    bucket.patch()

    print("Requester Pays has been disabled for {}".format(bucket_name))


def get_requester_pays_status(bucket_name):
    """Get a bucket's requester pays metadata"""
    # bucket_name = "my-bucket"
    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)
    requester_pays_status = bucket.requester_pays

    if requester_pays_status:
        print("Requester Pays is enabled for {}".format(bucket_name))
    else:
        print("Requester Pays is disabled for {}".format(bucket_name))


def download_file_requester_pays(
    bucket_name, project_id, source_blob_name, destination_file_name
):
    """Download file using specified project as the requester"""
    # bucket_name = "your-bucket-name"
    # project_id = "your-project-id"
    # source_blob_name = "source-blob-name"
    # destination_file_name = "local-destination-file-name"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name, user_project=project_id)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {} using a requester-pays request.".format(
            source_blob_name, destination_file_name
        )
    )


def set_retention_policy(bucket_name, retention_period):
    """Defines a retention policy on a given bucket"""
    # bucket_name = "my-bucket"
    # retention_period = 10

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    bucket.retention_period = retention_period
    bucket.patch()

    print(
        "Bucket {} retention period set for {} seconds".format(
            bucket.name, bucket.retention_period
        )
    )


def remove_retention_policy(bucket_name):
    """Removes the retention policy on a given bucket"""
    # bucket_name = "my-bucket"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    bucket.reload()

    if bucket.retention_policy_locked:
        print("Unable to remove retention period as retention policy is locked.")
        return

    bucket.retention_period = None
    bucket.patch()

    print("Removed bucket {} retention policy".format(bucket.name))


def lock_retention_policy(bucket_name):
    """Locks the retention policy on a given bucket"""
    # bucket_name = "my-bucket"

    storage_client = storage.Client()
    # get_bucket gets the current metageneration value for the bucket,
    # required by lock_retention_policy.
    bucket = storage_client.get_bucket(bucket_name)

    # Warning: Once a retention policy is locked it cannot be unlocked
    # and retention period can only be increased.
    bucket.lock_retention_policy()

    print("Retention policy for {} is now locked".format(bucket_name))
    print(
        "Retention policy effective as of {}".format(
            bucket.retention_policy_effective_time
        )
    )


def get_retention_policy(bucket_name):
    """Gets the retention policy on a given bucket"""
    # bucket_name = "my-bucket"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    bucket.reload()

    print("Retention Policy for {}".format(bucket_name))
    print("Retention Period: {}".format(bucket.retention_period))
    if bucket.retention_policy_locked:
        print("Retention Policy is locked")

    if bucket.retention_policy_effective_time:
        print("Effective Time: {}".format(bucket.retention_policy_effective_time))


def enable_default_event_based_hold(bucket_name):
    """Enables the default event based hold on a given bucket"""
    # bucket_name = "my-bucket"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    bucket.default_event_based_hold = True
    bucket.patch()

    print("Default event based hold was enabled for {}".format(bucket_name))


def get_default_event_based_hold(bucket_name):
    """Gets the default event based hold on a given bucket"""
    # bucket_name = "my-bucket"

    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)

    if bucket.default_event_based_hold:
        print("Default event-based hold is enabled for {}".format(bucket_name))
    else:
        print("Default event-based hold is not enabled for {}".format(bucket_name))


def disable_default_event_based_hold(bucket_name):
    """Disables the default event based hold on a given bucket"""
    # bucket_name = "my-bucket"

    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)
    bucket.default_event_based_hold = False
    bucket.patch()

    print("Default event based hold was disabled for {}".format(bucket_name))


def set_event_based_hold(bucket_name, blob_name):
    """Sets a event based hold on a given blob"""
    # bucket_name = "my-bucket"
    # blob_name = "my-blob"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.event_based_hold = True
    blob.patch()

    print("Event based hold was set for {}".format(blob_name))


def set_temporary_hold(bucket_name, blob_name):
    """Sets a temporary hold on a given blob"""
    # bucket_name = "my-bucket"
    # blob_name = "my-blob"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.temporary_hold = True
    blob.patch()

    print("Temporary hold was set for #{blob_name}")


def release_event_based_hold(bucket_name, blob_name):
    """Releases the event based hold on a given blob"""

    # bucket_name = "my-bucket"
    # blob_name = "my-blob"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.event_based_hold = False
    blob.patch()

    print("Event based hold was released for {}".format(blob_name))


def release_temporary_hold(bucket_name, blob_name):
    """Releases the temporary hold on a given blob"""

    # bucket_name = "my-bucket"
    # blob_name = "my-blob"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.temporary_hold = False
    blob.patch()

    print("Temporary hold was release for #{blob_name}")


import base64
import os


def generate_encryption_key():
    """Generates a 256 bit (32 byte) AES encryption key and prints the
    base64 representation.

    This is included for demonstration purposes. You should generate your own
    key. Please remember that encryption keys should be handled with a
    comprehensive security policy.
    """
    key = os.urandom(32)
    encoded_key = base64.b64encode(key).decode("utf-8")

    print("Base 64 encoded encryption key: {}".format(encoded_key))


def upload_encrypted_blob(
    bucket_name,
    source_file_name,
    destination_blob_name,
    base64_encryption_key,
):
    """Uploads a file to a Google Cloud Storage bucket using a custom
    encryption key.

    The file will be encrypted by Google Cloud Storage and only
    retrievable using the provided encryption key.
    """

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Encryption key must be an AES256 key represented as a bytestring with
    # 32 bytes. Since it's passed in as a base64 encoded string, it needs
    # to be decoded.
    encryption_key = base64.b64decode(base64_encryption_key)
    blob = bucket.blob(destination_blob_name, encryption_key=encryption_key)

    blob.upload_from_filename(source_file_name)

    print("File {} uploaded to {}.".format(source_file_name, destination_blob_name))


def download_encrypted_blob(
    bucket_name,
    source_blob_name,
    destination_file_name,
    base64_encryption_key,
):
    """Downloads a previously-encrypted blob from Google Cloud Storage.

    The encryption key provided must be the same key provided when uploading
    the blob.
    """
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"
    # base64_encryption_key = "base64-encoded-encryption-key"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Encryption key must be an AES256 key represented as a bytestring with
    # 32 bytes. Since it's passed in as a base64 encoded string, it needs
    # to be decoded.
    encryption_key = base64.b64decode(base64_encryption_key)
    blob = bucket.blob(source_blob_name, encryption_key=encryption_key)

    blob.download_to_filename(destination_file_name)

    print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))


def rotate_encryption_key(
    bucket_name, blob_name, base64_encryption_key, base64_new_encryption_key
):
    """Performs a key rotation by re-writing an encrypted blob with a new
    encryption key."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    current_encryption_key = base64.b64decode(base64_encryption_key)
    new_encryption_key = base64.b64decode(base64_new_encryption_key)

    # Both source_blob and destination_blob refer to the same storage object,
    # but destination_blob has the new encryption key.
    source_blob = bucket.blob(blob_name, encryption_key=current_encryption_key)
    destination_blob = bucket.blob(blob_name, encryption_key=new_encryption_key)

    token = None

    while True:
        token, bytes_rewritten, total_bytes = destination_blob.rewrite(
            source_blob, token=token
        )
        if token is None:
            break

    print("Key rotation complete for Blob {}".format(blob_name))


def add_member_to_crypto_key_policy(
    project_id, location_id, key_ring_id, crypto_key_id, member, role
):
    """Adds a member with a given role to the Identity and Access Management
    (IAM) policy for a given CryptoKey associated with a KeyRing."""

    from google.cloud import kms_v1

    # Creates an API client for the KMS API.
    client = kms_v1.KeyManagementServiceClient()

    # The resource name of the CryptoKey.
    resource = client.crypto_key_path_path(
        project_id, location_id, key_ring_id, crypto_key_id
    )
    # Get the current IAM policy.
    policy = client.get_iam_policy(resource)

    # Add member
    policy.bindings.add(role=role, members=[member])

    # Update the IAM Policy.
    client.set_iam_policy(resource, policy)

    # Print results
    print(
        "Member {} added with role {} to policy for CryptoKey {} \
           in KeyRing {}".format(
            member, role, crypto_key_id, key_ring_id
        )
    )


def enable_default_kms_key(bucket_name, kms_key_name):
    """Sets a bucket's default KMS key."""
    # bucket_name = "your-bucket-name"
    # kms_key_name = "projects/PROJ/locations/LOC/keyRings/RING/cryptoKey/KEY"

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    bucket.default_kms_key_name = kms_key_name
    bucket.patch()

    print(
        "Set default KMS key for bucket {} to {}.".format(
            bucket.name, bucket.default_kms_key_name
        )
    )


def upload_blob_with_kms(
    bucket_name, source_file_name, destination_blob_name, kms_key_name
):
    """Uploads a file to the bucket, encrypting it with the given KMS key."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"
    # kms_key_name = "projects/PROJ/locations/LOC/keyRings/RING/cryptoKey/KEY"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name, kms_key_name=kms_key_name)
    blob.upload_from_filename(source_file_name)

    print(
        "File {} uploaded to {} with encryption key {}.".format(
            source_file_name, destination_blob_name, kms_key_name
        )
    )


def create_key(project_id, service_account_email):
    """
    Create a new HMAC key using the given project and service account.
    """
    # project_id = 'Your Google Cloud project ID'
    # service_account_email = 'Service account used to generate HMAC key'

    storage_client = storage.Client(project=project_id)

    hmac_key, secret = storage_client.create_hmac_key(
        service_account_email=service_account_email, project_id=project_id
    )

    print("The base64 encoded secret is {}".format(secret))
    print("Do not miss that secret, there is no API to recover it.")
    print("The HMAC key metadata is:")
    print("Service Account Email: {}".format(hmac_key.service_account_email))
    print("Key ID: {}".format(hmac_key.id))
    print("Access ID: {}".format(hmac_key.access_id))
    print("Project ID: {}".format(hmac_key.project))
    print("State: {}".format(hmac_key.state))
    print("Created At: {}".format(hmac_key.time_created))
    print("Updated At: {}".format(hmac_key.updated))
    print("Etag: {}".format(hmac_key.etag))
    return hmac_key


def list_keys(project_id):
    """
    List all HMAC keys associated with the project.
    """
    # project_id = "Your Google Cloud project ID"

    storage_client = storage.Client(project=project_id)
    hmac_keys = storage_client.list_hmac_keys(project_id=project_id)
    print("HMAC Keys:")
    for hmac_key in hmac_keys:
        print("Service Account Email: {}".format(hmac_key.service_account_email))
        print("Access ID: {}".format(hmac_key.access_id))
    return hmac_keys


def get_key(access_id, project_id):
    """
    Retrieve the HMACKeyMetadata with the given access id.
    """
    # project_id = "Your Google Cloud project ID"
    # access_id = "ID of an HMAC key"

    storage_client = storage.Client(project=project_id)

    hmac_key = storage_client.get_hmac_key_metadata(access_id, project_id=project_id)

    print("The HMAC key metadata is:")
    print("Service Account Email: {}".format(hmac_key.service_account_email))
    print("Key ID: {}".format(hmac_key.id))
    print("Access ID: {}".format(hmac_key.access_id))
    print("Project ID: {}".format(hmac_key.project))
    print("State: {}".format(hmac_key.state))
    print("Created At: {}".format(hmac_key.time_created))
    print("Updated At: {}".format(hmac_key.updated))
    print("Etag: {}".format(hmac_key.etag))
    return hmac_key


def deactivate_key(access_id, project_id):
    """
    Deactivate the HMAC key with the given access ID.
    """
    # project_id = "Your Google Cloud project ID"
    # access_id = "ID of an active HMAC key"

    storage_client = storage.Client(project=project_id)

    hmac_key = storage_client.get_hmac_key_metadata(access_id, project_id=project_id)
    hmac_key.state = "INACTIVE"
    hmac_key.update()

    print("The HMAC key is now inactive.")
    print("The HMAC key metadata is:")
    print("Service Account Email: {}".format(hmac_key.service_account_email))
    print("Key ID: {}".format(hmac_key.id))
    print("Access ID: {}".format(hmac_key.access_id))
    print("Project ID: {}".format(hmac_key.project))
    print("State: {}".format(hmac_key.state))
    print("Created At: {}".format(hmac_key.time_created))
    print("Updated At: {}".format(hmac_key.updated))
    print("Etag: {}".format(hmac_key.etag))
    return hmac_key


def activate_key(access_id, project_id):
    """
    Activate the HMAC key with the given access ID.
    """
    # project_id = "Your Google Cloud project ID"
    # access_id = "ID of an inactive HMAC key"

    storage_client = storage.Client(project=project_id)

    hmac_key = storage_client.get_hmac_key_metadata(access_id, project_id=project_id)
    hmac_key.state = "ACTIVE"
    hmac_key.update()

    print("The HMAC key metadata is:")
    print("Service Account Email: {}".format(hmac_key.service_account_email))
    print("Key ID: {}".format(hmac_key.id))
    print("Access ID: {}".format(hmac_key.access_id))
    print("Project ID: {}".format(hmac_key.project))
    print("State: {}".format(hmac_key.state))
    print("Created At: {}".format(hmac_key.time_created))
    print("Updated At: {}".format(hmac_key.updated))
    print("Etag: {}".format(hmac_key.etag))
    return hmac_key


def delete_key(access_id, project_id):
    """
    Delete the HMAC key with the given access ID. Key must have state INACTIVE
    in order to succeed.
    """
    # project_id = "Your Google Cloud project ID"
    # access_id = "ID of an HMAC key (must be in INACTIVE state)"

    storage_client = storage.Client(project=project_id)

    hmac_key = storage_client.get_hmac_key_metadata(access_id, project_id=project_id)
    hmac_key.delete()

    print(
        "The key is deleted, though it may still appear in list_hmac_keys()" " results."
    )


def make_blob_public(bucket_name, blob_name):
    """Makes a blob publicly accessible."""
    # bucket_name = "your-bucket-name"
    # blob_name = "your-object-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.make_public()

    print("Blob {} is publicly accessible at {}".format(blob.name, blob.public_url))


def add_bucket_iam_member(bucket_name, role, member):
    """Add a new member to an IAM Policy"""
    # bucket_name = "your-bucket-name"
    # role = "IAM role, e.g. roles/storage.objectViewer"
    # member = "IAM identity, e.g. user: name@example.com"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    policy = bucket.get_iam_policy(requested_policy_version=3)

    policy.bindings.append({"role": role, "members": {member}})

    bucket.set_iam_policy(policy)

    print("Added {} with role {} to {}.".format(member, role, bucket_name))


def view_bucket_iam_members(bucket_name):
    """View IAM Policy for a bucket"""
    # bucket_name = "your-bucket-name"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    policy = bucket.get_iam_policy(requested_policy_version=3)

    for binding in policy.bindings:
        print("Role: {}, Members: {}".format(binding["role"], binding["members"]))


def remove_bucket_iam_member(bucket_name, role, member):
    """Remove member from bucket IAM Policy"""
    # bucket_name = "your-bucket-name"
    # role = "IAM role, e.g. roles/storage.objectViewer"
    # member = "IAM identity, e.g. user: name@example.com"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    policy = bucket.get_iam_policy(requested_policy_version=3)

    for binding in policy.bindings:
        print(binding)
        if binding["role"] == role and binding.get("condition") is None:
            binding["members"].discard(member)

    bucket.set_iam_policy(policy)

    print("Removed {} with role {} from {}.".format(member, role, bucket_name))


def add_bucket_conditional_iam_binding(
    bucket_name, role, title, description, expression, members
):
    """Add a conditional IAM binding to a bucket's IAM policy."""
    # bucket_name = "your-bucket-name"
    # role = "IAM role, e.g. roles/storage.objectViewer"
    # members = {"IAM identity, e.g. user: name@example.com}"
    # title = "Condition title."
    # description = "Condition description."
    # expression = "Condition expression."

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    policy = bucket.get_iam_policy(requested_policy_version=3)

    # Set the policy's version to 3 to use condition in bindings.
    policy.version = 3

    policy.bindings.append(
        {
            "role": role,
            "members": members,
            "condition": {
                "title": title,
                "description": description,
                "expression": expression,
            },
        }
    )

    bucket.set_iam_policy(policy)

    print("Added the following member(s) with role {} to {}:".format(role, bucket_name))

    for member in members:
        print("    {}".format(member))

    print("with condition:")
    print("    Title: {}".format(title))
    print("    Description: {}".format(description))
    print("    Expression: {}".format(expression))


def enable_uniform_bucket_level_access(bucket_name):
    """Enable uniform bucket-level access for a bucket"""
    # bucket_name = "my-bucket"

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    bucket.iam_configuration.uniform_bucket_level_access_enabled = True
    bucket.patch()

    print("Uniform bucket-level access was enabled for {}.".format(bucket.name))


def get_uniform_bucket_level_access(bucket_name):
    """Get uniform bucket-level access for a bucket"""
    # bucket_name = "my-bucket"

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    iam_configuration = bucket.iam_configuration

    if iam_configuration.uniform_bucket_level_access_enabled:
        print("Uniform bucket-level access is enabled for {}.".format(bucket.name))
        print(
            "Bucket will be locked on {}.".format(
                iam_configuration.uniform_bucket_level_locked_time
            )
        )
    else:
        print("Uniform bucket-level access is disabled for {}.".format(bucket.name))


def disable_uniform_bucket_level_access(bucket_name):
    """Disable uniform bucket-level access for a bucket"""
    # bucket_name = "my-bucket"

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    bucket.iam_configuration.uniform_bucket_level_access_enabled = False
    bucket.patch()

    print("Uniform bucket-level access was disabled for {}.".format(bucket.name))


def add_bucket_owner(bucket_name, user_email):
    """Adds a user as an owner on the given bucket."""
    # bucket_name = "your-bucket-name"
    # user_email = "name@example.com"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Reload fetches the current ACL from Cloud Storage.
    bucket.acl.reload()

    # You can also use `group()`, `domain()`, `all_authenticated()` and `all()`
    # to grant access to different types of entities.
    # You can also use `grant_read()` or `grant_write()` to grant different
    # roles.
    bucket.acl.user(user_email).grant_owner()
    bucket.acl.save()

    print("Added user {} as an owner on bucket {}.".format(user_email, bucket_name))


def add_blob_owner(bucket_name, blob_name, user_email):
    """Adds a user as an owner on the given blob."""
    # bucket_name = "your-bucket-name"
    # blob_name = "your-object-name"
    # user_email = "name@example.com"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Reload fetches the current ACL from Cloud Storage.
    blob.acl.reload()

    # You can also use `group`, `domain`, `all_authenticated` and `all` to
    # grant access to different types of entities. You can also use
    # `grant_read` or `grant_write` to grant different roles.
    blob.acl.user(user_email).grant_owner()
    blob.acl.save()

    print(
        "Added user {} as an owner on blob {} in bucket {}.".format(
            user_email, blob_name, bucket_name
        )
    )


def add_bucket_default_owner(bucket_name, user_email):
    """Adds a user as an owner in the given bucket's default object access
    control list."""
    # bucket_name = "your-bucket-name"
    # user_email = "name@example.com"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Reload fetches the current ACL from Cloud Storage.
    bucket.acl.reload()

    # You can also use `group`, `domain`, `all_authenticated` and `all` to
    # grant access to different types of entities. You can also use
    # `grant_read` or `grant_write` to grant different roles.
    bucket.default_object_acl.user(user_email).grant_owner()
    bucket.default_object_acl.save()

    print(
        "Added user {} as an owner in the default acl on bucket {}.".format(
            user_email, bucket_name
        )
    )


def remove_bucket_default_owner(bucket_name, user_email):
    """Removes a user from the access control list of the given bucket's
    default object access control list."""
    # bucket_name = "your-bucket-name"
    # user_email = "name@example.com"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Reload fetches the current ACL from Cloud Storage.
    bucket.acl.reload()

    # You can also use `group`, `domain`, `all_authenticated` and `all` to
    # remove access for different types of entities.
    bucket.default_object_acl.user(user_email).revoke_read()
    bucket.default_object_acl.user(user_email).revoke_write()
    bucket.default_object_acl.user(user_email).revoke_owner()
    bucket.default_object_acl.save()

    print(
        "Removed user {} from the default acl of bucket {}.".format(
            user_email, bucket_name
        )
    )


def print_bucket_acl(bucket_name):
    """Prints out a bucket's access control list."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    for entry in bucket.acl:
        print("{}: {}".format(entry["role"], entry["entity"]))


def print_blob_acl(bucket_name, blob_name):
    """Prints out a blob's access control list."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    for entry in blob.acl:
        print("{}: {}".format(entry["role"], entry["entity"]))


def remove_bucket_owner(bucket_name, user_email):
    """Removes a user from the access control list of the given bucket."""
    # bucket_name = "your-bucket-name"
    # user_email = "name@example.com"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Reload fetches the current ACL from Cloud Storage.
    bucket.acl.reload()

    # You can also use `group`, `domain`, `all_authenticated` and `all` to
    # remove access for different types of entities.
    bucket.acl.user(user_email).revoke_read()
    bucket.acl.user(user_email).revoke_write()
    bucket.acl.user(user_email).revoke_owner()
    bucket.acl.save()

    print("Removed user {} from bucket {}.".format(user_email, bucket_name))


def remove_blob_owner(bucket_name, blob_name, user_email):
    """Removes a user from the access control list of the given blob in the
    given bucket."""
    # bucket_name = "your-bucket-name"
    # blob_name = "your-object-name"
    # user_email = "name@example.com"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # You can also use `group`, `domain`, `all_authenticated` and `all` to
    # remove access for different types of entities.
    blob.acl.user(user_email).revoke_read()
    blob.acl.user(user_email).revoke_write()
    blob.acl.user(user_email).revoke_owner()
    blob.acl.save()

    print(
        "Removed user {} from blob {} in bucket {}.".format(
            user_email, blob_name, bucket_name
        )
    )


def generate_download_signed_url_v4(bucket_name, blob_name):
    """Generates a v4 signed URL for downloading a blob.

    Note that this method requires a service account key file. You can not use
    this if you are using Application Default Credentials from Google Compute
    Engine or from the Google Cloud SDK.
    """
    # bucket_name = 'your-bucket-name'
    # blob_name = 'your-object-name'
    import datetime

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version="v4",
        # This URL is valid for 15 minutes
        expiration=datetime.timedelta(minutes=15),
        # Allow GET requests using this URL.
        method="GET",
    )

    print("Generated GET signed URL:")
    print(url)
    print("You can use this URL with any user agent, for example:")
    print("curl '{}'".format(url))
    return url


def generate_signed_url(
    service_account_file,
    bucket_name,
    object_name,
    subresource=None,
    expiration=604800,
    http_method="GET",
    query_parameters=None,
    headers=None,
):

    import binascii
    import collections
    import datetime
    import hashlib
    import sys

    # pip install six
    import six
    from six.moves.urllib.parse import quote

    # pip install google-auth
    from google.oauth2 import service_account

    if expiration > 604800:
        print("Expiration Time can't be longer than 604800 seconds (7 days).")
        sys.exit(1)

    escaped_object_name = quote(six.ensure_binary(object_name), safe=b"/~")
    canonical_uri = "/{}".format(escaped_object_name)

    datetime_now = datetime.datetime.utcnow()
    request_timestamp = datetime_now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = datetime_now.strftime("%Y%m%d")

    google_credentials = service_account.Credentials.from_service_account_file(
        service_account_file
    )
    client_email = google_credentials.service_account_email
    credential_scope = "{}/auto/storage/goog4_request".format(datestamp)
    credential = "{}/{}".format(client_email, credential_scope)

    if headers is None:
        headers = dict()
    host = "{}.storage.googleapis.com".format(bucket_name)
    headers["host"] = host

    canonical_headers = ""
    ordered_headers = collections.OrderedDict(sorted(headers.items()))
    for k, v in ordered_headers.items():
        lower_k = str(k).lower()
        strip_v = str(v).lower()
        canonical_headers += "{}:{}\n".format(lower_k, strip_v)

    signed_headers = ""
    for k, _ in ordered_headers.items():
        lower_k = str(k).lower()
        signed_headers += "{};".format(lower_k)
    signed_headers = signed_headers[:-1]  # remove trailing ';'

    if query_parameters is None:
        query_parameters = dict()
    query_parameters["X-Goog-Algorithm"] = "GOOG4-RSA-SHA256"
    query_parameters["X-Goog-Credential"] = credential
    query_parameters["X-Goog-Date"] = request_timestamp
    query_parameters["X-Goog-Expires"] = expiration
    query_parameters["X-Goog-SignedHeaders"] = signed_headers
    if subresource:
        query_parameters[subresource] = ""

    canonical_query_string = ""
    ordered_query_parameters = collections.OrderedDict(sorted(query_parameters.items()))
    for k, v in ordered_query_parameters.items():
        encoded_k = quote(str(k), safe="")
        encoded_v = quote(str(v), safe="")
        canonical_query_string += "{}={}&".format(encoded_k, encoded_v)
    canonical_query_string = canonical_query_string[:-1]  # remove trailing '&'

    canonical_request = "\n".join(
        [
            http_method,
            canonical_uri,
            canonical_query_string,
            canonical_headers,
            signed_headers,
            "UNSIGNED-PAYLOAD",
        ]
    )

    canonical_request_hash = hashlib.sha256(canonical_request.encode()).hexdigest()

    string_to_sign = "\n".join(
        [
            "GOOG4-RSA-SHA256",
            request_timestamp,
            credential_scope,
            canonical_request_hash,
        ]
    )

    signature = binascii.hexlify(
        google_credentials.signer.sign(string_to_sign)
    ).decode()

    scheme_and_host = "{}://{}".format("https", host)
    signed_url = "{}{}?{}&x-goog-signature={}".format(
        scheme_and_host, canonical_uri, canonical_query_string, signature
    )

    return signed_url


def upload_bytesio_blob(
    bucket_name, blob_name, content, project="cloudflare-ds-data", encode=False
):
    """
    Uploads a BytesIO object to Google Cloud Storage as a blob.

    This function takes in a bucket name, a blob name, a BytesIO object, and a boolean indicating whether the
    BytesIO object should be encoded. If the encode option is set to True, it encodes the BytesIO object before
    uploading it to the specified bucket. It then prints a message indicating that the file has been uploaded.

    Args:
        bucket_name: The name of the bucket to upload the blob to.
        blob_name: The name of the blob to upload.
        content: The BytesIO object to be uploaded as a blob.
        encode: A boolean indicating whether the BytesIO object should be encoded before uploading.

    Returns:
        None
    """

    # Import the io module and Google Cloud Storage client
    import io
    from google.cloud.storage import Client

    # Create a BytesIO object
    buff = io.BytesIO()

    # Create a Google Cloud Storage client
    storage_client = Client(project=project)
    bucket = storage_client.bucket(bucket_name)

    # Encode the BytesIO object if necessary
    if encode:
        buff.seek(0)
        buff.write(content.encode())
        buff.seek(0)
    else:
        buff.seek(0)
        buff.write(content)
        buff.seek(0)

    # Upload the BytesIO object to the specified bucket as a blob
    blob = bucket.blob(blob_name)
    blob.upload_from_file(buff)

    # Print a message indicating that the file has been uploaded
    print(f"File {blob_name} uploaded to {bucket_name}.")


def download_bytesio_blob(bucket_name, blob_name, project):
    """
    Downloads a blob from Google Cloud Storage as a BytesIO object.

    This function takes in a bucket name, a blob name, and a project name and downloads the blob from the specified
    bucket using the provided project. It returns the downloaded blob as a BytesIO object.

    Args:
        bucket_name: The name of the bucket to download the blob from.
        blob_name: The name of the blob to download.
        project: The name of the Google Cloud project to use.

    Returns:
        The downloaded blob as a BytesIO object.
    """

    # Import the io module and Google Cloud Storage client
    import io

    from google.cloud.storage import Client

    # Create a Google Cloud Storage client
    client = Client(project=project)

    # Get the bucket and blob
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Create a BytesIO object and download the blob to it
    buffer = io.BytesIO()
    blob.download_to_file(buffer)

    # Seek to the beginning of the BytesIO object and return it
    buffer.seek(0)
    return buffer


def gcp_get_secrets(secretmanager_client, secret_path: str) -> dict:
    """

    @param secretmanager_client:
    @param secret_path:
    @return:
    """
    secret_value = secretmanager_client.access_secret_version(name=secret_path)
    secret_json = json.loads(
        secret_value.payload.data.decode("UTF-8").strip().replace("\n", "")
    )
    return secret_json


def create_secret(project_id, secret_id):
    """
    Create a new secret with the given name. A secret is a logical wrapper
    around a collection of secret versions. Secret versions hold the actual
    secret material.
    pip install google-cloud-secret-manager
    """

    # Import the Secret Manager client library.
    from google.cloud import secretmanager_v1beta1 as secretmanager

    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the parent project.
    parent = client.project_path(project_id)

    # Create the secret.
    response = client.create_secret(
        parent,
        secret_id,
        {
            "replication": {
                "automatic": {},
            },
        },
    )

    # Print the new secret name.
    print("Created secret: {}".format(response.name))


def add_secret_version(project_id, secret_id, payload):
    """
    Add a new secret version to the given secret with the provided payload.
    """

    # Import the Secret Manager client library.
    from google.cloud import secretmanager_v1beta1 as secretmanager

    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the parent secret.
    parent = client.secret_path(project_id, secret_id)

    # Convert the string payload into a bytes. This step can be omitted if you
    # pass in bytes instead of a str for the payload argument.
    payload = payload.encode("UTF-8")

    # Add the secret version.
    response = client.add_secret_version(parent, {"data": payload})

    # Print the new secret version name.
    print("Added secret version: {}".format(response.name))


def access_secret_version(project_id, secret_id, version_id="latest"):
    """
    Access the payload for the given secret version if one exists. The version
    can be a version number as a string (e.g. "5") or an alias (e.g. "latest").
    """

    # Import the Secret Manager client library.
    from google.cloud import secretmanager_v1beta1 as secretmanager

    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version.
    name = client.secret_version_path(project_id, secret_id, version_id)

    # Access the secret version.
    response = client.access_secret_version(name)

    # Print the secret payload.
    #
    # WARNING: Do not print the secret in a production environment - this
    # snippet is showing how to access the secret material.
    payload = response.payload.data.decode("UTF-8")
    # print('Plaintext: {}'.format(payload))
    return payload


def create_key(service_account_email):
    """Creates a key for a service account."""

    service = googleapiclient.discovery.build("iam", "v1")

    key = (
        service.projects()
        .serviceAccounts()
        .keys()
        .create(name="projects/-/serviceAccounts/" + service_account_email, body={})
        .execute()
    )

    print("Created key: " + key["name"])
    return key


def dl_gcp_jsonlines_iterable(bucket_name, prefix_dir, breakout=None):
    import io
    from google.cloud import storage
    import jsonlines
    from tqdm import tqdm

    client = storage.Client()

    bucket = client.get_bucket(bucket_name)
    result = []
    for idx, blob in tqdm(enumerate(bucket.list_blobs(prefix=prefix_dir))):
        if idx == 0:
            print(blob.name)
            continue
        buf = io.BytesIO()
        blob.download_to_file(buf)
        buf.seek(0)
        reader = jsonlines.Reader(buf)
        for obj in reader:
            result.append(obj)
        if breakout is not None:
            if breakout == idx:
                break
        # print("File {} downloaded from: {}/{}.".format(blob.name, bucket_name, prefix_dir))
    print(
        "Json Objects: {} downloaded from: {}/{}.".format(
            len(result), bucket_name, prefix_dir
        )
    )
    return result


def pretty_size(
    bytes,
):
    """Get human-readable file sizes.
    simplified version of https://pypi.python.org/pypi/hurry.filesize/
    """
    # bytes pretty-printing
    UNITS_MAPPING = [
        (1 << 50, " PB"),
        (1 << 40, " TB"),
        (1 << 30, " GB"),
        (1 << 20, " MB"),
        (1 << 10, " KB"),
        (1, (" byte", " bytes")),
    ]

    for factor, suffix in UNITS_MAPPING:
        if bytes >= factor:
            break
    amount = int(bytes / factor)

    if isinstance(suffix, tuple):
        singular, multiple = suffix
        if amount == 1:
            suffix = singular
        else:
            suffix = multiple
    return str(amount) + suffix


def bq_query(query, client, dry_run=True):
    # Avoid any heavy fees by recording dry_runs
    job_config = bigquery.QueryJobConfig(dry_run=dry_run, use_query_cache=True)

    # Start the query, passing in the extra configuration.
    query_job = client.query(
        (query),
        job_config=job_config,
    )  # Make an API request.
    # A dry run query completes immediately.
    if dry_run:
        print(
            "This query will process {} bytes.".format(
                pretty_size(query_job.total_bytes_processed)
            )
        )
    else:
        return query_job


def bq_plan_inspect(bquery_job):
    result = []
    for query_plan in bquery_job.query_plan:
        temp_plan = {}
        temp_plan["start"] = query_plan.start
        temp_plan["end"] = query_plan.end
        temp_plan["completed_parallel_inputs    "] = (
            query_plan.completed_parallel_inputs
        )
        temp_plan["compute_ms_avg    "] = query_plan.compute_ms_avg
        temp_plan["compute_ms_max"] = query_plan.compute_ms_max
        temp_plan["compute_ratio_avg"] = query_plan.compute_ratio_avg
        temp_plan["compute_ratio_max"] = query_plan.compute_ratio_max
        temp_plan["entry_id"] = query_plan.entry_id
        temp_plan["input_stages"] = query_plan.input_stages
        temp_plan["name"] = query_plan.name
        temp_plan["parallel_inputs"] = query_plan.parallel_inputs
        temp_plan["read_ms_avg"] = query_plan.read_ms_avg
        temp_plan["read_ms_max"] = query_plan.read_ms_max
        temp_plan["read_ratio_avg"] = query_plan.read_ratio_avg
        temp_plan["read_ratio_max"] = query_plan.read_ratio_max
        temp_plan["records_read"] = query_plan.records_read
        temp_plan["records_written"] = query_plan.records_written
        temp_plan["shuffle_output_bytes"] = query_plan.shuffle_output_bytes
        temp_plan["shuffle_output_bytes_spilled"] = (
            query_plan.shuffle_output_bytes_spilled
        )
        temp_plan["status"] = query_plan.status
        temp_plan["wait_ms_avg"] = query_plan.wait_ms_avg
        temp_plan["wait_ms_max"] = query_plan.wait_ms_max
        temp_plan["wait_ratio_avg"] = query_plan.wait_ratio_avg
        temp_plan["wait_ratio_max"] = query_plan.wait_ratio_max
        temp_plan["write_ms_avg"] = query_plan.write_ms_avg
        temp_plan["write_ms_max"] = query_plan.write_ms_max
        temp_plan["write_ratio_avg"] = query_plan.write_ratio_avg
        temp_plan["write_ratio_max"] = query_plan.write_ratio_max
        temp_plan["steps"] = []
        for step in query_plan.steps:
            temp_step = {}
            temp_step["kind"] = step.kind
            temp_step["substeps"] = step.substeps
            temp_plan["steps"].append(temp_step)
        result.append(temp_plan)
    return result


def upload_local_directory_to_gcs(local_path, bucket, gcs_path):
    assert os.path.isdir(local_path)
    import glob

    for local_file in glob.glob(local_path + "/**"):
        if not os.path.isfile(local_file):
            upload_local_directory_to_gcs(
                local_file, bucket, gcs_path + "/" + os.path.basename(local_file)
            )
        else:
            remote_path = os.path.join(gcs_path, local_file[1 + len(local_path) :])
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)


def upload_from_directory(bucket, directory_path, dest_bucket_name, dest_blob_name):
    rel_paths = glob.glob(directory_path + "/**", recursive=True)

    for local_file in rel_paths:
        remote_path = f'{dest_blob_name}/{"/".join(local_file.split(os.sep)[1:])}'
        if os.path.isfile(local_file):
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_file)


def upload_parquet_to_gcs(
    dataframe, storage_client, bucket_name, file_name, object_name, partition_cols=None
):
    """
    Write a pandas DataFrame to a parquet file and upload it to Google Cloud Storage.

    Parameters:
    - dataframe (pandas DataFrame): the data to be written to the parquet file
    - storage_client (google.cloud.storage.Client): a client for interacting with the GCS API
    - bucket_name (str): the name of the GCS bucket where the file will be uploaded
    - file_name (str): the name to give to the file in GCS
    """
    if partition_cols:
        # Write the DataFrame to a parquet file
        dataframe.to_parquet(file_name, partition_cols=partition_cols)
    else:
        import pyarrow

        dataframe.to_parquet(file_name, index=False)

    # Get a reference to the bucket where you want to upload the file
    bucket = storage_client.get_bucket(bucket_name)

    # Create a new GCS blob with the parquet file as its content
    blob = bucket.blob(object_name)

    try:
        blob.upload_from_filename(file_name)
        os.remove(file_name)

    except Exception as e:
        upload_from_directory(bucket, file_name, bucket_name, object_name)

    # Delete the local copy of the file
    return blob


def save_model_to_gcs(model, bucket_name, file_name):
    """Saves a model to Google Cloud Storage.

    Parameters:
    model (object): The model to save.
    bucket_name (str): The name of the bucket to save the model to.
    file_name (str): The file name to use for the saved model.

    Returns:
    None
    """
    # Create a Storage client
    client = storage.Client()

    # Get the bucket (creating it if it doesn't exist)
    try:
        bucket = client.create_bucket(bucket_name)
    except Exception:
        bucket = client.get_bucket(bucket_name)

    caret.save_model(model, file_name)

    # Save the model to the bucket
    with open(file_name + ".pkl", "rb") as model_file:
        bucket.blob(file_name).upload_from_file(model_file)


def load_model_from_gcs(project, bucket_name, file_name):
    """Loads a model from Google Cloud Storage.

    Parameters:
    bucket_name (str): The name of the bucket where the model is stored.
    file_name (str): The file name of the saved model.

    Returns:
    object: The loaded model.
    """
    from pycaret import classification as caret

    # Create a Storage client
    client = storage.Client(project=project)

    # Get the bucket
    bucket = client.get_bucket(bucket_name)
    # Get the blob (the saved model)
    blob = bucket.get_blob(file_name)

    # Download the model from the bucket to a local file
    blob.download_to_filename(file_name + ".pkl")

    # Load the model from the file
    model = caret.load_model(file_name)

    return model


def encode_save_return_df(df, col, client, use_existing=False, account_name=""):
    """
    Encodes a column in a dataframe and saves the encoder to Google Cloud Storage.

    This function takes in a dataframe, a column name, and a boolean indicating whether to use an existing encoder
    or to create a new one. If a new encoder is to be created, it fits the encoder to the column in the dataframe
    and then saves the encoder to Google Cloud Storage. It then transforms the column using the encoder, creates a
    new dataframe with the encoded column, and returns the new dataframe.

    Args:
        df: A Pandas dataframe containing the column to be encoded.
        col: The name of the column to be encoded.
        use_existing: A boolean indicating whether to use an existing encoder or to create a new one.

    Returns:
        A dataframe containing the encoded column.
    """
    if use_existing:
        # Load OneHotEncoder object from GCS
        enc = load_encoder(client, account_name, f"encoder_{col}")
        encoded_bin = enc.transform(df[[col]])

    else:
        # Create OneHotEncoder object with 'ignore' option for handling unknown values
        enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

        # Fit and transform zi_rev_bin using OneHotEncoder
        encoded_bin = enc.fit_transform(df[[col]])

        # Save OneHotEncoder object to GCS
        save_encoder(enc, client, account_name, f"encoder_{col}")

    # Create dataframe with encoded zi_rev_bin
    try:
        df_res_encoded = pd.DataFrame(encoded_bin, columns=enc.get_feature_names())
    except Exception as e:
        df_res_encoded = pd.DataFrame(encoded_bin, columns=enc.get_feature_names_out())

    return df_res_encoded


def upload_blob_from_bytesio(bucket_name, blob_name, buff, project):
    """
    Uploads a BytesIO object to Google Cloud Storage as a blob using a specified project.

    This function takes in a bucket name, a blob name, a BytesIO object, and a project name and uploads the
    BytesIO object to the specified bucket using the provided project. It then prints a message indicating that
    the file has been uploaded.

    Args:
        bucket_name: The name of the bucket to upload the blob to.
        blob_name: The name of the blob to upload.
        buff: The BytesIO object to be uploaded as a blob.
        project: The name of the Google Cloud project to use.

    Returns:
        None
    """

    # Import the io module and Google Cloud Storage client
    import io

    from google.cloud.storage import Client

    # Create a Google Cloud Storage client
    storage_client = Client(project=project)
    bucket = storage_client.bucket(bucket_name)

    # Upload the BytesIO object to the specified bucket as a blob
    blob = bucket.blob(blob_name)
    blob.upload_from_file(buff)

    # Print a message indicating that the file has been uploaded
    print(f"File {blob_name} uploaded to {bucket_name}.")


# Define save_encoder function
def save_encoder(enc, client, bucket_name, blob_name, account_name):
    """
    Saves a OneHotEncoder object to Google Cloud Storage.

    This function takes in a OneHotEncoder object, a Google Cloud Storage client, a bucket name, and a blob name
    and saves the encoder object to the specified bucket using the provided blob name.

    Args:
        enc: A OneHotEncoder object to be saved.
        client: A Google Cloud Storage client used to access the bucket.
        bucket_name: The name of the bucket to save the encoder object to.
        blob_name: The name of the blob to save the encoder object to.

    Returns:
        None
    """
    # Import OneHotEncoder, google.cloud.storage.Client, and io modules
    from io import BytesIO  # Save OneHotEncoder object to GCS

    from google.cloud.storage import Client
    from joblib import dump
    from sklearn.preprocessing import OneHotEncoder

    enc_bytes = BytesIO()
    dump(enc, enc_bytes)
    enc_bytes.seek(0)
    upload_blob_from_bytesio(bucket_name, blob_name, enc_bytes, account_name)


# Define load_encoder function
def load_encoder(client, bucket_name, blob_name, account_name):
    """
    Loads a OneHotEncoder object from Google Cloud Storage.

    This function takes in a Google Cloud Storage client, a bucket name, and a blob name and loads the OneHotEncoder
    object from the specified bucket and blob.

    Args:
        client: A Google Cloud Storage client used to access the bucket.
        bucket_name: The name of the bucket to load the encoder object from.
        blob_name: The name of the blob to load the encoder object from.

    Returns:
        The loaded OneHotEncoder object.
    """
    from io import BytesIO  # Save OneHotEncoder object to GCS

    from joblib import load

    # Load OneHotEncoder object from GCS
    enc_bytes = download_bytesio_blob(bucket_name, blob_name, account_name)
    enc_bytes.seek(0)
    return load(enc_bytes)


def get_blob_names(bucket, string):
    """Returns a list of the names of the blobs in the provided bucket where the name contains the provided string.

    Args:
      bucket (google.cloud.storage.bucket.Bucket): The bucket to search.
      string (str): The string to search for in the blob names.

    Returns:
      List[str]: The names of the blobs that contain the string.
    """
    # Use a list comprehension to get the names of the blobs that contain the string
    blob_names = [blob.name for blob in bucket.list_blobs() if string in blob.name]

    return blob_names


def query_and_cache(
    client, query, gcp=False, force=False, bucket_name="", account_name=""
):
    """
    Query a client and cache the results of the query in a local or remote location.

    Args:
        client (object): The client to be queried.
        query (str): The query to be executed.
        gcp (bool, optional): Whether to use Google Cloud Platform for caching. Defaults to False.
        force (bool, optional): Whether to force the function to re-query the client and update the cache. Defaults to False.

    Returns:
        pd.DataFrame: The resulting DataFrame object.
    """

    import binascii
    import io
    import pandas as pd

    # Create a query_hash by hashing the query and encoding it as hexadecimal
    query_hash = binascii.hexlify(str(hash(query)).encode()).decode()

    # Check if gcp is False
    if gcp == False:
        # Check if force is True
        if force:
            # Query the client and save the results to a file with the same name as the query_hash
            df = client.query(query).to_dataframe()
            df.to_parquet(f"{str(query_hash)}.plk")
            return df

        # If force is not True, try to read the DataFrame from a file with the same name as the query_hash
        try:
            df = pd.read_parquet(f"{str(query_hash)}.plk")
        # If the file does not exist, query the client and save the results to a file with the same name as the query_hash
        except Exception:
            df = client.query(query).to_dataframe()
            df.to_parquet(f"{str(query_hash)}.plk")

        # Return the DataFrame
        return df

    # If gcp is True, follow a similar process but use the download_bytesio_blob and upload_blob_from_bytesio functions to read from and write to a remote location
    if gcp == True:
        if force:
            df = client.query(query).to_dataframe()
            buff = io.BytesIO()
            df.to_parquet(buff)
            buff.seek(0)
            upload_blob_from_bytesio(bucket_name, query_hash, buff, account_name)
            return df
        try:
            buff = download_bytesio_blob(bucket_name, query_hash, account_name)
            df = pd.read_parquet(buff)
        except Exception:
            df = client.query(query).to_dataframe()
            buff = io.BytesIO()
            df.to_parquet(buff)
            buff.seek(0)
            upload_blob_from_bytes
