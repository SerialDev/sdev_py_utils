"""Slack support utilitites"""

import requests


def report_to_slack(webhook_url, message):
    data = {"text": message}
    try:
        re = requests.post(webhook_url, data=json.dumps(data))
    except Exception as e:
        print(e)
        return -1
    return re.status_code


def from_sftp(sftp, folder_name, filename):
    import io

    buffer = io.BytesIO()
    with sftp.open(folder_name + "/" + filename, "r") as f:
        buffer.write(f.read())
    buffer.seek(0)
    return buffer


def con_sftp():
    import pysftp

    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None

    sftp = pysftp.Connection(
        host="",
        username="",
        port=2222,
        password="",
        # private_key=".ppk",
        cnopts=cnopts,
    )
    return sftp


def to_sftp(data, sftp, folder_name, filename):
    with sftp.open(folder_name + "/" + filename, "w") as f:
        f.write(data)
    return 1


def list_file_sizes_sftp(sftp, folder_name):
    folders = sftp.listdir(folder_name)
    for i in folders:
        print(i, sftp.lstat(folder_name + "/" + i))
    return 1


def delete_sftp(sftp, folder_name, file_name):
    result = sftp.remove(folder_name + "/" + file_name)
    return result


def phoneNumConverter(phonenum):
    if isinstance(phonenum, str):
        phonenum = re.sub(r"\s+", "", phonenum)  ## muutos
        phonenum = re.sub(r"-", "", phonenum)  ## muutos
        if phonenum[:1] == "0":
            phonenum = "+358" + phonenum[1:]
            return phonenum
        else:
            return phonenum
    else:
        return phonenum
