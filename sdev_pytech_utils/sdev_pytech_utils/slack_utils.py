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
