def timeline_label_datapoint(ordinal_string, start_time, end_time):
    return {"timeRange": [start_time, end_time], "val": ordinal_string}


def timeline_group_datapoint(label_name, label_datapoints):
    return {"label": label_name, "data": label_datapoints}


def timeline_chart_datapoint(group_name, group_datapoints):
    return {"group": group_name, "data": group_datapoints}
