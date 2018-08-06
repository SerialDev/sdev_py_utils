import boto3
import datetime


class ec2_utils(object):
    def __init__(self, key, secret, region_name="eu-west-1"):

        self.key = key
        self.secret = secret
        self.session = boto3.Session(region_name=region_name,
                                     aws_access_key_id = self.key,
                                     aws_secret_access_key = self.secret,
                                     # aws_session_token=response['Credentials']['SessionToken']
        )

        self.client = self.session.client("ec2")
        self.resource = self.session.resource("ec2")


    def instance_usage_dict(self, i, ):
        instance = {}
        #Info
        instance["info"] = {}
        instance["info"]["id"] = i.id
        instance["info"]["instance_id"] = i.instance_id
        instance["info"]["instance_type"] = i.instance_type
        instance["info"]["instance_lifecycle"] = i.instance_lifecycle
        instance["info"]["key_name"] = i.key_name
        instance["info"]["key_pair.key_fingerprint"] = i.key_pair.key_fingerprint
        instance["info"]["launch_time"] = i.launch_time
        instance["info"]["time_since_launch"] = (datetime.datetime.now() - i.launch_time.replace(tzinfo=None)).total_seconds() # time since launch
        instance["info"]["monitoring"] = i.monitoring
        instance["info"]["architecture"] = i.architecture
        instance["info"]["cpu_options"] = i.cpu_options
        instance["info"]["ebs_optimized"] = i.ebs_optimized
        instance["info"]["ena_support"] = i.ena_support
        instance["info"]["hypervisor"] = i.hypervisor
        instance["info"]["kernel_id"] = i.kernel_id
        instance["info"]["network_interfaces_attribute"] = i.network_interfaces_attribute
        instance["info"]["placement"] = i.placement
        instance["info"]["placement_group.name"] = i.placement_group.name
        instance["info"]["private_dns_name"] = i.private_dns_name
        instance["info"]["private_ip_address"] = i.private_ip_address
        instance["info"]["security_groups"] = i.security_groups
        instance["info"]["state"] = i.state
        instance["info"]["state_reason"] = i.state_reason
        instance["info"]["state_transition_reason"] = i.state_transition_reason
        instance["info"]["tags"] = i.tags
        instance["info"]["vpc_id"] = i.vpc_id
        instance["info"]["iam_instance_profile"] = i.iam_instance_profile
        instance["info"]["elastic_gpu_associations"] = i.elastic_gpu_associations
        instance["info"]["ami_launch_index"] = i.ami_launch_index
        instance["info"]["client_token"] = i.client_token
        instance["info"]["classic_address"] = i.classic_address

        instance["info"]["image"] = {}
        instance["info"]["image"]["description"] = i.image.description
        instance["info"]["image"]["id"] = i.image.id
        instance["info"]["image"]["image_location"] = i.image.image_location
        instance["info"]["image"]["image_owner_alias"] = i.image.image_owner_alias
        instance["info"]["image"]["image_type"] = i.image.image_type
        instance["info"]["image"]["kernel_id"] = i.image.kernel_id
        instance["info"]["image"]["name"] = i.image.name
        instance["info"]["image"]["owner_id"] = i.image.owner_id
        instance["info"]["image"]["platform"] = i.image.platform
        instance["info"]["image"]["root_device_name"] = i.image.root_device_name
        instance["info"]["image"]["root_device_type"] = i.image.root_device_type
        instance["info"]["image"]["state"] = i.image.state
        instance["info"]["image"]["state_reason"] = i.image.state_reason
        instance["info"]["image"]["tags"] = i.image.tags
        instance["info"]["image"]["virtualization_type"] = i.image.virtualization_type

        #actions
        instance["action"] = {}
        instance["action"]["start"] = i.start
        instance["action"]["reboot"] = i.reboot
        instance["action"]["reload"] = i.reload
        instance["action"]["stop"] = i.stop
        instance["action"]["terminate"] = i.terminate
        instance["action"]["unmonitor"] = i.unmonitor
        instance["action"]["monitor"] = i.monitor

        # Locks
        instance["action"]["wait_until_exists"] = i.wait_until_exists
        instance["action"]["wait_until_running"] = i.wait_until_running
        instance["action"]["wait_until_stopped"] = i.wait_until_stopped
        instance["action"]["wait_until_terminated"] = i.wait_until_terminated

        return instance


    def get_instance_types(self,):
        instance_type = {}
        instance_type["general_purpose"] ={}
        instance_type["general_purpose"]["t2"] = {}
        instance_type["general_purpose"]["t2"]["nano"] = "t2.nano"
        instance_type["general_purpose"]["t2"]["micro"] = "t2.micro"
        instance_type["general_purpose"]["t2"]["small"] = "t2.small"
        instance_type["general_purpose"]["t2"]["medium"] = "t2.medium"
        instance_type["general_purpose"]["t2"]["large"] = "t2.large"
        instance_type["general_purpose"]["t2"]["xlarge"] = "t2.xlarge"
        instance_type["general_purpose"]["t2"]["2xlarge"] = "t2.2xlarge"
        instance_type["general_purpose"]["m4"] = {}
        instance_type["general_purpose"]["m4"]["large"] = "m4.large"
        instance_type["general_purpose"]["m4"]["xlarge"] = "m4.xlarge"
        instance_type["general_purpose"]["m4"]["2xlarge"] = "m4.2xlarge"
        instance_type["general_purpose"]["m4"]["4xlarge"] = "m4.4xlarge"
        instance_type["general_purpose"]["m4"]["10xlarge"] = "m4.10xlarge"
        instance_type["general_purpose"]["m4"]["16xlarge"] = "m4.16xlarge"
        instance_type["general_purpose"]["m5"] = {}
        instance_type["general_purpose"]["m5"]["large"] = "m5.large"
        instance_type["general_purpose"]["m5"]["xlarge"] = "m5.xlarge"
        instance_type["general_purpose"]["m5"]["2xlarge"] = "m5.2xlarge"
        instance_type["general_purpose"]["m5"]["4xlarge"] = "m5.4xlarge"
        instance_type["general_purpose"]["m5"]["12xlarge"] = "m5.12xlarge"
        instance_type["general_purpose"]["m5"]["24xlarge"] = "m5.24xlarge"
        instance_type["general_purpose"]["m5d"] = {}
        instance_type["general_purpose"]["m5d"]["large"] = "m5d.large"
        instance_type["general_purpose"]["m5d"]["xlarge"] = "m5d.xlarge"
        instance_type["general_purpose"]["m5d"]["2xlarge"] = "m5d.2xlarge"
        instance_type["general_purpose"]["m5d"]["4xlarge"] = "m5d.4xlarge"
        instance_type["general_purpose"]["m5d"]["12xlarge"] = "m5d.12xlarge"
        instance_type["general_purpose"]["m5d"]["24xlarge"] = "m5d.24xlarge"

        instance_type["compute_optimized"] = {}
        instance_type["compute_optimized"]["c4"] = {}
        instance_type["compute_optimized"]["c4"]["large"] = "c4.large"
        instance_type["compute_optimized"]["c4"]["xlarge"] = "c4.xlarge"
        instance_type["compute_optimized"]["c4"]["2xlarge"] = "c4.2xlarge"
        instance_type["compute_optimized"]["c4"]["4xlarge"] = "c4.4xlarge"
        instance_type["compute_optimized"]["c4"]["8xlarge"] = "c4.8xlarge"
        instance_type["compute_optimized"]["c5"] = {}
        instance_type["compute_optimized"]["c5"]["large"] = "c5.large"
        instance_type["compute_optimized"]["c5"]["xlarge"] = "c5.xlarge"
        instance_type["compute_optimized"]["c5"]["2xlarge"] = "c5.2xlarge"
        instance_type["compute_optimized"]["c5"]["4xlarge"] = "c5.4xlarge"
        instance_type["compute_optimized"]["c5"]["9xlarge"] = "c5.9xlarge"
        instance_type["compute_optimized"]["c5"]["18xlarge"] = "c5.18xlarge"
        instance_type["compute_optimized"]["c5d"] = {}
        instance_type["compute_optimized"]["c5d"]["xlarge"] = "c5d.xlarge"
        instance_type["compute_optimized"]["c5d"]["2xlarge"] = "c5d.2xlarge"
        instance_type["compute_optimized"]["c5d"]["4xlarge"] = "c5d.4xlarge"
        instance_type["compute_optimized"]["c5d"]["9xlarge"] = "c5d.9xlarge"
        instance_type["compute_optimized"]["c5d"]["18xlarge"] = "c5d.18xlarge"

        instance_type["memory_optimized"] = {}
        instance_type["memory_optimized"]["r4"] = {}
        instance_type["memory_optimized"]["r4"]["large"] = "r4.large"
        instance_type["memory_optimized"]["r4"]["xlarge"] = "r4.xlarge"
        instance_type["memory_optimized"]["r4"]["2xlarge"] = "r4.2xlarge"
        instance_type["memory_optimized"]["r4"]["4xlarge"] = "r4.4xlarge"
        instance_type["memory_optimized"]["r4"]["8xlarge"] = "r4.8xlarge"
        instance_type["memory_optimized"]["r4"]["16xlarge"] = "r4.16xlarge"
        instance_type["memory_optimized"]["r5"] = {}
        instance_type["memory_optimized"]["r5"]["large"] = "r5.large"
        instance_type["memory_optimized"]["r5"]["xlarge"] = "r5.xlarge"
        instance_type["memory_optimized"]["r5"]["2xlarge"] = "r5.2xlarge"
        instance_type["memory_optimized"]["r5"]["4xlarge"] = "r5.4xlarge"
        instance_type["memory_optimized"]["r5"]["12xlarge"] = "r5.12xlarge"
        instance_type["memory_optimized"]["r5"]["24xlarge"] = "r5.24xlarge"
        instance_type["memory_optimized"]["r5d"] = {}
        instance_type["memory_optimized"]["r5d"]["large"] = "r5d.large"
        instance_type["memory_optimized"]["r5d"]["xlarge"] = "r5d.xlarge"
        instance_type["memory_optimized"]["r5d"]["2xlarge"] = "r5d.2xlarge"
        instance_type["memory_optimized"]["r5d"]["4xlarge"] = "r5d.4xlarge"
        instance_type["memory_optimized"]["r5d"]["12xlarge"] = "r5d.12xlarge"
        instance_type["memory_optimized"]["r5d"]["24xlarge"] = "r5d.24xlarge"
        instance_type["memory_optimized"]["x1"] = {}
        instance_type["memory_optimized"]["x1"]["16xlarge"] = "x1.16xlarge"
        instance_type["memory_optimized"]["x1"]["32xlarge"] = "x1.32xlarge"
        instance_type["memory_optimized"]["x1e"] = {}
        instance_type["memory_optimized"]["x1e"]["xlarge"] = "x1e.xlarge"
        instance_type["memory_optimized"]["x1e"]["2xlarge"] = "x1e.2xlarge"
        instance_type["memory_optimized"]["x1e"]["4xlarge"] = "x1e.4xlarge"
        instance_type["memory_optimized"]["x1e"]["8xlarge"] = "x1e.8xlarge"
        instance_type["memory_optimized"]["x1e"]["16xlarge"] = "x1e.16xlarge"
        instance_type["memory_optimized"]["x1e"]["32xlarge"] = "x1e.32xlarge"
        instance_type["memory_optimized"]["z1d"] = {}
        instance_type["memory_optimized"]["z1d"]["large"] = "z1d.large"
        instance_type["memory_optimized"]["z1d"]["xlarge"] = "z1d.xlarge"
        instance_type["memory_optimized"]["z1d"]["2xlarge"] = "z1d.2xlarge"
        instance_type["memory_optimized"]["z1d"]["3xlarge"] = "z1d.3xlarge"
        instance_type["memory_optimized"]["z1d"]["6xlarge"] = "z1d.6xlarge"
        instance_type["memory_optimized"]["z1d"]["12xlarge"] = "z1d.12xlarge"

        instance_type["storage_optimized"] = {}
        instance_type["storage_optimized"]["d2"] = {}
        instance_type["storage_optimized"]["d2"]["xlarge"] = "d2.xlarge"
        instance_type["storage_optimized"]["d2"]["2xlarge"] = "d2.2xlarge"
        instance_type["storage_optimized"]["d2"]["4xlarge"] = "d2.4xlarge"
        instance_type["storage_optimized"]["d2"]["8xlarge"] = "d2.8xlarge"
        instance_type["storage_optimized"]["h1"] = {}
        instance_type["storage_optimized"]["h1"]["2xlarge"] = "h1.2xlarge"
        instance_type["storage_optimized"]["h1"]["4xlarge"] = "h1.4xlarge"
        instance_type["storage_optimized"]["h1"]["8xlarge"] = "h1.8xlarge"
        instance_type["storage_optimized"]["h1"]["16xlarge"] = "h1.16xlarge"
        instance_type["storage_optimized"]["i3"] = {}
        instance_type["storage_optimized"]["i3"]["large"] = "i3.large"
        instance_type["storage_optimized"]["i3"]["xlarge"] = "i3.xlarge"
        instance_type["storage_optimized"]["i3"]["2xlarge"] = "i3.2xlarge"
        instance_type["storage_optimized"]["i3"]["4xlarge"] = "i3.4xlarge"
        instance_type["storage_optimized"]["i3"]["8xlarge"] = "i3.8xlarge"
        instance_type["storage_optimized"]["i3"]["16xlarge"] = "i3.16xlarge"
        instance_type["storage_optimized"]["i3"]["metal"] = "i3.metal"

        instance_type["accelerated_computing"] = {}
        instance_type["accelerated_computing"]["f1"] = {}
        instance_type["accelerated_computing"]["f1"]["2xlarge"] = "f1.2xlarge"
        instance_type["accelerated_computing"]["f1"]["16xlarge"] = "f1.16xlarge"
        instance_type["accelerated_computing"]["g3"] = {}
        instance_type["accelerated_computing"]["g3"]["4xlarge"] = "g3.4xlarge"
        instance_type["accelerated_computing"]["g3"]["8xlarge"] = "g3.8xlarge"
        instance_type["accelerated_computing"]["g3"]["16xlarge"] = "g3.16xlarge"
        instance_type["accelerated_computing"]["p2"] = {}
        instance_type["accelerated_computing"]["p2"]["xlarge"] = "p2.xlarge"
        instance_type["accelerated_computing"]["p2"]["8xlarge"] = "p2.8xlarge"
        instance_type["accelerated_computing"]["p2"]["16xlarge"] = "p2.16xlarge"
        instance_type["accelerated_computing"]["p3"] = {}
        instance_type["accelerated_computing"]["p3"]["2xlarge"] = "p3.2xlarge"
        instance_type["accelerated_computing"]["p3"]["8xlarge"] = "p3.8xlarge"
        instance_type["accelerated_computing"]["p3"]["16xlarge"] = "p3.16xlarge"
        return instance_type


    def get_images(self, name="*", ami="*"):
        filter = [{"Name":"name",
                   "Values": [name],
                   "Name":"image-id", "Values":[ami]}]
        images = self.resource.images.filter(Filters=filter)
        for i in images:
            yield i

    def sort_images(self, images):
        image_info = {}
        for i in images:
            image_info[i] = i.creation_date
        images = sorted(image_info, key=image_info.get, reverse=True)
        image_list = []

        for i in images:
            image_info = {}
            image_info["id"] = i.id
            image_info["name"] = i.name
            image_info["architecture"] = i.architecture
            image_info["owner_id"] = i.owner_id
            image_info["creation_date"] = i.creation_date
            image_list.append(image_info)
        return image_list


    def get_keypairs(self, name=""):
        pairs = self.client.describe_key_pairs()['KeyPairs']
        result = []
        for i in pairs:
            if name in i['KeyName']:
                result.append(i['KeyName'])
        return result


    def get_subnet_id(self, name="*", id="*"):
        filter = [{"Name": "tag:Name", "Values":[name],
                   "Name": "subnet-id", "Values":[id]}]
        subnets = []
        for i in self.resource.subnets.filter(Filters=filter):
            subnets.append(i)
        return subnets

    def get_security_group(self, group_name="*"):
        filter = [{"Name":"group-name",
           "Values": [group_name],
           }]
        # security_groups = []
        # for i in self.client.describe_security_groups(Filters = filter):
        #     security_groups.append(i)
        return self.client.describe_security_groups(Filters = filter)



    def get_ec2_instances(self, ):

        # create filter for instances in running state
        filters = [
            {
                'Name': 'instance-state-name',
                'Values': ['running']
            }
        ]

        instances = self.resource.instances.filter(Filters=filters)

        instances = self.resource.instances.filter()
        instances_list = []
        for i in instances:
            instance = self.instance_usage_dict(i)
            instances_list.append(instance)
        return instances_list



    def create_instance(self, keypair_name, instance_tag,
                        instance_type,
                        security_group_name="*XF2IZM76QJCX",
                        subnet_id="*4d*",
                        image_name="*Ubuntu*",
                        image_ami="ami-8a392060",
                        volume_size=200,):

        self.resource.create_instances(
            ImageId = self.sort_images(self.get_images(name=image_name, ami=image_ami))[0]['id'],
            KeyName = self.get_keypairs(keypair_name)[0],
            SecurityGroupIds=[self.get_security_group(security_group_name)['SecurityGroups'][0]['GroupId']],
            SubnetId = self.get_subnet_id(id=subnet_id)[0].id,
            InstanceType = instance_type,
            BlockDeviceMappings = [{"DeviceName": "/dev/sdh",
                                    "VirtualName": "aa_gpu_instance",
                                    "Ebs": {"VolumeSize" : volume_size}}],
            MaxCount = 1,
            MinCount = 1,
            TagSpecifications = [{"ResourceType":"instance", "Tags":[{"Key": "Name", "Value": instance_tag }]}],
            Placement={"AvailabilityZone": "eu-west-1b",},
        )
