import os
import paramiko
import subprocess

def get_file(server='', username='', password='', destination_data='', source_data=''):
    ssh = paramiko.SSHClient
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
    ssh.connect(server, username=username, password=password)
    sftp = ssh.open_sftp()
    sftp.get(source_data, destination_data)

    print('File downloaded successfully')

def send_file(server='', username='', password='', destination_data='', source_data=''):
    ssh = paramiko.SSHClient
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
    ssh.connect(server, username=username, password=password)
    sftp = ssh.open_sftp()
    sftp.put(source_data, destination_data)

    print('File uploaded successfully')

def execute_command_remotely(server='', username='', password='', command=''):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
    ssh.connect(server, username=username, password=password)
    stdin, stdout, stderr = ssh.exec_command('getconf PATH')
    print('PATH: ' , stdout.readlines())
    print('stderr:', stderr.readlines())
    stdin, stdout, stderr = ssh.exec_command(command)
    print('stdout: ' , stdout.readlines())
    print('stderr:', stderr.readlines())
    ssh.close()
    return stdin, stdout, stderr

def ssh_client(server='', username='', password=''):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
    ssh.connect(server, username=username, password=password)
    return ssh

def command_reader(*args):
    result = ""
    for i in args:
        if i.rstrip().endswith(";"):
            result += (i.strip() + " ")
        else:
            result += (i.strip() +"; ")
    return result

def ssh_commands(ssh, *args):
    # stdin, stdout, stderr = ssh.exec_command('getconf PATH')
    # print('PATH: ' , stdout.readlines())
    # print('stderr:', stderr.readlines())
    command = command_reader(*args)
    stdin, stdout, stderr = ssh.exec_command(command)
    stdout = stdout.readlines()
    stderr = stderr.readlines()
    # print('stdout: ' , stdout)
    # print('stderr:', stderr)
    return stdin, stdout, stderr


def ssh_get_file(ssh, destination_data='', source_data=''):
    sftp = ssh.open_sftp()
    sftp.get(source_data, destination_data)

    print('File downloaded successfully')


def execute_command_locally(bashCommand = ''):
    print (subprocess.Popen(bashCommand, shell=True, stdout=subprocess.PIPE).stdout.read())        
    print('DONE executing local command')        
