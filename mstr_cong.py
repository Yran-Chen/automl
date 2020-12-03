import os
import stat
import paramiko
import traceback

from pathlib import Path


class Paramiko_SSH(object):

    def __init__(self, ip, port=22, username=None, password=None, timeout=30):
        self.ip = ip  # ssh远程连接的服务器ip
        self.port = 22  # ssh的端口一般默认是22，
        self.username = username  # 服务器用户名
        self.password = password  # 密码
        self.timeout = timeout  # 连接超时

        # paramiko.SSHClient() 创建一个ssh对象，用于ssh登录以及执行操作
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # paramiko.Transport()创建一个文件传输对象，用于实现文件的传输
        self.t = paramiko.Transport(sock=(self.ip, self.port))

    def _password_connect(self):
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(hostname=self.ip, port=22, username=self.username, password=self.password)
        self.t.connect(username=self.username, password=self.password)

    def close(self):
        self.t.close()  # 断开文件传输的连接
        self.ssh.close()  # 断开ssh连接

    def execute_cmd(self, cmd):

        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        res, err = stdout.read(), stderr.read()
        result = res if res else err

        return result.decode()

    # 从远程服务器获取文件到本地
    def _sftp_get(self, remotefile, localfile):

        sftp = paramiko.SFTPClient.from_transport(self.t)
        sftp.get(remotefile, localfile)

    # 从本地上传文件到远程服务器
    def _sftp_put(self, localfile, remotefile):

        sftp = paramiko.SFTPClient.from_transport(self.t)
        sftp.put(localfile, remotefile)

    # 递归遍历远程服务器指定目录下的所有文件
    def _get_all_files_in_remote_dir(self, sftp, remote_dir):
        all_files = list()
        if remote_dir[-1] == '/':
            remote_dir = remote_dir[0:-1]

        files = sftp.listdir_attr(remote_dir)
        for file in files:
            filename = remote_dir + '/' + file.filename

            if stat.S_ISDIR(file.st_mode):  # 如果是文件夹的话递归处理
                all_files.extend(self._get_all_files_in_remote_dir(sftp, filename))
            else:
                all_files.append(filename)

        return all_files

    def sftp_get_dir(self, remote_dir, local_dir):
        try:

            sftp = paramiko.SFTPClient.from_transport(self.t)

            all_files = self._get_all_files_in_remote_dir(sftp, remote_dir)

            for file in all_files:

                local_filename = file.replace(remote_dir, local_dir).replace("/", '\\')
                local_filepath = os.path.dirname(local_filename).replace("/", '\\')
                print(local_filename)

                if not os.path.exists(local_filepath):
                    os.makedirs(local_filepath)

                sftp.get(file, local_filename)
        except:
            print('ssh get dir from master failed.')
            print(traceback.format_exc())

    # 递归遍历本地服务器指定目录下的所有文件
    def _get_all_files_in_local_dir(self, local_dir):
        all_files = list()

        for root, dirs, files in os.walk(local_dir, topdown=True):
            for file in files:
                filename = os.path.join(root, file)
                all_files.append(filename)

        return all_files

    def sftp_put_dir(self, local_dir, remote_dir):
        try:
            sftp = paramiko.SFTPClient.from_transport(self.t)

            if remote_dir[-1] == "/":
                remote_dir = remote_dir[0:-1]

            all_files = self._get_all_files_in_local_dir(local_dir)
            for file in all_files:

                remote_filename = file.replace(local_dir, remote_dir).replace("\\", '/')
                remote_path = os.path.dirname(remote_filename).replace("\\", '/')
                print(remote_filename)
                try:
                    sftp.stat(remote_path)
                    print('stat remote path succ.')
                except:
                    # os.popen('mkdir -p %s' % remote_path)
                    self.execute_cmd('mkdir -p %s' % remote_path)  # 使用这个远程执行命令
                    print('mkdir succ.')
                sftp.put(file, remote_filename)

        except:
            print('ssh get dir from master failed.')
            print(traceback.format_exc())


def assign_task(ipadress, username, password, cmd, localsrc=None, remotesrc=None):

        result = None
        hostname = ipadress
        miko_p = Paramiko_SSH(hostname, username=username, password=password)
        miko_p._password_connect()

        """
        For each task.
        """
        if (localsrc is not None) and (remotesrc is not None):
            print('Uploading...')
            try:
                miko_p._sftp_put(localfile=localsrc, remotefile=remotesrc)
            except:
                miko_p.sftp_put_dir(local_dir=localsrc, remote_dir=remotesrc)

        if cmd is not None:
            result = miko_p.execute_cmd(cmd)

            print('#' * 42)
            print('IP: {} \n'.format(hostname))
            print(result)

        miko_p.close()
        return result


def get_log_file(ipadress, username, password, localsrc, remotesrc, cmd=None):
        hostname = ipadress
        print(hostname,username,password)
        miko_p = Paramiko_SSH(hostname, username=username, password=password)
        miko_p._password_connect()

        """
        For each task.
        """
        if cmd is not None:
            print('#' * 42)
            print('IP: {} \n'.format(hostname))
            result = miko_p.execute_cmd(cmd)
            print(result)

        if (localsrc is not None) and (remotesrc is not None):
            print('Syncing log file...')
            # assign different file dir according to ip address.
            try:
                miko_p.sftp_get_dir(
                    local_dir=os.path.join(localsrc, "log_{}".format(hostname.split('.')[-1])),
                    remote_dir=remotesrc)
            except:
                print('wrong dir.')

        miko_p.close()


def get_save_file(ipadress, username, password, localsrc, remotesrc, cmd=None):
    hostname = ipadress
    print(hostname, username, password)
    miko_p = Paramiko_SSH(hostname, username=username, password=password)
    miko_p._password_connect()

    """
    For each task.
    """
    if cmd is not None:
        print('#' * 42)
        print('IP: {} \n'.format(hostname))
        result = miko_p.execute_cmd(cmd)
        print(result)

    if (localsrc is not None) and (remotesrc is not None):
        print('Syncing savedata file...')
        # assign different file dir according to ip address.
        try:
            miko_p.sftp_get_dir(
                local_dir=localsrc,
                remote_dir=remotesrc)
        except:
            print('wrong dir.')

    miko_p.close()

"""
def run_main_prog(ipadress,username,password,port):

    for i,hostname in enumerate(ipadress):

        miko_p = Paramiko_SSH(hostname,username=username,password=password)
        miko_p._password_connect()

        # sshcmd = 'python /data/!workspace/automl/main.py --op={}'.format(OP_DICT[i])
        sshcmd = 'ls /data/!workspace'


        print ('#' * 42)
        print('IP: {} \n'.format(hostname))
        print(result)

        miko_p.close()
        
def sync_main_file(ipadress,username,password,localsrc,remotesrc):

    for i,hostname in enumerate(ipadress):
        put_file(hostname,username,password,localsrc,remotesrc)

def put_file(hostname,username,passwd,localfile,romotefile):
    try:
        transfer = paramiko.Transport((hostname,22))
        transfer.connect(username=username,password=passwd)
        sftp = paramiko.SFTPClient.from_transport(transfer)
    except Exception as E:
        print('出现错误:',E)
    else:
        sftp.put(localfile,romotefile)
        print('上传成功!')
    finally:
        sftp.close()
"""


PROCESS_NUM = 4

if __name__ == '__main__':
    # '172.16.10.119',  '172.16.10.117', '172.16.10.116'
    ipadress = ['172.16.10.119', '172.16.10.117', '172.16.10.116']
    ipadress_kudu = ['172.16.8.107','172.16.8.108','172.16.8.109']
    username = 'root'
    password = 'DrpEco@2020'
    password_kudu = 'flink123'
    port = 22

    num = 0

    OP_DICT = ['zscore', 'cbrt', 'sigmoid', 'stdscaler']
    OP_DICT_NEW = ['zscore', 'sigmoid', 'stdscaler' ]
    OP_DICT_1019 = ['stdscaler', 'cbrt', 'sigmoid', 'freq']
    OP_DICT_1020 = ['freq']
    OP_DICT_1028 = ['cbrt','sigmoid']

    from multiprocessing import Process, Pool
    pool = Pool(processes=PROCESS_NUM)

    train_119_cmd = 'nohup python /data/!workspace/automl/main_remote.py ' \
                    '--op=freq --selected=! --percent=1.0 --name=lr_beta_5000 ' \
                    '> test.log  2>&1 &'

    train_117_cmd = 'nohup python /data/!workspace/automl/main_remote.py ' \
                    '--op=cbrt --selected=! --percent=1.0 --name=lr_beta_5000 ' \
                    '> test.log  2>&1 &'


    pool.apply_async(assign_task, ('172.16.10.119', username, password, train_119_cmd,))
    pool.apply_async(assign_task, ('172.16.10.117', username, password, train_117_cmd,))

    for i, _task_ipadress in enumerate(ipadress) :

        # cmd for server.
        # sshcmd = 'ls'
        train_cmd = 'nohup python /data/!workspace/automl/main_remote.py ' \
                 '--op={0} --selected=! --percent=1.0 --name=lr_beta_5000 ' \
                    '> test.log  2>&1 &'.format(OP_DICT_NEW[i])



        # train_test_cmd = 'python /data/!workspace/automl/main_remote.py --op={0} --selected=!f_00470 --percent=1.0 --name=bug_1016_multipool_changedfile'.format(OP_DICT[i])
        # pool.apply_async(assign_task, (_task_ipadress, username, password, train_cmd,))

        #rm cache. (test)
        # sshcmd = 'rm -rf /data/!workspace/Savefile/lr_beta'
        # pool.apply_async(assign_task, (_task_ipadress, username, password, sshcmd,))

        # fetch log file.
        # pool.apply_async(get_log_file, (_task_ipadress, username, password, r"D:\!DTStack\Logfile_remote",'/data/!workspace/Logfile'))
        # pool.apply_async(get_save_file, (_task_ipadress, username, password, r"D:\!DTStack\Savefile_remote\lr_beta", '/data/!workspace/Savefile/lr_beta'))

        # sync main.py
        sshcmd = ''
        # pool.apply_async(assign_task, (_task_ipadress, username, password, sshcmd, r"D:\!DTStack\automl",'/data/!workspace/automl'))

        print('msg sent to {}.'.format(_task_ipadress))
    pool.close()
    pool.join()


    for _task_i in ipadress_kudu:
        pass
       # for kudu task.
       #  assign_task(ipadress_kudu,username,password_kudu,localsrc=r'D:\!Downloads\Compressed\ap1013.zip',remotesrc='/root/ap1013.zip')
        # assign_task(ipadress_kudu, username, password_kudu,cmd='unzip  /root/ap4.zip')
       #  assign_task(ipadress_kudu, username, password_kudu,cmd='/data/anaconda3/bin/python setup.py sdist')