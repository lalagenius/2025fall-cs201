# Build LLM在本地 window 运行

Updated 15:23 GMT+8 Aug 12, 2025

2025 summer, Complied by Hongfei Yan





《从零构建大模型》代码，https://github.com/rasbt/LLMs-from-scratch

Build a Large Language Model (From Scratch)



> Windows 10 专业版，版本号22H2，安装日期 2021/6/12



# 用 WSL 安装 Ubuntu

> 可参考：https://mp.weixin.qq.com/s/kWtMuk5WqTfWeqV53TmzcQ



点击开始菜单，搜索“PowerShell”，选择“以管理员身份运行”。

```
wsl --install
```

- 启用 Windows 的“适用于 Linux 的 Windows 子系统”和“虚拟机平台”功能
- 下载并安装最新的 Linux 内核
- 安装默认的 Linux 发行版（通常是 Ubuntu）

按照提示重启机器



**启动 Linux**

在开始菜单搜索“Ubuntu”，点击打开

```
Installing, this may take a few minutes...
Please create a default UNIX user account. The username does not need to match your Windows username.
For more information visit: https://aka.ms/wslusers
Enter new UNIX username: yhf       # 这里可以修改为你起的用户名
New password:
Retype new password:
passwd: password updated successfully
Installation successful!
To run a command as administrator (user "root"), use "sudo <command>".
See "man sudo_root" for details.

Welcome to Ubuntu 24.04.1 LTS (GNU/Linux 6.6.87.2-microsoft-standard-WSL2 x86_64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/pro

 System information as of Tue Aug 12 11:22:20 CST 2025

  System load:  0.0                 Processes:             59
  Usage of /:   0.1% of 1006.85GB   Users logged in:       0
  Memory usage: 4%                  IPv4 address for eth0: 172.20.184.17
  Swap usage:   0%


This message is shown once a day. To disable it please create the
/home/yhf/.hushlogin file.
```

执行完之后会让设置用户和输入密码，按提示操作即可。

运行linux命令看看是否正常，例如：

```
$ ls

$ df -h .
Filesystem      Size  Used Avail Use% Mounted on
/dev/sdd       1007G  1.2G  955G   1% /

$ python --version
Command 'python' not found, did you mean:
  command 'python3' from deb python3
  command 'python' from deb python-is-python3

$ python3
Python 3.12.3 (main, Nov  6 2024, 18:32:19) [GCC 13.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> exit()

$ python3 --version
Python 3.12.3

$ lsb_release -a
No LSB modules are available.
Distributor ID: Ubuntu
Description:    Ubuntu 24.04.1 LTS
Release:        24.04
Codename:       noble


```



# WSL 安装的 Ubuntu 移动到 D 盘

WSL 中安装 Ubuntu 会默认安装到 C 盘的用户数据目录下，在后续使用过程中会导致占用空间越来越大，可以把它迁移到 D 盘。

参考：https://mp.weixin.qq.com/s/p7ByxGLexvTbToilldiSjg?scene=1

先在 D 盘创建好文件夹：一个用于存放导出的备份文件，一个用于存放最终的 WSL 实例。

```
D:\WSL\images
D:\WSL\instances
```



在 PowerShell 中运行以下命令，查看安装的发行版的完整名称以备用：

```
wsl --list --verbose
```

运行结果如下：

```
  NAME      STATE           VERSION
* Ubuntu    Stopped         2
```



**导出备份**

在 PowerShell 中运行以下命令，可以把已安装的发行版导出到指定的备份目录：

```
wsl --export Ubuntu-22.04 D:\WSL\images\ubuntu.tar
```

运行结果如下：

```
正在导出，这可能需要几分钟时间。 (10995 MB)

操作成功完成。
```



**卸载已安装的发行版**

在 PowerShell 中运行以下命令，可以把已安装的发行版从 C 盘中删除：

```
wsl --unregister Ubuntu
```

运行结果如下：

```
正在注销。
操作成功完成。
```



再次执行查看版本的命令，列表中已经没有了：

```
wsl -l -v
```

运行结果如下：

```
适用于 Linux 的 Windows 子系统没有已安装的分发。
可通过安装包含以下说明的分发来解决此问题：

使用“wsl.exe --list --online' ”列出可用的分发
和 “wsl.exe --install <Distro>” 进行安装。
```



**导入备份**

在 PowerShell 中运行以下命令，可以把已导出的备份重新导入到指定目录：

```
wsl --import Ubuntu D:\WSL\instances\Ubuntu D:\WSL\images\ubuntu.tar
```

运行结果如下：

```
操作成功完成。
```



再次执行命令查看版本，可以看到 Ubuntu 又装回来了：

```
wsl -l -v
```

运行结果如下：

```
  NAME      STATE           VERSION
* Ubuntu    Stopped         2
```



并且 D 盘对应目录下也有了文件：

```
D:\WSL\instances\Ubuntu\ext4.vhdx
```





# Build LLM 环境 Setup

## Create a virtual environment

Ubuntu已经自带了python 3.12。安装过程如果报错，可以拷贝下来直接问AI。



**Install uv**

> I highly recommend installing Python packages in a separate virtual environment to avoid modifying system-wide packages that your OS may depend on. To create a virtual environment in the current folder, follow the three steps below.

```
$ sudo apt update
$ sudo apt upgrade -y
$ sudo do-release-upgrade
$ sudo apt install python3-pip

$ sudo apt install pipx
$ $ pipx install uv
$ pipx ensurepath
$ source .bashrc
```



**Create the virtual environment**

**Activate the virtual environment**

```
$ uv venv --python=python3
$ source .venv/bin/activate
```



从 https://github.com/rasbt/LLMs-from-scratch 下载 LLMs-from-scratch-main.zip

```
$ mkdir git
$ cd git
$ git clone https://github.com/rasbt/llMs-from-scratch
$ cd llMs-from-scratch/
```



**Install packages**

```
$ uv pip install packaging
$ uv pip install -r requirements.txt
```



**Finalizing the setup**

That’s it! Your environment should now be ready for running the code in the repository.

Optionally, you can run an environment check by executing the `python_environment_check.py` script in this repostiory:

```
$ python setup/02_installing-python-libraries/python_environment_check.py
```



**Start working with the code**

```
$ jupyter lab
[I 2025-08-12 13:12:05.612 ServerApp] jupyter_lsp | extension was successfully linked.
[I 2025-08-12 13:12:05.617 ServerApp] jupyter_server_terminals | extension was successfully linked.
[I 2025-08-12 13:12:05.620 ServerApp] jupyterlab | extension was successfully linked.
[I 2025-08-12 13:12:05.624 ServerApp] Writing Jupyter server cookie secret to /home/yhf/.local/share/jupyter/runtime/jupyter_cookie_secret
[I 2025-08-12 13:12:06.183 ServerApp] notebook_shim | extension was successfully linked.
[I 2025-08-12 13:12:06.258 ServerApp] notebook_shim | extension was successfully loaded.
[I 2025-08-12 13:12:06.261 ServerApp] jupyter_lsp | extension was successfully loaded.
[I 2025-08-12 13:12:06.262 ServerApp] jupyter_server_terminals | extension was successfully loaded.
[I 2025-08-12 13:12:06.265 LabApp] JupyterLab extension loaded from /home/yhf/.venv/lib/python3.12/site-packages/jupyterlab
[I 2025-08-12 13:12:06.265 LabApp] JupyterLab application directory is /home/yhf/.venv/share/jupyter/lab
[I 2025-08-12 13:12:06.266 LabApp] Extension Manager is 'pypi'.
[I 2025-08-12 13:12:06.407 ServerApp] jupyterlab | extension was successfully loaded.
[I 2025-08-12 13:12:06.408 ServerApp] Serving notebooks from local directory: /home/yhf/git/llMs-from-scratch
[I 2025-08-12 13:12:06.408 ServerApp] Jupyter Server 2.16.0 is running at:
[I 2025-08-12 13:12:06.408 ServerApp] http://localhost:8888/lab?token=ca0a332cc783b6cdc78b3499a7ddc4b089e5e1328abb191d
[I 2025-08-12 13:12:06.408 ServerApp]     http://127.0.0.1:8888/lab?token=ca0a332cc783b6cdc78b3499a7ddc4b089e5e1328abb191d
[I 2025-08-12 13:12:06.408 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2025-08-12 13:12:06.841 ServerApp]

    To access the server, open this file in a browser:
        file:///home/yhf/.local/share/jupyter/runtime/jpserver-8959-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/lab?token=ca0a332cc783b6cdc78b3499a7ddc4b089e5e1328abb191d
        http://127.0.0.1:8888/lab?token=ca0a332cc783b6cdc78b3499a7ddc4b089e5e1328abb191d

```

拷贝给出的链接，在浏览器中打开 http://localhost:8888/lab?token=ca0a332cc783b6cdc78b3499a7ddc4b089e5e1328abb191d

![image-20250812142645241](https://raw.githubusercontent.com/GMyhf/img/main/img/image-20250812142645241.png)



如果局域网其他机器需要访问，在启动的时候加 `--ip=0.0.0.0`（监听所有网络接口）：

```
$ jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

这样它会在 **192.168.1.5:8888** 上提供服务。如果这台机器的局域网ip是 192.168.1.5



```
$ hostname -I
```

172.20.184.17



**用 Windows 端口转发到 WSL**

在 Windows PowerShell（管理员）执行：

```
netsh interface portproxy add v4tov4 listenport=8888 listenaddress=0.0.0.0 connectport=8888 connectaddress=<WSL_IP>
```

`<WSL_IP>` 用你刚才在 WSL 里 `hostname -I` 得到的 IP

这样：

- 任何访问你 Windows 局域网 IP（192.168.1.5:8888）的请求
- 都会被转发到 WSL 的 172.x.x.x:8888

局域网其他机器就可以直接访问：

```
http://192.168.1.5:8888/lab?token=xxxx
```

