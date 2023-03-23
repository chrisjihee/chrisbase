import atexit
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path
from threading import Timer

import requests
from flask import Flask
from tqdm import tqdm

from chrisbase.io import run_command, read_command_out


def fname(ver=None):
    system = platform.system()
    ver_str = '' if not ver or ver < 2 else ver
    if system == "Darwin":
        command = f"ngrok{ver_str}"
    elif system == "Windows":
        command = f"ngrok{ver_str}.exe"
    elif system == "Linux":
        command = f"ngrok{ver_str}"
    else:
        raise Exception(f"{system} is not supported")
    return command


def subdir():
    return "ngrok"


def download_file(url, block_size=1024 * 4):
    url_fname = url.split('/')[-1]
    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    download_path = str(Path(tempfile.gettempdir(), url_fname))
    with tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Download to {download_path}") as p:
        with open(download_path, 'wb') as f:
            for data in r.iter_content(block_size):
                p.update(len(data))
                f.write(data)
    return download_path


def download_ngrok(to, ver=None):
    # print(f"Check file exists: {Path(to, fname(ver=ver))}")
    if Path(to, fname(ver=ver)).exists():
        return
    system = platform.system()
    if system == "Darwin":
        if not ver or ver < 3:
            url = "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-darwin-amd64.zip"
        else:
            url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-darwin-amd64.zip"
    elif system == "Windows":
        if not ver or ver < 3:
            url = "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-windows-amd64.zip"
        else:
            url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-windows-amd64.zip"
    elif system == "Linux":
        if not ver or ver < 3:
            url = "https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip"
        else:
            url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz"
    else:
        raise Exception(f"{system} is not supported")
    download_path = download_file(url)
    print(f"Extract to {to}")
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(to)
    if str(Path(to, fname())) != str(Path(to, fname(ver=ver))):
        print(f"Copy {Path(to, fname())} to {Path(to, fname(ver=ver))}")
        shutil.copy(Path(to, fname()), Path(to, fname(ver=ver)))


def install_ngrok(home=None, ver=None):
    if not home:
        home = tempfile.gettempdir()
    ngrok_dir = str(Path(home, subdir()))
    download_ngrok(ngrok_dir, ver=ver)
    executable = str(Path(ngrok_dir, fname(ver=ver)))
    os.chmod(executable, 0o777)
    return executable


def configure_ngrok(authtoken, home=None, ver=None):
    ngrok_exe = install_ngrok(home=home, ver=ver)
    if not ver or ver < 3:
        command = (ngrok_exe, "authtoken", authtoken)
    else:
        command = (ngrok_exe, "config", "add-authtoken", authtoken)
    run_command(*command)
    return read_command_out(*command).strip().split("file: ")[-1].strip()


def version_ngrok(home=None, ver=None):
    ngrok_exe = install_ngrok(home=home, ver=ver)
    command = (ngrok_exe, "version")
    return read_command_out(*command).strip().split(" ")[-1]


def run_ngrok(port, home=None, ver=None, sleep_sec=3):
    executable = install_ngrok(home, ver=ver)
    print(f"{executable} http {port}")
    ngrok = subprocess.Popen([executable, 'http', str(port)])
    atexit.register(ngrok.terminate)
    localhost_url = "http://localhost:4040/api/tunnels"  # URL with tunnel details
    time.sleep(sleep_sec)

    tunnel_info = requests.get(localhost_url).text  # Get the tunnel information
    j = json.loads(tunnel_info)
    assert 'tunnels' in j and len(j['tunnels']) > 0, f"No tunnel found: {tunnel_info}"
    assert 'public_url' in j['tunnels'][0], f"No public_url found: {tunnel_info}"
    tunnel_url = j['tunnels'][0]['public_url']  # Do the parsing of the get
    return tunnel_url


def new_ngrok(port, home=None, ver=None):
    ngrok_address = run_ngrok(port, home, ver)
    print(f" * Running on {ngrok_address}")
    print(f" * Traffic stats available on http://127.0.0.1:4040")


def run_with_ngrok(flask_app: Flask, home=None, ver=None):
    """
    The provided Flask app will be securely exposed to the public internet via ngrok when run,
    and the its ngrok address will be printed to stdout
    :param flask_app: a Flask application object
    :param home: directory of ngrok
    :param ver: ngrok version
    :return: None
    """
    old_run = flask_app.run

    def new_run(*args, **kwargs):
        port = kwargs.get('port', 5000)
        thread = Timer(1, new_ngrok, args=(port, home, ver))
        thread.daemon = True
        thread.start()
        old_run(*args, **kwargs)

    flask_app.run = new_run
