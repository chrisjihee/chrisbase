import json
from ipaddress import IPv4Address

import httpx
import netifaces


def local_ip_addrs():
    for inf in netifaces.interfaces():
        inf_addrs = netifaces.ifaddresses(inf).get(netifaces.AF_INET)
        if inf_addrs:
            for inf_addr in [x.get('addr') for x in inf_addrs]:
                if inf_addr and IPv4Address(inf_addr).is_global:
                    yield inf_addr


ips = sorted(list(local_ip_addrs()))


def check_ip_addrs():
    for ip in ips:
        with httpx.Client(transport=httpx.HTTPTransport(local_address=ip)) as cli:
            res = cli.get("https://api64.ipify.org?format=json", timeout=10.0)
            checked_ip = json.loads(res.text)['ip']
            response = {
                'source': '.'.join(ip.rsplit('.', maxsplit=2)[1:]),
                'status': res.status_code,
                'elapsed': res.elapsed.total_seconds() * 1000,
                'size': res.num_bytes_downloaded / 1024
            }
            print("  * " + ' ----> '.join(map(lambda x: f"[{x}]", [
                f"{response['source']:<7s}",
                f"{response['status']}",
                f"{response['elapsed']:7,.0f}ms",
                f"{response['size']:7,.2f}KB",
                f"Checked IP: {checked_ip:<15s}",
            ])))
