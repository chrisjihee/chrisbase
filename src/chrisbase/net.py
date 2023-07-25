import json
import logging
import os
from ipaddress import IPv4Address

import httpx
import netifaces

from chrisbase.data import CommonArguments

logger = logging.getLogger(__name__)


def local_ip_addrs():
    for inf in netifaces.interfaces():
        inf_addrs = netifaces.ifaddresses(inf).get(netifaces.AF_INET)
        if inf_addrs:
            for inf_addr in [x.get('addr') for x in inf_addrs]:
                if inf_addr and IPv4Address(inf_addr).is_global:
                    yield inf_addr


ips = sorted(list(local_ip_addrs()))


def num_ip_addrs():
    return len(ips)


def check_ip_addr(ip):
    with httpx.Client(transport=httpx.HTTPTransport(local_address=ip)) as cli:
        res = cli.get("https://api64.ipify.org?format=json", timeout=10.0)
        checked_ip = json.loads(res.text)['ip']
        response = {
            'source': '.'.join(ip.rsplit('.', maxsplit=2)[1:]),
            'status': res.status_code,
            'elapsed': res.elapsed.total_seconds() * 1000,
            'size': res.num_bytes_downloaded / 1024
        }
        logger.info("  * " + ' ----> '.join(map(lambda x: f"[{x}]", [
            f"{response['source']:<7s}",
            f"{response['status']}",
            f"{response['elapsed']:7,.0f}ms",
            f"{response['size']:7,.2f}KB",
            f"Checked IP: {checked_ip:<15s}",
        ])))
        return response


def check_ip_addrs(args: CommonArguments, num_workers=os.cpu_count()):
    assert args.env.output_home, f"Output home is not set"
    logger.info(f"Use {num_workers} workers to check {num_ip_addrs()} IP addresses")
    with (args.env.output_home / f"{args.env.job_name}.jsonl").open("w") as out:
        if num_workers <= 1:
            for ip in ips:
                out.writelines([check_ip_addr(ip), '\n'])
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            pool = ProcessPoolExecutor(max_workers=num_workers)
            jobs = [pool.submit(check_ip_addr, ip=ip) for ip in ips]
            for p in as_completed(jobs):
                out.writelines([json.dumps(p.result(), ensure_ascii=False), '\n'])
