import json
import logging
from ipaddress import IPv4Address

import httpx
import netifaces

from chrisbase.data import CommonArguments
from chrisbase.util import MongoDB

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


def check_ip_addr(ip, _id=None):
    with httpx.Client(transport=httpx.HTTPTransport(local_address=ip)) as cli:
        res = cli.get("https://api64.ipify.org?format=json", timeout=10.0)
        checked_ip = json.loads(res.text)['ip']
        response = {
            'source': '.'.join(ip.rsplit('.', maxsplit=2)[1:]),
            'status': res.status_code,
            'elapsed': round(res.elapsed.total_seconds(), 3),
            'size': round(res.num_bytes_downloaded / 1024, 6),
        }
        if _id:
            response['_id'] = _id
        logger.info("  * " + ' ----> '.join(map(lambda x: f"[{x}]", [
            f"{response['source']:<7s}",
            f"{response['status']}",
            f"{response['elapsed'] * 1000:7,.0f}ms",
            f"{response['size']:7,.2f}KB",
            f"Checked IP: {checked_ip:<15s}",
        ])))
        return response


def check_ip_addrs(args: CommonArguments, mongo: MongoDB | None = None):
    logger.info(f"Use {args.env.max_workers} workers to check {num_ip_addrs()} IP addresses")
    if args.env.max_workers < 2:
        for i, ip in enumerate(ips):
            res = check_ip_addr(ip=ip, _id=i + 1)
            if mongo:
                mongo.table.insert_one(res)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        pool = ProcessPoolExecutor(max_workers=args.env.max_workers)
        jobs = [pool.submit(check_ip_addr, ip=ip, _id=i + 1) for i, ip in enumerate(ips)]
        for job in as_completed(jobs):
            if mongo:
                mongo.table.insert_one(job.result())
