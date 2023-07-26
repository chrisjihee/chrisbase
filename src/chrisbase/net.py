import logging
from ipaddress import IPv4Address

import netifaces

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
