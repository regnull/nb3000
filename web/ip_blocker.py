import ipaddress
from typing import List, Set

class IPBlocker:
    def __init__(self):
        self.blocked_ranges: Set[ipaddress.IPv4Network] = set()
        self._init_alibaba_ranges()

    def _init_alibaba_ranges(self):
        """Initialize Alibaba Cloud IP ranges"""
        alibaba_ranges = [
            # Alibaba Cloud Mainland China
            "47.82.0.0/15",
            "47.88.0.0/14",
            "47.89.0.0/16",
            "47.90.0.0/15",
            "47.92.0.0/14",
            "47.96.0.0/11",
            "47.112.0.0/13",
            "47.120.0.0/13",
            "47.128.0.0/11",
            "47.144.0.0/14",
            "47.148.0.0/14",
            "47.152.0.0/13",
            "47.160.0.0/12",
            "47.176.0.0/12",
            "47.192.0.0/13",
            "47.200.0.0/13",
            "47.208.0.0/12",
            "47.224.0.0/13",
            "47.232.0.0/13",
            "47.240.0.0/13",
            "47.248.0.0/13",
            "47.252.0.0/14",
            "47.254.0.0/16",
            # Alibaba Cloud International
            "8.34.208.0/20",
            "8.35.192.0/21",
            "8.35.200.0/23",
            "8.35.202.0/24",
            "8.35.203.0/24",
            "8.35.204.0/22",
            "8.35.208.0/20",
            "8.35.224.0/19",
            "8.36.0.0/18",
            "8.36.64.0/19",
            "8.36.96.0/21",
            "8.36.104.0/21",
            "8.36.112.0/20",
            "8.36.128.0/19",
            "8.36.192.0/18",
            "8.37.0.0/17",
            "8.37.128.0/20",
            "8.37.144.0/20",
            "8.37.160.0/21",
            "8.37.168.0/21",
            "8.37.176.0/20",
            "8.37.192.0/19",
            "8.37.224.0/19",
            "8.38.0.0/17",
            "8.38.128.0/18",
            "8.38.192.0/19",
            "8.38.224.0/19",
            "8.39.0.0/18",
            "8.39.64.0/19",
            "8.39.96.0/19",
            "8.39.128.0/17",
            "8.40.0.0/14",
            "8.40.4.0/22",
            "8.40.8.0/21",
            "8.40.16.0/20",
            "8.40.32.0/19",
            "8.40.64.0/18",
            "8.40.128.0/17",
            "8.41.0.0/16",
            "8.42.0.0/15",
            "8.44.0.0/14",
            "8.45.0.0/16",
            "8.46.0.0/15",
            "8.48.0.0/13",
            "8.56.0.0/14",
            "8.60.0.0/15",
            "8.62.0.0/16",
            "8.63.0.0/16",
            "8.64.0.0/14",
            "8.68.0.0/14",
            "8.72.0.0/13",
            "8.80.0.0/12",
            "8.96.0.0/13",
            "8.104.0.0/13",
            "8.112.0.0/13",
            "8.120.0.0/13",
            "8.128.0.0/10",
            "8.192.0.0/10",
            "8.208.0.0/12",
            "8.224.0.0/12",
            "8.240.0.0/12",
            "8.248.0.0/13",
            "8.252.0.0/14",
            "8.254.0.0/15",
            "8.255.0.0/16",
            "8.255.128.0/17",
            "8.255.192.0/18",
            "8.255.224.0/19",
            "8.255.240.0/20",
            "8.255.248.0/21",
            "8.255.252.0/22",
            "8.255.254.0/23",
            "8.255.255.0/24",
            "8.255.255.128/25",
            "8.255.255.192/26",
            "8.255.255.224/27",
            "8.255.255.240/28",
            "8.255.255.248/29",
            "8.255.255.252/30",
            "8.255.255.254/31",
            "8.255.255.255/32",
            # Add more ranges as needed
        ]
        
        for range_str in alibaba_ranges:
            try:
                network = ipaddress.IPv4Network(range_str)
                self.blocked_ranges.add(network)
            except ValueError as e:
                print(f"Warning: Invalid IP range {range_str}: {e}")

    def is_blocked(self, ip: str) -> bool:
        """
        Check if an IP address is in any of the blocked ranges
        
        Args:
            ip: IP address to check (e.g., "1.2.3.4")
            
        Returns:
            bool: True if IP is blocked, False otherwise
        """
        try:
            ip_addr = ipaddress.IPv4Address(ip)
            return any(ip_addr in network for network in self.blocked_ranges)
        except ValueError:
            # Invalid IP address format
            return False

    def add_range(self, range_str: str) -> bool:
        """
        Add a new IP range to block
        
        Args:
            range_str: IP range in CIDR notation (e.g., "1.2.3.0/24")
            
        Returns:
            bool: True if range was added successfully, False otherwise
        """
        try:
            network = ipaddress.IPv4Network(range_str)
            self.blocked_ranges.add(network)
            return True
        except ValueError:
            return False

    def remove_range(self, range_str: str) -> bool:
        """
        Remove an IP range from the blocked list
        
        Args:
            range_str: IP range in CIDR notation (e.g., "1.2.3.0/24")
            
        Returns:
            bool: True if range was removed successfully, False otherwise
        """
        try:
            network = ipaddress.IPv4Network(range_str)
            if network in self.blocked_ranges:
                self.blocked_ranges.remove(network)
                return True
            return False
        except ValueError:
            return False

    def get_blocked_ranges(self) -> List[str]:
        """
        Get list of all blocked IP ranges
        
        Returns:
            List[str]: List of blocked IP ranges in CIDR notation
        """
        return [str(network) for network in sorted(self.blocked_ranges)] 