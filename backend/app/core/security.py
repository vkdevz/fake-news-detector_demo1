import re
import socket
import ipaddress
from urllib.parse import urlparse
from typing import Tuple

# Disallowed internal/private IP ranges (IPv4 & IPv6)
PRIVATE_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"), # Link-local / cloud metadata
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("::/128"),
    ipaddress.ip_network("fc00::/7"),      # Unique local addresses
    ipaddress.ip_network("fe80::/10"),     # Link-local unicast
    ipaddress.ip_network("2001:db8::/32"), # Documentation
]

ALLOWED_WEB_PORTS = {80, 443, 8080, 8443}

# Patterns attempting prompt injection in untrusted text
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+(instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"you\s+must\s+(declare|classify|label)\s+this\s+(article|claim|news|text)\s+(as\s+)?(true|real|verified|supported)", re.IGNORECASE),
    re.compile(r"system\s*:\s*verdict\s*=", re.IGNORECASE),
    re.compile(r"do\s+not\s+(fact\s*check|verify|analyze)", re.IGNORECASE),
    re.compile(r"override\s+(all\s+)?verification\s+logic", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior\s+guidelines", re.IGNORECASE),
    re.compile(r"developer\s+mode\s*:\s*enabled", re.IGNORECASE),
]

def is_safe_url(url: str) -> Tuple[bool, str]:
    """
    Validates URL to protect against SSRF (Server-Side Request Forgery).
    Prevents requests to localhost, loopback, private RFC1918 subnets, cloud metadata endpoints,
    internal services, and forbidden ports.
    """
    if not url or not isinstance(url, str):
        return False, "URL cannot be empty."

    try:
        parsed = urlparse(url.strip())
        if parsed.scheme.lower() not in ("http", "https"):
            return False, f"Unsupported scheme '{parsed.scheme}'. Only HTTP and HTTPS are permitted."
        
        hostname = parsed.hostname
        if not hostname:
            return False, "Invalid URL: hostname missing."
        
        hostname_clean = hostname.strip("[]").lower()

        # Block known dangerous hostnames
        if hostname_clean in ("localhost", "127.0.0.1", "0.0.0.0", "metadata.google.internal", "instance-data"):
            return False, "Access to localhost or cloud metadata endpoints is forbidden."

        # Check port
        port = parsed.port
        if port is not None and port not in ALLOWED_WEB_PORTS:
            return False, f"Port {port} is forbidden for external verification."

        # Directly check if hostname is an IP literal
        try:
            direct_ip = ipaddress.ip_address(hostname_clean)
            for network in PRIVATE_NETWORKS:
                if direct_ip in network:
                    return False, f"Access to private/local address ({direct_ip}) is forbidden."
        except ValueError:
            pass # Not an IP literal, will resolve via DNS

        # Resolve hostname via getaddrinfo to capture both IPv4 and IPv6 addresses
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            for item in addr_info:
                sockaddr = item[4]
                ip_str = sockaddr[0]
                ip_obj = ipaddress.ip_address(ip_str)
                for network in PRIVATE_NETWORKS:
                    if ip_obj in network:
                        return False, f"Resolved address ({ip_str}) points to a private/local network."
        except socket.gaierror:
            return False, f"Unable to resolve hostname '{hostname}'."

        return True, "URL is safe."
    except Exception as e:
        return False, f"URL validation error: {str(e)}"

def sanitize_untrusted_text(text: str, max_length: int = 50_000) -> str:
    """
    Cleans untrusted web text to defuse potential prompt injection payloads,
    strip dangerous control characters, and enforce maximum length.
    """
    if not text:
        return ""
    
    # Bound input length to protect against memory exhaustion
    bounded = text[:max_length]
    
    cleaned = bounded
    for pattern in PROMPT_INJECTION_PATTERNS:
        cleaned = pattern.sub("[FILTERED_INJECTION_ATTEMPT]", cleaned)
    
    # Strip null bytes and unusual control chars
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', cleaned)
    return cleaned.strip()

def validate_payload_size(content: str, max_chars: int = 100_000) -> Tuple[bool, str]:
    """
    Validates that user content does not exceed allowed payload size to prevent ReDoS / DoS.
    """
    if len(content) > max_chars:
        return False, f"Input payload exceeds maximum allowed size of {max_chars} characters."
    return True, "Payload size valid."
