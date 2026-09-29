"""Modbus plugin for protocol_proxy.

Proxies are launched with ``python -m protocol_proxy.proxy <module>:<class>`` (see protocol_proxy.proxy.launch), so
this package may import its proxy class directly. ModbusProxy declares no LAUNCHER: it takes no protocol-specific
launch options, since all client settings arrive via REGISTER_DEVICE.
"""
import logging

from .client import ModbusClient
from .modbus_proxy import ModbusProxy

__all__ = ['ModbusClient', 'ModbusProxy', 'PROXY_CLASS', 'run_modbus_device']

PROXY_CLASS = ModbusProxy

_log = logging.getLogger(__name__)


async def run_modbus_device(device_address, transport_protocol: str = 'tcp', **kwargs):
    """Convenience for creating a stand-alone ModbusClient (e.g., from a script or shell)."""
    return await ModbusClient.create(device_address, transport_protocol, **kwargs)
