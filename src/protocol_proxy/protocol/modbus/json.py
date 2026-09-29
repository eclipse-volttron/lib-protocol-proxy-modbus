import ipaddress
import json
import logging

from enum import Enum


_log = logging.getLogger(__name__)

def _jsonable(val):
    """Recursively convert a value into JSON-serializable primitives.

    Dict keys become strings (JSON requires it), enums become their names, and anything else which is not a JSON
    primitive (e.g., pymodbus exception responses) is rendered with str().
    """
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, dict):
        return {str(k): _jsonable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in val]
    if isinstance(val, Enum):
        return val.name
    if isinstance(val, (bytes, bytearray)):
        return val.hex()
    if isinstance(val, (ipaddress.IPv4Address, ipaddress.IPv6Address)):
        return str(val)
    return str(val)

# TODO: When can we handle an error, rather than just serializing it?
def _serialize(val):
    """Split a client response into JSON-serializable (results, errors).

    Client methods return {'results': ..., 'errors': ...}; anything else is treated as a result with no errors.
    """
    if isinstance(val, dict) and set(val) == {'results', 'errors'}:
        return _jsonable(val['results']), _jsonable(val['errors'])
    return _jsonable(val), {}

def serialize(val):
    ret_val, err_val = {}, {}
    try:
        ret_val, err_val = _serialize(val)
    except Exception as e:
        _log.exception(f"When exception occurred, ret_val had been: {ret_val}")
        try:
            raw_str = str(val)
        except Exception:
            raw_str = object.__repr__(val)
        err_val = {
            "error": "SerializationError",
            "details": str(e),
            "raw_type": str(type(val)),
            "raw_str": raw_str
        }
    ret_val = {'result': ret_val, 'error': err_val}
    return json.dumps(ret_val).encode('utf8')
