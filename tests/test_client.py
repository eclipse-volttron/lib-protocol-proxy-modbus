"""Tests for ModbusClient: request argument filtering, raw and decoded reads, raw and encoded writes, and map
configuration."""
import asyncio

import pytest

from protocol_proxy.protocol.modbus.client import ModbusClient
from protocol_proxy.protocol.modbus.registers import RegisterMap, RegisterSpec

from conftest import FakeDevice, FLOAT_1_0


def run(coro):
    return asyncio.run(coro)


ALL_KWARGS = dict(count=4, device_id=7, no_response_expected=True, read_code=1, object_id=2, records=['rec'], bogus=1)


class TestRequestArguments:
    """Each pymodbus coroutine must receive only the arguments its signature accepts."""

    @pytest.mark.parametrize('register_map, method, expected', [
        ('coil', 'read_coils', {'address': 10, 'count': 4, 'device_id': 7, 'no_response_expected': True}),
        ('discrete_input', 'read_discrete_inputs', {'address': 10, 'count': 4, 'device_id': 7, 'no_response_expected': True}),
        ('holding', 'read_holding_registers', {'address': 10, 'count': 4, 'device_id': 7, 'no_response_expected': True}),
        ('input', 'read_input_registers', {'address': 10, 'count': 4, 'device_id': 7, 'no_response_expected': True}),
        ('device_information', 'read_device_information', {'read_code': 1, 'object_id': 2, 'device_id': 7, 'no_response_expected': True}),
        ('exception_status', 'read_exception_status', {'device_id': 7, 'no_response_expected': True}),
        ('fifo_queue', 'read_fifo_queue', {'address': 10, 'device_id': 7, 'no_response_expected': True}),
        ('file_record', 'read_file_record', {'records': ['rec'], 'device_id': 7, 'no_response_expected': True}),
    ])
    def test_read_coroutines(self, client, device, register_map, method, expected):
        run(client._get_read_coroutine(register_map, 10, **ALL_KWARGS))
        assert device.calls == [(method, expected)]

    @pytest.mark.parametrize('register_map, method', [('holding', 'write_registers'), ('coil', 'write_coils')])
    def test_write_coroutines(self, client, device, register_map, method):
        run(client._get_write_coroutine(register_map, 10, [1, 2], count=2, device_id=7, bogus=1))
        assert device.calls == [(method, {'address': 10, 'values': [1, 2], 'device_id': 7})]

    def test_ignored_arguments_are_logged(self, client, caplog):
        with caplog.at_level('DEBUG'):
            run(client._get_read_coroutine('holding', 10, count=1, bogus=1))
        assert "Ignoring arguments" in caplog.text and "'bogus'" in caplog.text

    def test_unknown_register_map(self, client):
        with pytest.raises(ValueError, match='unknown Modbus register map'):
            client._get_read_coroutine('nope', 0)
        with pytest.raises(ValueError, match='unknown Modbus register map'):
            client._get_write_coroutine('input', 0, [])        # input registers are read-only

    def test_file_record_requires_records(self, client):
        with pytest.raises(ValueError, match="requires a 'records'"):
            client._get_read_coroutine('file_record', 0)


class TestRawRead:
    def test_returns_registers_aligned_with_queries(self, client, device):
        result = run(client.read([(1001, 2), (1010, 1)]))
        assert result == {'results': [FLOAT_1_0, [42]], 'errors': [None, None]}
        assert device.requests() == [(1001, 2), (1010, 1)]   # gap of 7 with max_gap 0: separate requests
        assert device.connections == 1                        # one connection for the batch

    def test_adjacent_and_overlapping_queries_merge(self, client, device):
        # Given out of order and overlapping; results come back per query, in the caller's order.
        result = run(client.read([(1003, 2), (1001, 3), (1004, 1)]))
        assert result == {'results': [[0x4000, 0], [0x3f80, 0, 0x4000], [0]], 'errors': [None, None, None]}
        assert device.requests() == [(1001, 4)]

    def test_max_gap_merges_nearby_queries(self, device):
        client = ModbusClient(device, 'x', max_gap=7)      # 1003-1009 is a gap of exactly 7
        result = run(client.read([(1001, 2), (1010, 1)]))
        assert result == {'results': [FLOAT_1_0, [42]], 'errors': [None, None]}
        assert device.requests() == [(1001, 10)]

    def test_merge_respects_protocol_limit(self, client, device):
        run(client.read([(0, 100), (100, 100)]))
        assert device.requests() == [(0, 100), (100, 100)]
        run(client.read([(0, 60), (60, 60)]))
        assert device.requests()[-1] == (0, 120)

    def test_merged_read_falls_back_per_query_on_error(self, client, device):
        result = run(client.read([(1099, 1), (1100, 2), (1102, 1)]))
        assert result['results'] == [[0], None, [0]]
        assert result['errors'][0] is None and result['errors'][1].isError() and result['errors'][2] is None
        assert device.requests() == [(1099, 4), (1099, 1), (1100, 2), (1102, 1)]

    def test_no_fallback_when_device_does_not_answer(self, device):
        # A timeout means the device is silent; splitting the request would repeat the timeout for each piece.
        device.unreachable = {1100}
        client = ModbusClient(device, 'x')
        result = run(client.read([(1099, 1), (1100, 2), (1102, 1)]))
        assert result['results'] == [None, None, None]
        assert all('No response' in e for e in result['errors'])
        assert device.requests() == [(1099, 4)]                # one attempt, no per-query retries

    def test_different_gateways_run_concurrently(self):
        events = []
        a, b = FakeDevice(delay=0.02, events=events), FakeDevice(delay=0.02, events=events)
        ca, cb = ModbusClient(a, 'a'), ModbusClient(b, 'b')

        async def both():
            await asyncio.gather(ca.read([(1, 1), (5, 1)]), cb.read([(1, 1), (5, 1)]))
        run(both())
        # Both devices' first requests start before either finishes: separate clients do not block each other.
        kinds = [e[0] for e in events[:3]]
        assert kinds.count('start') >= 2, events

    def test_bad_query_rejected(self, client):
        with pytest.raises(ValueError, match='positive count'):
            run(client.read([(1001, 0)]))

    def test_coils_return_bits(self, client):
        result = run(client.read([(3, 4)], register_map='coil'))
        assert result == {'results': [[True, False, False, True]], 'errors': [None]}

    def test_error_response_recorded_per_query(self, client):
        result = run(client.read([(1100, 2), (1010, 1)]))
        assert result['results'] == [None, [42]]
        assert result['errors'][0].isError() and result['errors'][1] is None

    def test_unit_id_is_forwarded(self, client, device):
        run(client.read([(1001, 1)], unit_id=5))
        assert device.calls[0][1]['device_id'] == 5

    def test_exceptions_become_error_strings(self, client, device):
        async def boom(*_, **__):
            raise RuntimeError('link down')
        device.read_holding_registers = boom
        result = run(client.read([(1001, 1)]))
        assert result['results'] == [None] and 'link down' in result['errors'][0]

    def test_connection_is_kept_open_between_requests(self, client, device):
        run(client.read([(1001, 1)]))
        run(client.read([(1001, 1)]))
        run(client.write([(1010, None, [1])]))
        assert device.connections == 1 and device.closed == 0 and device.connected

    def test_reconnects_when_connection_dropped(self, client, device):
        run(client.read([(1001, 1)]))
        device.connected = False                              # e.g., the device closed the socket
        run(client.read([(1001, 1)]))
        assert device.connections == 2

    def test_connection_failure_is_reported_per_query(self, client, device):
        device.connectable = False
        result = run(client.read([(1001, 1), (1010, 1)]))
        assert result['results'] == [None, None]
        assert all('Unable to connect' in e for e in result['errors'])
        assert device.calls == [] and device.connections == 2      # one attempt per request, none reach the device
        result = run(client.write([(1010, None, [1])]))
        assert 'Unable to connect' in result['errors'][0]

    def test_half_open_connection_is_dropped_and_reopened_on_next_request(self, client, device):
        run(client.read([(1001, 1)]))
        device.stale = True                                   # device restarted; our socket is half-open
        failed = run(client.read([(1001, 1)]))
        assert failed['results'] == [None] and 'No response' in failed['errors'][0]
        assert device.closed == 1 and not device.connected    # connection dropped, not retried now
        recovered = run(client.read([(1001, 1)]))
        assert recovered == {'results': [[0x3f80]], 'errors': [None]}
        assert device.connections == 2 and len(device.requests()) == 3

    def test_later_blocks_in_the_same_poll_recover(self, client, device):
        run(client.read([(1001, 1)]))
        device.stale = True
        result = run(client.read([(1001, 1), (1010, 1)]))       # two requests: first fails, second reconnects
        assert result == {'results': [None, [42]], 'errors': [result['errors'][0], None]}
        assert device.connections == 2

    def test_unreachable_device_is_not_retried(self, client, device):
        run(client.read([(1001, 1)]))
        device.stale = True
        device.connectable = False                            # nobody answers a new connection either
        failed = run(client.read([(1001, 1)]))
        assert 'No response' in failed['errors'][0] and len(device.requests()) == 2
        again = run(client.read([(1001, 1)]))
        assert 'Unable to connect' in again['errors'][0] and len(device.requests()) == 2   # no request without a link

    def test_write_failure_drops_connection_too(self, client, device):
        run(client.read([(1001, 1)]))
        device.stale = True
        failed = run(client.write([(1010, None, [5])]))
        assert failed['results'] == [None] and device.closed == 1
        assert run(client.write([(1010, None, [5])]))['errors'] == [None] and device.registers[1010] == 5

    def test_concurrent_calls_are_serialized(self, client, device):
        order = []
        real = device._respond

        def slow_respond(name, bound):
            order.append(('start', bound['address']))
            return real(name, bound)
        device._respond = slow_respond

        async def both():
            await asyncio.gather(client.read([(1001, 1), (1003, 1)]), client.write([(1010, None, [9])]))
        run(both())
        # The read's two requests are not interleaved with the write: one client holds the connection at a time.
        addresses = [a for _, a in order]
        assert addresses in ([1001, 1003, 1010], [1010, 1001, 1003])

    def test_queries_required_without_decode(self, client):
        with pytest.raises(ValueError, match='Queries are required'):
            run(client.read())


class TestDecodedRead:
    def test_requires_configured_map(self, client):
        with pytest.raises(ValueError, match='No register data types'):
            run(client.read(decode=True))

    def test_whole_map_uses_planned_blocks(self, configured_client, device):
        result = run(configured_client.read(decode=True))
        assert result['results'] == {1001: 1.0, 1003: 2.0, 1010: 42, 1201: 7}
        assert list(result['errors']) == [1100]               # the failing int32, keyed by request start
        assert device.requests() == [(1001, 10), (1100, 2), (1201, 1)]

    def test_adjacent_queries_merge_and_decode_straddled_point(self, configured_client, device, caplog):
        # (1001, 2) and (1003, 2) merge into one request, so the float at 1003-1004 is decoded even though
        # neither query covered it alone. Queries are merged, not re-planned from the map.
        result = run(configured_client.read([(1001, 2), (1002, 2), (1004, 1)], decode=True))
        assert result == {'results': {1001: 1.0, 1003: 2.0}, 'errors': {}}
        assert device.requests() == [(1001, 4)]
        assert 'extends outside' not in caplog.text

    def test_partly_covered_point_is_skipped_with_warning(self, configured_client, device, caplog):
        result = run(configured_client.read([(1002, 2)], decode=True))     # cuts through both floats
        assert result == {'results': {}, 'errors': {}}
        assert device.requests() == [(1002, 2)]
        assert caplog.text.count('extends outside the request') == 2

    def test_merged_decode_falls_back_per_query_on_error(self, configured_client, device):
        # 1099 and 1102 are fine; 1100-1101 fail. Merged (1099, 4) fails, so each query is retried alone.
        configured_client.add_registers([(1099, 'uint16'), (1102, 'uint16')])
        device.registers.update({1099: 5, 1102: 6})
        result = run(configured_client.read([(1099, 1), (1100, 2), (1102, 1)], decode=True))
        assert result['results'] == {1099: 5, 1102: 6}
        assert list(result['errors']) == [1100] and result['errors'][1100].isError()
        assert device.requests() == [(1099, 4), (1099, 1), (1100, 2), (1102, 1)]

    def test_planned_read_does_not_fall_back_on_timeout(self, device):
        device.unreachable = {1100}
        client = ModbusClient(device, 'x')
        client.add_registers([(1099, 'uint16'), (1100, 'int32'), (1102, 'uint16')])
        result = run(client.read(decode=True))
        assert result['results'] == {} and list(result['errors']) == [1099]
        assert device.requests() == [(1099, 4)]

    def test_planned_read_falls_back_per_point_on_error(self, client, device):
        client.add_registers([(1099, 'uint16'), (1100, 'int32'), (1102, 'uint16')])   # one contiguous block
        device.registers.update({1099: 5, 1102: 6})
        result = run(client.read(decode=True))
        assert result['results'] == {1099: 5, 1102: 6}
        assert list(result['errors']) == [1100]
        assert device.requests() == [(1099, 4), (1099, 1), (1100, 2), (1102, 1)]

    def test_coils(self, configured_client, device):
        result = run(configured_client.read(decode=True, register_map='coil'))
        assert result == {'results': {3: True, 5: [False, True]}, 'errors': {}}
        assert device.requests() == [(3, 1), (5, 2)]

    def test_decode_failure_is_reported_not_raised(self, configured_client, monkeypatch):
        original = RegisterMap.decode

        def bad_decode(self, block, raw):
            if block.start == 1201:
                raise RuntimeError('corrupt')
            return original(self, block, raw)
        monkeypatch.setattr(RegisterMap, 'decode', bad_decode)
        result = run(configured_client.read(decode=True))
        assert 1001 in result['results'] and 'corrupt' in result['errors'][1201]


class TestWrite:
    def test_raw_write(self, client, device):
        result = run(client.write([(1001, 2, [1, 2])]))
        assert result == {'results': [{'address': 1001, 'count': 2}], 'errors': [None]}
        assert (device.registers[1001], device.registers[1002]) == (1, 2)

    def test_encoded_write(self, configured_client, device):
        result = run(configured_client.write([(1001, None, 2.5), (1010, 1, 9)], encode=True))
        assert result == {'results': [{'address': 1001, 'count': 2}, {'address': 1010, 'count': 1}],
                          'errors': [None, None]}
        assert (device.registers[1001], device.registers[1002], device.registers[1010]) == (0x4020, 0, 9)

    def test_encoded_write_validates_count_and_address(self, configured_client, device):
        result = run(configured_client.write([(1010, 2, 9), (1050, None, 1), (1005, None, 1)], encode=True))
        assert result['results'] == [None, None, None]
        assert 'Count 2 does not match' in result['errors'][0]
        assert 'No holding register configured at address 1050' in result['errors'][1]
        assert 'Pad registers' in result['errors'][2]
        assert device.calls == []                             # nothing reached the device

    def test_encoded_coil_write(self, configured_client, device):
        result = run(configured_client.write([(5, None, [True, False])], register_map='coil', encode=True))
        assert result == {'results': [{'address': 5, 'count': 2}], 'errors': [None]}
        assert (device.coils[5], device.coils[6]) == (True, False)

    def test_encode_requires_configured_map(self, client):
        with pytest.raises(ValueError, match='No register data types'):
            run(client.write([(1, None, 1)], encode=True))


class TestConfiguration:
    def test_add_registers_accepts_specs_dicts_and_tuples(self, client):
        added = client.add_registers([RegisterSpec(1, 'uint16'), {'address': 2, 'data_type': '>f'}, (4, 'uint16')])
        assert [s.address for s in added] == [1, 2, 4]
        assert len(client.get_register_map('holding')) == 3
        assert client.get_register_map('input') is None
        assert client.get_register_map('input', create=True) is not None

    def test_add_registers_rejects_bad_spec_type(self, client):
        with pytest.raises(TypeError):
            client.add_registers([42])

    def test_max_gap_from_client_kwargs(self, device):
        client = ModbusClient(device, 'x', max_gap=6)
        client.add_registers([(1003, '>f'), (1010, 'uint16')])
        assert client.describe_registers()['blocks'] == [[1003, 8]]

    def test_configure_replaces_listed_tables_and_keeps_others(self, configured_client):
        summary = configured_client.configure_registers({'input': [(1, 'uint16'), (2, 'pad', 2), (4, 'uint16')]})
        assert summary == {'configured': {'input': {'specs': 2, 'pads': 1, 'max_gap': 0, 'blocks': [[1, 4]]}},
                           'cleared': []}
        assert {t for _, t in configured_client.register_maps} == {'holding', 'coil', 'input'}

    def test_configure_is_atomic(self, configured_client):
        before = configured_client.register_maps[(1, 'holding')]
        with pytest.raises(ValueError, match='Invalid spec 1 for input table on unit 1'):
            configured_client.configure_registers({'holding': [(1, 'uint16')], 'input': [(1, 'uint16'), (1, 'uint16')]})
        assert configured_client.register_maps[(1, 'holding')] is before
        assert (1, 'input') not in configured_client.register_maps

    def test_configure_clear_others_is_scoped_to_unit(self, configured_client):
        configured_client.add_registers([(5, 'uint16')], register_map='input', unit_id=2)
        summary = configured_client.configure_registers({'holding': [(1001, 'uint16')]}, max_gap=3, clear_others=True)
        assert summary['cleared'] == ['coil']
        assert summary['configured']['holding']['max_gap'] == 3
        assert set(configured_client.register_maps) == {(1, 'holding'), (2, 'input')}

    def test_describe_unknown_map(self, client):
        assert client.describe_registers('input', 3) is None
