#
# Copyright (C) 2026-present ScyllaDB
#
# SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
#

"""
storage_proxy paths that need more than a single-rack, untraced, SIGKILLed
cluster to be reached:

* the batchlog endpoint choice in endpoint_filter(), which returns early
  unless the coordinator sees at least three other live token owners, and the
  "no batchlog endpoint" branch of mutate_atomically;
* the remote Paxos verb handlers when the coordinator sends a trace_info,
  including the learn forwarded to a second replica of a remote DC;
* storage_proxy::abort_batch_writes(), which is only called when the
  batchlog manager is drained during a graceful shutdown.
"""

import asyncio
import logging
import time

import pytest
from cassandra import ConsistencyLevel, Unavailable
from cassandra.cluster import NoHostAvailable
from cassandra.query import SimpleStatement

from test.cluster.util import new_test_keyspace
from test.pylib.scylla_cluster_manager import ScyllaClusterManager
from test.pylib.util import wait_for

logger = logging.getLogger(__name__)


def is_unavailable(e: Exception) -> bool:
    if isinstance(e, Unavailable):
        return True
    # With a single-host load balancing policy the driver's retry of an
    # Unavailable error on the "next host" ends in NoHostAvailable.
    return isinstance(e, NoHostAvailable) and any(isinstance(err, Unavailable) for err in e.errors.values())


def logged_batch(ks: str, table: str, base: int, n: int = 3) -> str:
    inserts = " ".join(f"INSERT INTO {ks}.{table} (pk, ck, v) VALUES ({base + i}, {i}, {base});" for i in range(n))
    return f"BEGIN BATCH {inserts} APPLY BATCH"


async def wait_for_trace_activities(cql, session_id, expected: list[str], exclude_source) -> dict[str, set]:
    """Wait until, for every string in `expected`, system_traces.events of the
    given session holds an activity containing it, recorded by a node other
    than `exclude_source`. Remote nodes flush their trace events in the
    background, so they can show up after the request itself completed."""
    found: dict[str, set] = {}

    async def all_found():
        rows = await cql.run_async(SimpleStatement(
            "SELECT activity, source FROM system_traces.events WHERE session_id = %s",
            consistency_level=ConsistencyLevel.ONE), [session_id])
        found.clear()
        for r in rows:
            for e in expected:
                if e in r.activity and str(r.source) != exclude_source:
                    found.setdefault(e, set()).add(str(r.source))
        if all(e in found for e in expected):
            return True
        logger.info(f"Trace session {session_id}: still missing {[e for e in expected if e not in found]}")
        return None

    await wait_for(all_found, time.time() + 60, period=1)
    return found


async def test_logged_batch_endpoints_over_four_racks(manager: ScyllaClusterManager) -> None:
    """Port of dtest batch_test.py::TestBatch::test_batchlog_manager_issue
    (a reproducer of scylladb/scylladb#3229): logged batches at CL=ALL on a
    4-node cluster with one node per rack. With 3 other live nodes in 3 other
    racks, endpoint_filter() has to choose 2 batchlog endpoints in 2 random
    racks. Then nodes are stopped until no batchlog endpoint is left: a logged
    batch at CL=ONE becomes unavailable, while CL=ANY uses the coordinator
    itself as the batchlog endpoint.
    """
    racks = [{"dc": "dc1", "rack": f"rack{i}"} for i in range(4)]
    servers = await manager.servers_add(4, property_file=racks)
    coordinator = servers[0]

    # RF == number of racks, so the keyspace is rf-rack-valid.
    async with new_test_keyspace(manager, "WITH replication = {'class': 'NetworkTopologyStrategy', 'dc1': 4} "
                                          "AND tablets = {'enabled': false}") as ks:
        cql = manager.get_cql()
        await cql.run_async(f"CREATE TABLE {ks}.tab (pk int, ck int, v int, PRIMARY KEY (pk, ck))")
        # A batch to a CDC-enabled table goes through cdc augmentation first.
        await cql.run_async(f"CREATE TABLE {ks}.cdc_tab (pk int, ck int, v int, PRIMARY KEY (pk, ck)) "
                            "WITH cdc = {'enabled': true}")

        logs = [await manager.server_open_log(s.server_id) for s in servers]
        marks = [await log.mark() for log in logs]

        sessions = [await manager.get_cql_exclusive(s) for s in servers]
        nr_batches = 100
        for i, session in enumerate(sessions):
            await asyncio.gather(*[session.run_async(SimpleStatement(
                    logged_batch(ks, "tab", (i * nr_batches + b) * 10), consistency_level=ConsistencyLevel.ALL))
                for b in range(nr_batches)])
            await session.run_async(SimpleStatement(
                logged_batch(ks, "cdc_tab", i * 10), consistency_level=ConsistencyLevel.ALL))

        rows = await cql.run_async(SimpleStatement(f"SELECT count(*) FROM {ks}.tab",
                                                   consistency_level=ConsistencyLevel.ALL))
        assert rows[0].count == len(servers) * nr_batches * 3

        # A traced logged batch records its batchlog endpoints in the session parameters.
        res = await asyncio.to_thread(sessions[0].execute, SimpleStatement(
            logged_batch(ks, "tab", 100000), consistency_level=ConsistencyLevel.ALL), trace=True)
        trace = await asyncio.to_thread(res.get_query_trace, 30)
        batch_endpoints = trace.parameters.get("batch_endpoints", "")
        logger.info(f"batch_endpoints: {batch_endpoints}")
        assert batch_endpoints.count("/") == 2, f"expected 2 batchlog endpoints, got '{batch_endpoints}'"
        coordinator_host_id = await manager.get_host_id(coordinator.server_id)
        assert str(coordinator_host_id) not in batch_endpoints

        for log, mark in zip(logs, marks):
            for bad in ["unknown endpoint", "fail to connect: connect: Invalid argument"]:
                assert not await log.grep(bad, from_mark=mark), f"unexpected '{bad}' in {log.file}"

        coordinator_cql = sessions[0]

        # Only one other live node left: it is the only batchlog endpoint.
        for s in servers[2:]:
            await manager.server_stop_gracefully(s.server_id)
            await manager.server_not_sees_other_server(coordinator.ip_addr, s.ip_addr)
        await coordinator_cql.run_async(SimpleStatement(logged_batch(ks, "tab", 200000),
                                                        consistency_level=ConsistencyLevel.ONE))

        # No other live node left: there is no batchlog endpoint.
        await manager.server_stop_gracefully(servers[1].server_id)
        await manager.server_not_sees_other_server(coordinator.ip_addr, servers[1].ip_addr)
        try:
            await coordinator_cql.run_async(SimpleStatement(logged_batch(ks, "tab", 300000),
                                                            consistency_level=ConsistencyLevel.ONE))
            pytest.fail("a logged batch at CL=ONE with no live batchlog endpoint should be unavailable")
        except Exception as e:
            assert is_unavailable(e), f"unexpected error: {e!r}"
        # At CL=ANY the coordinator stores the batchlog itself.
        await coordinator_cql.run_async(SimpleStatement(logged_batch(ks, "tab", 400000),
                                                        consistency_level=ConsistencyLevel.ANY))

        for s in servers[1:]:
            await manager.server_start(s.server_id)
        await manager.servers_see_each_other(servers)


async def test_lwt_remote_paxos_handlers_are_traced(manager: ScyllaClusterManager) -> None:
    """Covers the Paxos checks of dtest
    alternator_tests.py::TesterAlternator::test_cluster_traces at the CQL
    level: traced LWT statements send a trace_info with every Paxos verb, so
    the remote replicas open trace sessions of their own and record the
    paxos_prepare / paxos_accept / paxos_prune messages they receive. With a
    second DC, the learn sent to one replica of the remote DC is forwarded by
    that replica to the other one.
    """
    property_file = [{"dc": dc, "rack": rack} for dc in ("dc1", "dc2") for rack in ("rack1", "rack2")]
    servers = await manager.servers_add(4, property_file=property_file)
    coordinator = servers[0]

    async with new_test_keyspace(manager, "WITH replication = {'class': 'NetworkTopologyStrategy', 'dc1': 2, 'dc2': 2} "
                                          "AND tablets = {'enabled': false}") as ks:
        await manager.get_cql().run_async(f"CREATE TABLE {ks}.tab (pk int PRIMARY KEY, v int)")
        cql = await manager.get_cql_exclusive(coordinator)

        statements = [
            SimpleStatement(f"INSERT INTO {ks}.tab (pk, v) VALUES (1, 1) IF NOT EXISTS"),
            SimpleStatement(f"UPDATE {ks}.tab SET v = 2 WHERE pk = 1 IF v = 1"),
            SimpleStatement(f"SELECT * FROM {ks}.tab WHERE pk = 1", consistency_level=ConsistencyLevel.SERIAL),
        ]
        session_ids = []
        for stmt in statements:
            res = await asyncio.to_thread(cql.execute, stmt, trace=True)
            if stmt.query_string.startswith("SELECT"):
                assert res.one().v == 2
            else:
                assert res.one().applied, f"'{stmt.query_string}' was not applied"
            session_ids.append(res.response_future.get_query_trace_ids()[0])

        # The conditional writes run all the Paxos phases on every replica.
        for session_id in session_ids[:2]:
            found = await wait_for_trace_activities(cql, session_id, [
                "paxos_prepare: message received from",
                "paxos_accept: message received from",
                "paxos_prune: message received from",
            ], exclude_source=coordinator.ip_addr)
            logger.info(f"Trace session {session_id}: {found}")
        # A serial read runs at least the prepare phase.
        await wait_for_trace_activities(cql, session_ids[2], ["paxos_prepare: message received from"],
                                        exclude_source=coordinator.ip_addr)


@pytest.mark.skip_mode(mode='release', reason='error injections are not supported in release mode')
async def test_batch_writes_aborted_on_graceful_stop(manager: ScyllaClusterManager) -> None:
    """A graceful stop of a coordinator with logged batch writes in flight:
    the batchlog manager drain must abort them (storage_proxy::abort_batch_writes)
    instead of letting the shutdown wait for them.

    The coordinator's own apply of the batch mutations is blocked with an
    error injection, while the other two replicas acknowledge, so at CL=ALL
    every batch write handler stays alive and waits for a local target only.
    Such handlers survive cancel_nonlocal_write_response_handlers() and the
    CQL server shutdown (whose wait for in-flight requests is cut short by
    request_timeout_on_shutdown_in_seconds, while the write timeout is long),
    and are cancelled when the batchlog manager is drained. More than
    max_cancellations_per_yield (256) handlers make the cancellation loop
    yield with an iterator guard.
    """
    coordinator_config = {
        "write_request_timeout_in_ms": 600000,
        "request_timeout_on_shutdown_in_seconds": 1,
    }
    # A single shard, so that all the batch write handlers are in one storage_proxy.
    cmdline = ["--smp", "1"]
    racks = [{"dc": "dc1", "rack": f"rack{i}"} for i in range(3)]
    coordinator = await manager.server_add(cmdline=cmdline, config=coordinator_config, property_file=racks[0])
    servers = [coordinator] + await manager.servers_add(2, cmdline=cmdline, property_file=racks[1:])

    async with new_test_keyspace(manager, "WITH replication = {'class': 'NetworkTopologyStrategy', 'replication_factor': 3}") as ks:
        await manager.get_cql().run_async(f"CREATE TABLE {ks}.tab (pk int, ck int, v int, PRIMARY KEY (pk, ck))")
        cql = await manager.get_cql_exclusive(coordinator)
        log = await manager.server_open_log(coordinator.server_id)
        mark = await log.mark()

        await manager.api.enable_injection(coordinator.ip_addr, "database_apply", one_shot=False,
                                           parameters={"ks_name": ks, "cf_name": "tab", "what": "wait"})
        nr_batches = 120
        partitions_per_batch = 3
        futures = [cql.run_async(SimpleStatement(logged_batch(ks, "tab", b * 10, partitions_per_batch),
                                                 consistency_level=ConsistencyLevel.ALL))
                   for b in range(nr_batches)]
        try:
            # Every batch mutation is applied locally, so wait until all of them are blocked.
            async def all_writes_blocked():
                blocked = len(await log.grep("database_apply: wait", from_mark=mark))
                return True if blocked >= nr_batches * partitions_per_batch else None
            await wait_for(all_writes_blocked, time.time() + 120, period=0.5)
            assert not any(f.done() for f in futures)

            stop = asyncio.create_task(manager.server_stop_gracefully(coordinator.server_id))
            await log.wait_for("Asked to drain", from_mark=mark, timeout=120)
            await log.wait_for("Drained", from_mark=mark, timeout=120)
        finally:
            # Let the blocked local writes finish, so the database drain doesn't wait for them.
            try:
                await manager.api.message_injection(coordinator.ip_addr, "database_apply")
            except Exception as e:
                logger.info(f"Failed to message the injection, the node is probably down: {e}")

        await stop

        # The clients' requests fail (their connection is closed), none of them hangs.
        results = await asyncio.wait_for(asyncio.gather(*futures, return_exceptions=True), timeout=60)
        failed = [r for r in results if isinstance(r, Exception)]
        logger.info(f"{len(failed)} of {len(results)} batches failed, e.g. {failed[:1]!r}")
        assert len(failed) == len(results)

        await manager.server_start(coordinator.server_id)
        await manager.servers_see_each_other(servers)
