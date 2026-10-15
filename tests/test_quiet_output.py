import asyncio
import io
import sys

import judges.generic.private_io as private_io


def test_quiet_output_keeps_shared_sink_open_until_last_overlapping_exit():
    quiet_output = getattr(private_io, "quiet_output", None)
    assert callable(quiet_output), "quiet_output should be available"

    original_stdout, original_stderr = sys.stdout, sys.stderr
    sentinel_stdout, sentinel_stderr = io.StringIO(), io.StringIO()
    sys.stdout, sys.stderr = sentinel_stdout, sentinel_stderr

    async def exercise():
        entered = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]

        async def call(index):
            with quiet_output():
                entered[index].set()
                await release[index].wait()
                assert not sys.stdout.closed
                assert not sys.stderr.closed

        first = asyncio.create_task(call(0))
        await entered[0].wait()
        shared_stdout, shared_stderr = sys.stdout, sys.stderr
        second = asyncio.create_task(call(1))
        await entered[1].wait()
        assert sys.stdout is shared_stdout
        assert sys.stderr is shared_stderr

        # The first context exits while the second is still active.
        release[0].set()
        await first
        assert sys.stdout is shared_stdout
        assert sys.stderr is shared_stderr
        assert not shared_stdout.closed
        assert not shared_stderr.closed

        release[1].set()
        await second
        assert sys.stdout is sentinel_stdout
        assert sys.stderr is sentinel_stderr
        assert shared_stdout.closed
        assert shared_stderr.closed

    try:
        asyncio.run(exercise())
    finally:
        sys.stdout, sys.stderr = original_stdout, original_stderr
