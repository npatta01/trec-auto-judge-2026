import asyncio

import pytest

try:
    from judges.generic.concurrency import bounded_map
except ImportError:
    bounded_map = None


def test_bounded_map_limits_work_and_preserves_input_order():
    assert callable(bounded_map), "bounded_map should be available"

    async def exercise():
        active = 0
        peak = 0

        async def work(value):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep((5 - value) * 0.001)
            active -= 1
            return value * 10

        result = await bounded_map([0, 1, 2, 3, 4], work, concurrency=2)
        return result, peak

    result, peak = asyncio.run(exercise())
    assert result == [0, 10, 20, 30, 40]
    assert peak == 2


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5])
def test_bounded_map_rejects_non_positive_or_non_integer_concurrency(concurrency):
    assert callable(bounded_map), "bounded_map should be available"

    async def exercise():
        async def work(value):
            return value

        with pytest.raises(ValueError):
            await bounded_map([1], work, concurrency=concurrency)

    asyncio.run(exercise())


def test_bounded_map_drains_started_work_and_does_not_start_queued_items_after_error():
    assert callable(bounded_map), "bounded_map should be available"

    async def exercise():
        both_started = asyncio.Event()
        release_sibling = asyncio.Event()
        failed = asyncio.Event()
        drained = asyncio.Event()
        started = []
        error = RuntimeError("synthetic work failure")

        async def work(value):
            started.append(value)
            if len(started) == 2:
                both_started.set()
            if value == 0:
                await both_started.wait()
                failed.set()
                raise error
            if value == 1:
                await release_sibling.wait()
                drained.set()
                return value
            return value

        task = asyncio.create_task(
            bounded_map([0, 1, 2, 3], work, concurrency=2)
        )
        await failed.wait()
        await asyncio.sleep(0)
        assert not task.done(), "the in-flight sibling must be drained"
        assert started == [0, 1], "queued work must not start after failure"

        release_sibling.set()
        with pytest.raises(RuntimeError, match="synthetic work failure") as caught:
            await task
        assert caught.value is error
        assert drained.is_set()
        assert started == [0, 1]

    asyncio.run(exercise())
