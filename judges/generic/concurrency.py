"""Small bounded async scheduling helpers."""

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar


T = TypeVar("T")
R = TypeVar("R")


async def bounded_map(
    items: Iterable[T], work: Callable[[T], Awaitable[R]], concurrency: int
) -> list[R]:
    """Map async work with a fixed in-flight limit and stable result order.

    Once a failure is observed, no more items are scheduled. Already-started
    work is allowed to finish before the first observed failure is re-raised.
    """
    if type(concurrency) is not int or concurrency <= 0:
        raise ValueError("concurrency must be a positive integer")

    values = list(items)
    results: list[R] = [None] * len(values)  # type: ignore[list-item]
    active: dict[asyncio.Task, int] = {}
    next_index = 0

    async def run_one(index: int) -> R:
        return await work(values[index])

    def schedule() -> None:
        nonlocal next_index
        while next_index < len(values) and len(active) < concurrency:
            active[asyncio.create_task(run_one(next_index))] = next_index
            next_index += 1

    schedule()
    while active:
        done, _ = await asyncio.wait(
            active, return_when=asyncio.FIRST_COMPLETED
        )
        failure = None
        for task in sorted(done, key=active.__getitem__):
            index = active.pop(task)
            try:
                results[index] = task.result()
            except BaseException as error:
                if failure is None:
                    failure = error

        if failure is not None:
            if active:
                await asyncio.gather(*active, return_exceptions=True)
                active.clear()
            raise failure
        schedule()

    return results
