import asyncio
from pathlib import Path
import unittest
from unittest.mock import patch

import asr_server as server


class OwnedDecoderTest(unittest.TestCase):
    def test_v44_ignores_decoder_descendant_but_keeps_external_job(self):
        parents = {201: 200, 200: 999, 301: 300, 300: 1}

        def stat(path, *args, **kwargs):
            pid = int(path.parent.name)
            return f"{pid} (process with spaces) S {parents[pid]} 0 0"

        output = "201 ffmpeg -i input.m4a -ar 16000 audio.wav\n301 ffmpeg -hwaccel cuda movie.mkv\n"
        with (
            patch.object(server.os, "getpid", return_value=999),
            patch.object(Path, "read_text", stat),
            patch.object(server.subprocess, "check_output", return_value=output),
            patch.object(server, "ASR_GPU_WATCH_PROCESS_PATTERNS", ["ffmpeg"]),
        ):
            self.assertEqual(server._watched_gpu_processes(), [(301, "ffmpeg -hwaccel cuda movie.mkv")])

    def test_v44_process_races_malformed_stat_and_parent_cycle_are_safe(self):
        for value in [OSError("gone"), "malformed", "123 (bad) S invalid", "123 (loop) S 123"]:
            with self.subTest(value=str(value)), patch.object(server.os, "getpid", return_value=999):
                if isinstance(value, Exception):
                    ctx = patch.object(Path, "read_text", side_effect=value)
                else:
                    ctx = patch.object(Path, "read_text", return_value=value)
                with ctx:
                    self.assertFalse(server._is_service_process(123))

    def test_v44_service_as_container_pid_one_owns_children(self):
        with (
            patch.object(server.os, "getpid", return_value=1),
            patch.object(Path, "read_text", return_value="123 (ffmpeg) S 1 0 0"),
        ):
            self.assertTrue(server._is_service_process(123))


class QueueWakeupTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.semaphore = asyncio.Semaphore(1)
        self.patches = [patch.object(server, "_concurrency_semaphore", self.semaphore),
                        patch.object(server, "ASR_BUSY_WAIT_TIMEOUT", .5)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    async def test_v45_release_wakes_waiter_without_polling_delay(self):
        await server._acquire_concurrency_slot("first")
        waiter = asyncio.create_task(server._acquire_concurrency_slot("second"))
        await asyncio.sleep(0)
        self.assertFalse(waiter.done())
        self.semaphore.release()
        await asyncio.wait_for(waiter, timeout=.1)
        self.semaphore.release()

    async def test_v45_timeout_does_not_steal_next_slot(self):
        await server._acquire_concurrency_slot("first")
        with patch.object(server, "ASR_BUSY_WAIT_TIMEOUT", .01):
            with self.assertRaises(server.HTTPException) as raised:
                await server._acquire_concurrency_slot("second")
        self.assertEqual(raised.exception.status_code, 503)
        self.semaphore.release()
        await server._acquire_concurrency_slot("third")
        self.assertTrue(self.semaphore.locked())
        self.semaphore.release()

    async def test_v45_canceled_waiter_preserves_capacity(self):
        await server._acquire_concurrency_slot("first")
        waiter = asyncio.create_task(server._acquire_concurrency_slot("second"))
        await asyncio.sleep(0)
        waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiter
        self.semaphore.release()
        await server._acquire_concurrency_slot("third")
        self.assertTrue(self.semaphore.locked())
        self.semaphore.release()


if __name__ == "__main__":
    unittest.main()
