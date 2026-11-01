import asyncio
from pathlib import Path
import unittest
from unittest.mock import Mock, mock_open, patch

import asr_server as server


class OwnedDecoderTest(unittest.TestCase):
    def test_v49_cpu_ffmpeg_does_not_reserve_cleanup_gpu(self):
        with (
            patch.object(server, "_external_gpu_users", return_value=[]),
            patch.object(server.subprocess, "check_output", return_value="301 ffmpeg -i input.wav output.wav\n"),
            patch.object(server, "_is_service_process", return_value=False),
        ):
            self.assertEqual(server._watched_gpu_processes(), [])

    def test_v49_only_matching_actual_gpu_executable_is_watched(self):
        processes = [(301, "/usr/lib/jellyfin-ffmpeg/ffmpeg", "141 MiB"),
                     (302, "/usr/bin/other", "200 MiB")]
        with (
            patch.object(server, "_external_gpu_users", return_value=processes),
            patch.object(server, "ASR_GPU_WATCH_PROCESS_PATTERNS", ["ffmpeg"]),
        ):
            self.assertEqual(server._watched_gpu_processes(), [(301, processes[0][1])])

    def test_v49_gpu_query_scoped_to_cleanup_visible_devices(self):
        with (
            patch.object(server, "ASR_LLM_CUDA_VISIBLE_DEVICES", "GPU-cleanup"),
            patch.object(server.subprocess, "check_output", return_value="301, ffmpeg, 141 MiB\n") as query,
        ):
            self.assertEqual(server._gpu_compute_apps(), [(301, "ffmpeg", "141 MiB")])
        self.assertIn("--id=GPU-cleanup", query.call_args.args[0])

    def test_v49_gpu_query_failure_does_not_reserve_devices(self):
        with patch.object(server.subprocess, "check_output", side_effect=server.subprocess.TimeoutExpired("nvidia-smi", 5)):
            self.assertEqual(server._gpu_compute_apps(), [])

    def test_v49_watcher_only_stops_for_actual_shared_gpu_transcode(self):
        for watched in ([], [(301, "ffmpeg")]):
            with (
                self.subTest(watched=watched),
                patch.object(server, "_watched_gpu_processes", return_value=watched),
                patch.object(server, "_llm_server_processes", return_value={456}),
                patch.object(server, "ASR_LLM_IDLE_TIMEOUT", 0),
                patch.object(server, "_stop_llm_server") as stop,
            ):
                server._llm_watcher_tick()
                self.assertEqual(stop.call_count, bool(watched))

    def test_v49_startup_blocks_only_actual_watched_gpu_users(self):
        with (
            patch.object(server, "_llm_health", return_value=False),
            patch.object(server, "_gpu_watch_enabled", return_value=True),
            patch.object(server, "_watched_gpu_processes", return_value=[(301, "ffmpeg")]),
            patch.object(server.subprocess, "Popen") as spawn,
        ):
            with self.assertRaisesRegex(RuntimeError, "GPU reserved"):
                server._start_llm_server()
            spawn.assert_not_called()

    def test_v49_unwatched_gpu_job_does_not_veto_startup(self):
        with (
            patch.object(server, "_llm_process", None),
            patch.object(server, "_llm_health", side_effect=[False, True]),
            patch.object(server, "_gpu_watch_enabled", return_value=True),
            patch.object(server, "_external_gpu_users", return_value=[(301, "other-gpu-job", "200 MiB")]),
            patch.object(server, "ASR_GPU_WATCH_PROCESS_PATTERNS", ["ffmpeg"]),
            patch.object(server, "open", mock_open(), create=True),
            patch.object(server.Path, "write_text"),
            patch.object(server.subprocess, "Popen", return_value=Mock(poll=Mock(return_value=None))) as spawn,
        ):
            server._start_llm_server()
            spawn.assert_called_once()

    def test_v49_gpu_descendants_remain_service_owned(self):
        with (
            patch.object(server, "_llm_server_processes", return_value=set()),
            patch.object(server, "_gpu_compute_apps", return_value=[(301, "ffmpeg", "141 MiB"), (302, "ffmpeg", "141 MiB")]),
            patch.object(server, "_is_service_process", side_effect=lambda pid: pid == 301),
        ):
            self.assertEqual(server._external_gpu_users(), [(302, "ffmpeg", "141 MiB")])

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
