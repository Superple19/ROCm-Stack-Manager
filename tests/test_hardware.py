import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rocm_stack_manager.core.hardware import (
    HardwareObservation,
    detected_gfx_targets,
    hardware_from_devices,
    normalize_gfx,
    probe_hardware,
)
from rocm_stack_manager.core.verify import RuntimeObservation


class HardwareTests(unittest.TestCase):
    def test_normalize_gfx_does_not_use_gpu_name(self):
        self.assertEqual(normalize_gfx("gfx1201:sramecc+"), "gfx1201")
        self.assertIsNone(normalize_gfx("AMD Radeon RX 9070 XT"))

    def test_hardware_from_runtime_devices(self):
        observation = hardware_from_devices(
            ({"index": 0, "name": "RX 9070 XT", "gfx": "gfx1201"},),
            source="application-runtime",
            scope="target-runtime",
        )
        self.assertEqual(observation.status, "detected")
        self.assertEqual(observation.gfx_targets, ("gfx1201",))
        self.assertEqual(detected_gfx_targets(observation), ("gfx1201",))

    def test_tool_probe_parses_gfx_and_driver(self):
        completed = subprocess.CompletedProcess(
            ["hipInfo.exe"],
            0,
            stdout="Device 0\nName: gfx1201\nDriver version: 7.15.26312\n",
            stderr="",
        )
        with patch(
            "rocm_stack_manager.core.hardware._tool_candidates",
            return_value=(Path("hipInfo.exe"),),
        ), patch("rocm_stack_manager.core.hardware.subprocess.run", return_value=completed):
            observation = probe_hardware()

        self.assertEqual(observation.status, "detected")
        self.assertEqual(observation.gfx_targets, ("gfx1201",))
        self.assertEqual(observation.driver_version, "7.15.26312")
        self.assertEqual(observation.tool, "hipInfo.exe")

    def test_tool_probe_preserves_missing_tool_state(self):
        with patch("rocm_stack_manager.core.hardware._tool_candidates", return_value=()):
            observation = probe_hardware()

        self.assertEqual(observation.status, "not_detected")
        self.assertIn("not found", observation.error)

    def test_service_prefers_runtime_devices_over_tool_probe(self):
        from rocm_stack_manager.ui.services import ManagerService
        from rocm_stack_manager.core.detection import TargetLayout

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = TargetLayout(root, root, None, (), "source")
            service = ManagerService()
            service.target = target
            runtime = RuntimeObservation(
                target_root=root,
                python_executable=None,
                host_platform="windows",
                runtime_status="detected",
                hardware_status="detected",
                devices=({"gfx": "gfx1201"},),
            )
            with patch("rocm_stack_manager.ui.services.probe_hardware") as fallback:
                observation = service.hardware(runtime)

        fallback.assert_not_called()
        self.assertEqual(observation.source, "application-runtime")
        self.assertEqual(observation.gfx_targets, ("gfx1201",))

    def test_service_can_probe_host_without_target(self):
        from rocm_stack_manager.ui.services import ManagerService

        expected = HardwareObservation(
            scope="host",
            host_platform="windows",
            status="detected",
            gfx_targets=("gfx1201",),
        )
        with patch("rocm_stack_manager.ui.services.probe_hardware", return_value=expected):
            observation = ManagerService().host_hardware()

        self.assertIs(observation, expected)

    def test_hardware_observation_serializes_scope_and_source(self):
        observation = HardwareObservation(
            scope="host",
            host_platform="linux",
            status="not_detected",
            source="rocm-tool",
            error="missing",
        )
        values = observation.as_dict()
        self.assertEqual(values["scope"], "host")
        self.assertEqual(values["source"], "rocm-tool")
        self.assertEqual(values["error"], "missing")
