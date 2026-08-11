import unittest

from rocm_stack_manager.core.platforms import (
    UNSUPPORTED_PLATFORM,
    UnsupportedPlatformError,
    host_platform,
    require_supported_host_platform,
)


class PlatformTests(unittest.TestCase):
    def test_normalizes_supported_platforms(self):
        self.assertEqual(host_platform("win32"), "windows")
        self.assertEqual(host_platform("linux"), "linux")

    def test_unknown_platform_is_not_linux(self):
        self.assertEqual(host_platform("darwin"), UNSUPPORTED_PLATFORM)
        with self.assertRaises(UnsupportedPlatformError):
            require_supported_host_platform("darwin")
