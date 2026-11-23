import asyncio
import pytest
from unittest.mock import Mock

from harbor.environments.base import BaseEnvironment
from harbor.environments.launch import LaunchSpec


class _Dummy(BaseEnvironment):
    """Defines only what's needed to exist as a class; never instantiated."""

    @staticmethod
    def type() -> str:
        return "dummy"

    def _validate_definition(self): ...
    async def start(self, force_build): ...
    async def stop(self, delete): ...
    async def upload_file(self, source_path, target_path): ...
    async def upload_dir(self, source_dir, target_dir): ...
    async def download_file(self, source_path, target_path): ...
    async def download_dir(self, source_dir, target_dir): ...
    async def exec(self, *args, **kwargs): ...


def test_base_launch_rejects_by_default():
    with pytest.raises(NotImplementedError):
        _Dummy.launch(LaunchSpec(image="x:1"))


def test_image_digest_default_is_none():
    # The base default contributes no digest; backends that can resolve one
    # override this. Called as an unbound coroutine with a Mock self so we do
    # not have to satisfy every BaseEnvironment abstractmethod.
    assert asyncio.run(BaseEnvironment.image_digest(Mock())) is None
