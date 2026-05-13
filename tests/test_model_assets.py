import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_event_poc.model_assets import checkpoint_url, resolve_checkpoint_path


class ModelAssetsTests(unittest.TestCase):
    def test_checkpoint_url_uses_hf_endpoint_when_set(self):
        with patch.dict(os.environ, {"HF_ENDPOINT": "https://hf-mirror.com"}, clear=False):
            url = checkpoint_url("630k-audioset-best.pt")

        self.assertEqual(
            url,
            "https://hf-mirror.com/lukewys/laion_clap/resolve/main/630k-audioset-best.pt",
        )

    def test_resolve_checkpoint_path_returns_existing_file_without_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "630k-audioset-best.pt"
            checkpoint.write_bytes(b"checkpoint")

            resolved = resolve_checkpoint_path(checkpoint_dir=Path(tmpdir), download=False)

        self.assertEqual(resolved, checkpoint)


if __name__ == "__main__":
    unittest.main()
