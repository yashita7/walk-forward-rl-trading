"""Regression checks that credential files stay local while the template is public."""
from pathlib import Path
import subprocess


def test_credential_files_are_ignored_and_example_is_publishable():
    for name in [".env", ".env.local", ".env.production", "nested/.env", "production.env",
                 ".envrc", "secrets/provider.json", "credentials.json", "private.pem",
                 "identity.pfx", "id_ed25519"]:
        result = subprocess.run(["git", "check-ignore", "--no-index", "-q", name], check=False)
        assert result.returncode == 0, f"Credential path is not ignored: {name}"
    result = subprocess.run(["git", "check-ignore", "--no-index", "-q", ".env.example"], check=False)
    assert result.returncode == 1


def test_no_local_environment_or_private_keys_are_tracked():
    names = subprocess.check_output(["git", "ls-files", "-z"], text=True).split("\0")
    for name in filter(None, names):
        path = Path(name)
        assert path.name == ".env.example" or not path.name.startswith(".env")
        assert path.suffix not in {".pem", ".key", ".p12", ".pfx", ".jks", ".keystore"}
        assert path.name not in {"id_rsa", "id_ed25519", "credentials", "credentials.json"}
