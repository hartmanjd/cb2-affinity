"""Download and verify the fitted model files that are kept outside git.

Why this exists: each training notebook saves its fitted estimators as
``models.joblib.gz``. Those files are large binaries (up to ~35 MB each), and git
keeps a full copy of every version forever, which made pushes and clones slow.
The models are therefore attached to a GitHub Release instead of being committed.

The run manifests (which *are* committed) still record each model's SHA-256
checksum under ``output_sha256``. This script uses those recorded checksums as
the source of truth: a downloaded file is only accepted when its bytes match the
checksum written by the notebook that trained it.

Usage (run from the repository root):

    python scripts/fetch_models.py            # download any missing models, then verify all
    python scripts/fetch_models.py --check    # verify only; download nothing
    python scripts/fetch_models.py --upload   # maintainer: publish new local models to the release

Only the Python standard library is used, so this works before any other
dependency is installed. ``--upload`` also needs the GitHub CLI (``gh``).
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODELS_FOLDER = Path("provenance/models")
MODEL_FILENAME = "models.joblib.gz"

# One long-lived release holds every published model version. Asset names include
# part of the checksum, so retrained models are added beside older ones rather
# than replacing them. Checking out an older commit therefore still downloads the
# exact models that commit's manifests describe.
REPOSITORY = "hartmanjd/cb2-affinity"
RELEASE_TAG = "fitted-models"
DOWNLOAD_URL = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}"


def sha256_file(path):
    """Hash a file in 1 MB chunks so large models are not read into memory at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_models(root=ROOT, folder=MODELS_FOLDER):
    """Return {repository-relative model path: recorded SHA-256} from the run manifests.

    Temporary ``.pending-*`` / ``.previous-*`` folders are skipped because they are
    unfinished or rollback copies, not accepted results.
    """
    root = Path(root)
    models = {}
    for manifest_path in sorted((root / folder).glob("**/manifest.json")):
        if any(part.startswith((".pending-", ".previous-")) for part in manifest_path.parts):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, digest in manifest.get("output_sha256", {}).items():
            if name.endswith(MODEL_FILENAME):
                models[name] = digest
    return models


def asset_name(name, digest):
    """Turn a model path into a unique, flat release file name.

    Example: provenance/models/random_forest/files/tuning/models.joblib.gz with a
    checksum starting 3f2a9c1b0d4e becomes
    random_forest__files__tuning__models.3f2a9c1b0d4e.joblib.gz
    """
    relative = Path(name).relative_to(MODELS_FOLDER)
    folders = "__".join(relative.parent.parts)
    return f"{folders}__models.{digest[:12]}.joblib.gz"


def problem_models(root=ROOT, folder=MODELS_FOLDER):
    """List (path, reason) for every expected model that is missing or has changed."""
    root = Path(root)
    problems = []
    for name, digest in expected_models(root, folder).items():
        path = root / name
        if not path.is_file():
            problems.append((name, "missing"))
        elif sha256_file(path) != digest:
            problems.append((name, "checksum does not match its manifest"))
    return problems


def require_models(root=ROOT, folder=MODELS_FOLDER):
    """Stop with a clear instruction when model files are missing or altered.

    Report builders and notebooks call this before their own checksum checks, so a
    fresh clone fails with "run fetch_models.py" instead of a bare file error.
    """
    problems = problem_models(root, folder)
    if problems:
        details = "\n".join(f"  - {name}: {reason}" for name, reason in problems)
        raise FileNotFoundError(
            "Fitted model files are not in place:\n" + details +
            "\nThey are stored outside git. Run from the repository root:\n"
            "  python scripts/fetch_models.py")


def download_models(root=ROOT, base_url=DOWNLOAD_URL):
    """Download each missing model and accept it only if its checksum matches."""
    root = Path(root)
    downloaded = 0
    for name, digest in expected_models(root).items():
        target = root / name
        if target.is_file() and sha256_file(target) == digest:
            continue  # Already present and correct; nothing to do.
        if target.is_file():
            # A local file with different bytes may be a new, not-yet-recorded run.
            # Never overwrite it silently; the person must decide what to keep.
            raise RuntimeError(f"{name} exists but does not match its manifest; "
                               "move it aside yourself before downloading.")
        url = f"{base_url}/{asset_name(name, digest)}"
        print(f"Downloading {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        # Download to a temporary file in the same folder, check it, then rename.
        # The rename is atomic, so an interrupted or corrupt download never
        # leaves a half-written file at the real model path.
        handle, temporary = tempfile.mkstemp(prefix=".download-", dir=target.parent)
        os.close(handle)
        try:
            with urllib.request.urlopen(url) as response, open(temporary, "wb") as stream:
                shutil.copyfileobj(response, stream)
            actual = sha256_file(temporary)
            if actual != digest:
                raise RuntimeError(f"Checksum mismatch for {name}: expected {digest}, got {actual}")
            os.replace(temporary, target)
            downloaded += 1
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)
    return downloaded


def run_gh(*arguments):
    """Run a GitHub CLI command and return its standard output."""
    result = subprocess.run(["gh", *arguments], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(arguments)} failed:\n{result.stderr.strip()}")
    return result.stdout


def upload_models(root=ROOT):
    """Publish local models that the release does not have yet (maintainer step).

    Run this after a notebook retrains a model and before pushing the commit that
    records the new checksums; otherwise other people cannot fetch the new model.
    """
    root = Path(root)
    require_models(root)  # Only publish files that match their manifests.
    # Create the release the first time; later uploads add to the same release.
    try:
        run_gh("release", "view", RELEASE_TAG, "--repo", REPOSITORY)
    except RuntimeError:
        run_gh("release", "create", RELEASE_TAG, "--repo", REPOSITORY,
               "--title", "Fitted models",
               "--notes", "Fitted model files kept outside git. Download them with "
                          "`python scripts/fetch_models.py`; checksums are in each run manifest.")
    listing = run_gh("release", "view", RELEASE_TAG, "--repo", REPOSITORY,
                     "--json", "assets", "--jq", ".assets[].name")
    published = set(listing.split())
    uploaded = 0
    with tempfile.TemporaryDirectory() as staging:
        for name, digest in expected_models(root).items():
            asset = asset_name(name, digest)
            if asset in published:
                continue
            # Copy under the unique asset name, because gh uploads by file name.
            staged = Path(staging) / asset
            shutil.copyfile(root / name, staged)
            print(f"Uploading {name} as {asset}")
            run_gh("release", "upload", RELEASE_TAG, str(staged), "--repo", REPOSITORY)
            uploaded += 1
    return uploaded


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="Verify local models without downloading.")
    parser.add_argument("--upload", action="store_true", help="Publish new local models to the GitHub release.")
    arguments = parser.parse_args()
    if arguments.upload:
        print(f"Uploaded {upload_models()} new model file(s) to release '{RELEASE_TAG}'.")
    elif not arguments.check:
        print(f"Downloaded {download_models()} model file(s).")
    try:
        require_models()
    except FileNotFoundError as error:
        sys.exit(str(error))
    print(f"All {len(expected_models())} model files match their manifest checksums.")


if __name__ == "__main__":
    main()
