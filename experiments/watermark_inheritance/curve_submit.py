"""Freeze runnable source and submit the curve behind a successful data job."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

from .common import write_json
from .prepare import file_sha256


def submit(root, data_job):
    root = Path(root).resolve()
    source = root / "setup/source"
    if source.exists():
        raise FileExistsError("A frozen source snapshot already exists")
    module = Path("experiments/watermark_inheritance")
    target = source / module
    target.mkdir(parents=True)
    (source / "experiments/__init__.py").write_text("")
    for pattern in ("*.py", "*.sbatch"):
        for path in module.glob(pattern):
            shutil.copy2(path, target / path.name)
    shutil.copytree("src", source / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    files = {str(p.relative_to(source)): file_sha256(p) for p in sorted(source.rglob("*")) if p.is_file()}
    write_json(root / "setup/source_manifest.json", {"base_commit": revision, "files": files,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "note": "Immutable runnable snapshot includes authorized uncommitted curve implementation"})
    command = ["sbatch", "--parsable", f"--dependency=afterok:{data_job}", str(module / "curve_gpu.sbatch")]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    job = result.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise ValueError(f"Unexpected sbatch output: {result.stdout}")
    write_json(root / "setup/submission.json", {"data_job": data_job, "gpu_job": job,
               "command": command, "submitted_utc": datetime.now(timezone.utc).isoformat()})
    print(job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-job", required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts/watermark_curve"))
    args = parser.parse_args()
    submit(args.root, args.data_job)
