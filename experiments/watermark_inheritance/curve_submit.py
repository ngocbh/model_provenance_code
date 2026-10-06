"""Freeze runnable source and submit the curve behind a successful data job."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

from .common import write_json
from .prepare import file_sha256


def submit(root, data_job, predecessor=None, walltime="26:00:00"):
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
    dependency = f"afterok:{data_job}" + (f",afterany:{predecessor}" if predecessor else "")
    command = ["sbatch", "--parsable", f"--dependency={dependency}", f"--time={walltime}",
               f"--output={root}/setup/logs/%x-%j.out", f"--error={root}/setup/logs/%x-%j.err",
               str(module / "curve_gpu.sbatch"), str(root)]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    job = result.stdout.strip().split(";")[0]
    if not job.isdigit():
        raise ValueError(f"Unexpected sbatch output: {result.stdout}")
    write_json(root / "setup/submission.json", {"data_job": data_job, "gpu_job": job, "predecessor": predecessor,
               "command": command, "submitted_utc": datetime.now(timezone.utc).isoformat()})
    print(job)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-job", required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts/watermark_curve"))
    parser.add_argument("--predecessor", help="Wait for this existing GPU allocation to finish, preserving the four-GPU cap")
    parser.add_argument("--walltime", default="26:00:00")
    args = parser.parse_args()
    submit(args.root, args.data_job, args.predecessor, args.walltime)
