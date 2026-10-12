"""Run shell commands from a jobs file with bounded concurrency; one log per job."""
import json
import subprocess
import sys
import time
from pathlib import Path


def main(jobs_file, concurrency, log_dir):
    jobs = [line.strip() for line in Path(jobs_file).read_text().splitlines() if line.strip() and not line.startswith('#')]
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    status_file = log_dir / 'queue_status.json'
    pending = list(enumerate(jobs))
    running = {}
    done = {}
    while pending or running:
        while pending and len(running) < concurrency:
            i, cmd = pending.pop(0)
            out = (log_dir / f'job_{i:03d}.log').open('w')
            running[i] = (subprocess.Popen(cmd, shell=True, stdout=out, stderr=subprocess.STDOUT), out, time.time(), cmd)
        for i in list(running):
            proc, out, started, cmd = running[i]
            code = proc.poll()
            if code is not None:
                out.close()
                done[i] = dict(cmd=cmd, exit=code, seconds=round(time.time() - started))
                del running[i]
        status_file.write_text(json.dumps(dict(at=time.time(), total=len(jobs), running=sorted(running),
                                               done=done, pending=len(pending)), indent=1))
        time.sleep(5)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
