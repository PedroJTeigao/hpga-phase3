"""Detached helper: wait for stage 2 of run_sequence_ga_comparison.py to finish, then launch stage 3
(seeds 3 and 4) only if there is time. Written down so the "if there is time" call is a rule, not a mood:

  launch iff stage 2 ended normally ("all planned runs attempted") AND
       now + 2 * (mean stage-2 seconds per seed) + MARGIN <= DEADLINE

DEADLINE is passed in (the user's absence window). Stage 2's status file is copied aside first so stage 3
cannot overwrite it. Everything is logged to stdout (redirect to a file).
"""
import json, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
STATUS = RAW / "sequence_ga_comparison_status.json"
MARGIN_S = 40 * 60
deadline = datetime.fromisoformat(sys.argv[1]).timestamp()
seeds_s2 = int(sys.argv[2]) if len(sys.argv) > 2 else 3


def say(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


while True:
    st = json.load(open(STATUS)) if STATUS.exists() else {}
    if st.get("finished"):
        break
    time.sleep(30)
say(f"stage 2 finished: {st.get('exit_reason')}")
if st.get("exit_reason") != "all planned runs attempted":
    say("NOT launching stage 3: stage 2 did not end normally"); sys.exit(0)
done = st.get("done", [])
per_seed = sum(d["seconds"] for d in done) / seeds_s2
est = 2 * per_seed
now = time.time()
fits = now + est + MARGIN_S <= deadline
say(f"mean seconds per seed in stage 2 = {per_seed:.0f}; estimate for 2 seeds = {est:.0f}s; margin {MARGIN_S}s; "
    f"now+est+margin = {datetime.fromtimestamp(now + est + MARGIN_S)}; deadline = {datetime.fromtimestamp(deadline)}; launch = {fits}")
if not fits:
    sys.exit(0)
shutil.copy(STATUS, RAW / "sequence_ga_comparison_status_seeds012.json")
log = open(RAW / "sequence_ga_comparison_run_seeds34.log", "w")
subprocess.Popen(["/scratch/pcanaste/venv/bin/python", str(ROOT / "experiments" / "run_sequence_ga_comparison.py"), "run", "--seeds", "3", "4"],
                 stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(ROOT))
say("stage 3 launched (seeds 3 4), detached")
