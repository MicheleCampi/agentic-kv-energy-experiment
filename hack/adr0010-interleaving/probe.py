import subprocess, time, urllib.request, sys, collections

def run(offset, port):
    p = subprocess.Popen([sys.executable, "stub.py", str(port), str(offset)])
    time.sleep(0.6)
    samples = []
    t0 = time.time()
    while p.poll() is None and time.time() - t0 < 40:
        try:
            body = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=1).read().decode()
            samples.append(float(body.split()[-1]))
        except Exception:
            pass
        time.sleep(0.25)
    p.wait()
    return samples

for label, off, port in (("LOCKSTEP  ", 0.0, 811), ("STAGGERED ", 1.25, 812)):
    s = run(off, port)
    if not s: print(label, "nessun campione"); continue
    c = collections.Counter(s)
    frac1 = sum(1 for v in s if v == 1) / len(s)
    print("%s offset=%.2fs  campioni=%3d  running==1: %.1f%%  media=%.3f  distrib=%s"
          % (label, off, len(s), 100*frac1, sum(s)/len(s), dict(sorted(c.items()))))
