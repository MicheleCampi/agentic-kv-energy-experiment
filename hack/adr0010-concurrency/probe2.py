import subprocess, time, urllib.request, sys, collections

def run(n, offset, port):
    p = subprocess.Popen([sys.executable, "stub.py", str(port), str(offset), str(n)])
    time.sleep(0.6)
    s, t0 = [], time.time()
    while p.poll() is None and time.time() - t0 < 60:
        try:
            s.append(float(urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=1).read().decode().split()[-1]))
        except Exception:
            pass
        time.sleep(0.25)
    p.wait()
    return s

port = 8300
print("  N   arm         idle(run==0)   mean    gap")
for n in (2, 4, 8):
    res = {}
    for arm, off in (("SYNC", 0.0), ("STAG", 1.25)):
        port += 1
        v = run(n, off, port)
        idle = 100*sum(1 for x in v if x == 0)/len(v) if v else float('nan')
        res[arm] = idle
        print("  %-3d %-10s  %6.1f%%      %.2f" % (n, arm, idle, sum(v)/len(v) if v else 0))
    print("      %-10s                        %+.1f pp" % ("gap", res["SYNC"]-res["STAG"]))
