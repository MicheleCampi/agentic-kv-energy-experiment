import subprocess, time, urllib.request, sys

def run(n, offset, arrival, het, port):
    p = subprocess.Popen([sys.executable, "stub.py", str(port), str(offset), str(n), arrival, "het" if het else "hom"])
    time.sleep(0.6)
    s, t0 = [], time.time()
    while p.poll() is None and time.time() - t0 < 70:
        try:
            s.append(float(urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=1).read().decode().split()[-1]))
        except Exception:
            pass
        time.sleep(0.25)
    p.wait()
    return 100*sum(1 for x in s if x == 0)/len(s) if s else float('nan')

port = 8400
for het in (False, True):
    row = "heterogeneous" if het else "homogeneous  "
    res = {}
    for label, off, arr in (("SYNC",0.0,"fixed"), ("POISSON",1.25,"poisson"), ("SPACED",1.25,"fixed")):
        port += 1
        res[label] = run(8, off, arr, het, port)
    print("  %s  SYNC %5.1f%%  POISSON %5.1f%%  SPACED %5.1f%%   free %+5.1f  earned %+5.1f"
          % (row, res["SYNC"], res["POISSON"], res["SPACED"],
             res["SYNC"]-res["POISSON"], res["POISSON"]-res["SPACED"]))
