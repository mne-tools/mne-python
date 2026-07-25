"""Probe CoreGraphics' ICC-colorspace cache for the weak-reference bug.

On macOS 26.4, CGColorSpaceCreateWithICCData appears to cache entries without
holding an owning reference: create -> release -> create returns a dangling
pointer and crashes (SIGTRAP) when used. This is the suspected root cause of
the Qt wait-cursor crash (QImage::toCGImage) seen on CI.

Safe discriminator (parent process): retain count right after a single create
should be 2 (object + cache's strong ref, macOS >= 26.5) vs 1 (weak cache,
macOS 26.4).

Crash probe (child process): create -> release -> create -> CGColorSpaceGetType.
Exit 0 = survived; killed by SIGTRAP/SIGSEGV = bug reproduced.

Stress probe (second child): several threads concurrently create/use/release
a colorspace from the same ICC data, so the cache entry's refcount repeatedly
bounces off zero while other threads look it up. If the cache lookup does not
atomically retain against the final release, a thread eventually gets a freed
entry -> SIGTRAP/SIGSEGV. This mimics Qt cursor conversion on the main thread
racing AppKit background-thread releases of the previous cursor's CGImage.
"""

# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import ctypes
import pathlib
import platform
import subprocess
import sys

CHILD = """
import ctypes, pathlib
cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
cg.CGColorSpaceCreateWithICCData.restype = ctypes.c_void_p
cg.CGColorSpaceCreateWithICCData.argtypes = [ctypes.c_void_p]
cg.CGColorSpaceGetType.restype = ctypes.c_int
cg.CGColorSpaceGetType.argtypes = [ctypes.c_void_p]
cf.CFDataCreate.restype = ctypes.c_void_p
cf.CFDataCreate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long]
cf.CFRelease.restype = None
cf.CFRelease.argtypes = [ctypes.c_void_p]
icc = bytearray(
    pathlib.Path("/System/Library/ColorSync/Profiles/sRGB Profile.icc").read_bytes()
)
icc[-4:] = b"MNE!"  # arbitrary profile -> generic cache path (like Qt's VP2030)
icc = bytes(icc)
d = cf.CFDataCreate(None, icc, len(icc))
p1 = cg.CGColorSpaceCreateWithICCData(d)
cf.CFRelease(p1)  # refcount -> 0; freed if the cache reference is weak
p2 = cg.CGColorSpaceCreateWithICCData(d)
print(f"child: p1={p1:#x} p2={p2:#x} same={p1 == p2}", flush=True)
print(f"child: type={cg.CGColorSpaceGetType(p2)}", flush=True)  # may SIGTRAP
print("child: survived", flush=True)
"""

STRESS_CHILD = """
import ctypes, pathlib, threading, time
cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
cg.CGColorSpaceCreateWithICCData.restype = ctypes.c_void_p
cg.CGColorSpaceCreateWithICCData.argtypes = [ctypes.c_void_p]
cg.CGColorSpaceGetType.restype = ctypes.c_int
cg.CGColorSpaceGetType.argtypes = [ctypes.c_void_p]
cf.CFDataCreate.restype = ctypes.c_void_p
cf.CFDataCreate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long]
cf.CFRelease.restype = None
cf.CFRelease.argtypes = [ctypes.c_void_p]
icc = bytearray(
    pathlib.Path("/System/Library/ColorSync/Profiles/sRGB Profile.icc").read_bytes()
)
icc[-4:] = b"MNE!"
icc = bytes(icc)
DURATION = 5.0
stop = time.monotonic() + DURATION
count = [0] * 4

def churn(idx):
    d = cf.CFDataCreate(None, icc, len(icc))
    while time.monotonic() < stop:
        p = cg.CGColorSpaceCreateWithICCData(d)  # cache hit + retain
        cg.CGColorSpaceGetType(p)  # may SIGTRAP if handed a freed entry
        cf.CFRelease(p)  # may drop entry rc to 0 while others look up
        count[idx] += 1

threads = [threading.Thread(target=churn, args=(i,)) for i in range(1, 4)]
for t in threads:
    t.start()
churn(0)
for t in threads:
    t.join()
print(f"stress child: survived {sum(count)} cycles across 4 threads", flush=True)
"""

EVICT_CHILD = """
import ctypes, pathlib
cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
cg.CGColorSpaceCreateWithICCData.restype = ctypes.c_void_p
cg.CGColorSpaceCreateWithICCData.argtypes = [ctypes.c_void_p]
cg.CGColorSpaceGetType.restype = ctypes.c_int
cg.CGColorSpaceGetType.argtypes = [ctypes.c_void_p]
cf.CFDataCreate.restype = ctypes.c_void_p
cf.CFDataCreate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long]
cf.CFRelease.restype = None
cf.CFRelease.argtypes = [ctypes.c_void_p]
icc = bytearray(
    pathlib.Path("/System/Library/ColorSync/Profiles/sRGB Profile.icc").read_bytes()
)
icc[-4:] = b"MNE!"
d_target = cf.CFDataCreate(None, bytes(icc), len(icc))
p1 = cg.CGColorSpaceCreateWithICCData(d_target)
cg.CGColorSpaceGetType(p1)
cf.CFRelease(p1)  # cache-only (rc=1) -> eviction candidate
# pressure the cache with many distinct profiles
for i in range(20000):
    v = bytearray(icc)
    v[-8:-4] = i.to_bytes(4, "big")
    dv = cf.CFDataCreate(None, bytes(v), len(v))
    q = cg.CGColorSpaceCreateWithICCData(dv)
    if q:
        cf.CFRelease(q)
    cf.CFRelease(dv)
p2 = cg.CGColorSpaceCreateWithICCData(d_target)
print(f"evict child: p1={p1:#x} p2={p2:#x} same={p1 == p2}", flush=True)
print(f"evict child: type={cg.CGColorSpaceGetType(p2)}", flush=True)  # crashable
print("evict child: survived", flush=True)
"""

PRESSURE_CHILD = """
import ctypes, pathlib, subprocess, threading, time
cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
cg.CGColorSpaceCreateWithICCData.restype = ctypes.c_void_p
cg.CGColorSpaceCreateWithICCData.argtypes = [ctypes.c_void_p]
cg.CGColorSpaceGetType.restype = ctypes.c_int
cg.CGColorSpaceGetType.argtypes = [ctypes.c_void_p]
cf.CFDataCreate.restype = ctypes.c_void_p
cf.CFDataCreate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long]
cf.CFRelease.restype = None
cf.CFRelease.argtypes = [ctypes.c_void_p]
icc = bytearray(
    pathlib.Path("/System/Library/ColorSync/Profiles/sRGB Profile.icc").read_bytes()
)
icc[-4:] = b"MNE!"
icc = bytes(icc)
DURATION = 12.0
stop = time.monotonic() + DURATION
fired = [0]

def pressure():
    # simulate memory-pressure notifications (CG caches flush on these)
    while time.monotonic() < stop:
        for level in ("warn", "critical"):
            r = subprocess.run(
                ["sudo", "-n", "memory_pressure", "-S", "-l", level],
                capture_output=True,
            )
            if r.returncode == 0:
                fired[0] += 1
        time.sleep(1.0)

def churn_distinct():
    i = 0
    while time.monotonic() < stop:
        v = bytearray(icc)
        v[-8:-4] = i.to_bytes(4, "big")
        dv = cf.CFDataCreate(None, bytes(v), len(v))
        q = cg.CGColorSpaceCreateWithICCData(dv)
        if q:
            cf.CFRelease(q)
        cf.CFRelease(dv)
        i += 1

threads = [
    threading.Thread(target=pressure),
    threading.Thread(target=churn_distinct),
]
for t in threads:
    t.start()
d = cf.CFDataCreate(None, icc, len(icc))
n = 0
while time.monotonic() < stop:
    p = cg.CGColorSpaceCreateWithICCData(d)  # may return flushed/stale entry
    cg.CGColorSpaceGetType(p)  # crashable
    cf.CFRelease(p)
    n += 1
for t in threads:
    t.join()
print(
    f"pressure child: survived {n} lookups, {fired[0]} pressure events fired",
    flush=True,
)
"""


def main():
    """Run the probe."""
    print(f"macOS {platform.mac_ver()[0]}", flush=True)
    cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    cf = ctypes.CDLL(
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )
    cg.CGColorSpaceCreateWithICCData.restype = ctypes.c_void_p
    cg.CGColorSpaceCreateWithICCData.argtypes = [ctypes.c_void_p]
    cf.CFDataCreate.restype = ctypes.c_void_p
    cf.CFDataCreate.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_long]
    cf.CFGetRetainCount.restype = ctypes.c_long
    cf.CFGetRetainCount.argtypes = [ctypes.c_void_p]

    system = pathlib.Path(
        "/System/Library/ColorSync/Profiles/sRGB Profile.icc"
    ).read_bytes()
    arbitrary = bytearray(system)
    arbitrary[-4:] = b"MNE!"
    for label, icc in [("system-sRGB", system), ("arbitrary", bytes(arbitrary))]:
        d = cf.CFDataCreate(None, icc, len(icc))
        p = cg.CGColorSpaceCreateWithICCData(d)
        rc = cf.CFGetRetainCount(p)
        cache = "strong/immortal (fixed)" if rc >= 2 else "WEAK (vulnerable)"
        print(
            f"parent: {label}: rc after single create = {rc} -> {cache}",
            flush=True,
        )

    for label, code in [
        ("crash probe", CHILD),
        ("stress probe", STRESS_CHILD),
        ("evict probe", EVICT_CHILD),
        ("pressure probe", PRESSURE_CHILD),
    ]:
        res = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        print(res.stdout, end="", flush=True)
        if res.returncode == 0:
            print(f"{label}: SURVIVED (no bug detected)", flush=True)
        else:
            print(
                f"{label}: DIED rc={res.returncode} "
                f"(negative = signal; -5 = SIGTRAP) -> BUG REPRODUCED",
                flush=True,
            )
            print(res.stderr[-500:], flush=True)


if __name__ == "__main__":
    main()
