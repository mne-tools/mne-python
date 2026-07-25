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
"""

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

    res = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True)
    print(res.stdout, end="", flush=True)
    if res.returncode == 0:
        print("crash probe: SURVIVED (no bug on this OS)", flush=True)
    else:
        print(
            f"crash probe: DIED rc={res.returncode} "
            f"(negative = signal; -5 = SIGTRAP) -> BUG REPRODUCED",
            flush=True,
        )
        print(res.stderr[-500:], flush=True)


if __name__ == "__main__":
    main()
