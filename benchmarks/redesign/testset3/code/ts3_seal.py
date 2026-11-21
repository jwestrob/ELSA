#!/usr/bin/env python3
"""Seal the testset3 package at the end: SHA-256 of every file listed in MANIFEST.json (written by make_manifest.py just
before), plus one combined SHA-256 over the sorted '<path>\t<sha256>\n' lines. usage: ts3_seal.py PKG_DIR"""
import hashlib, json, sys, time
P = sys.argv[1]
m = json.load(open(f"{P}/MANIFEST.json"))
files = {r["path"]: r["sha256"] for r in m["files"] if r["path"] not in ("seal.json",)}
combined = hashlib.sha256("".join(f"{k}\t{v}\n" for k, v in sorted(files.items())).encode()).hexdigest()
json.dump(dict(package=P, sealed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), n_files=len(files),
               total_bytes=m["total_bytes"], files=files, seal_sha256=combined,
               rule="sha256 over the sorted lines '<path>\\t<sha256>\\n' of every package file in MANIFEST.json"),
          open(f"{P}/seal.json", "w"), indent=1)
print(json.dumps(dict(seal_sha256=combined, n_files=len(files))), flush=True)
