"""Generates larger test meshes for the self-test: a smooth ellipsoid with normals and UVs, and a
dense terrain grid without normals (about 8 million triangles, roughly 400 MB)."""
import math, sys, os
out = sys.argv[1] if len(sys.argv) > 1 else '.'
R, S = 400, 500
with open(os.path.join(out, 'blob.obj'), 'w') as f:
    for i in range(R + 1):
        th = math.pi * i / R
        rows = []
        for j in range(S):
            ph = 2 * math.pi * j / S
            x, y, z = math.sin(th) * math.cos(ph) * 3, math.cos(th) * 0.6, math.sin(th) * math.sin(ph) * 1.5
            nx, ny, nz = x / 9, y / 0.36, z / 2.25
            l = math.sqrt(nx * nx + ny * ny + nz * nz) or 1
            rows.append('v %.6f %.6f %.6f\nvn %.6f %.6f %.6f\nvt %.6f %.6f\n' % (x, y, z, nx / l, ny / l, nz / l, j / S, i / R))
        f.write(''.join(rows))
    for i in range(R):
        f.write(''.join('f %d/%d/%d %d/%d/%d %d/%d/%d\nf %d/%d/%d %d/%d/%d %d/%d/%d\n' % ((a, a, a, a + S, a + S, a + S, b + S, b + S, b + S, a, a, a, b + S, b + S, b + S, b, b, b)) for a, b in ((i * S + j + 1, i * S + (j + 1) % S + 1) for j in range(S))))
N = 1950
with open(os.path.join(out, 'terrain.obj'), 'w') as f:
    for j in range(N):
        f.write(''.join('v %.5f %.5f %.5f\n' % (i * 0.01, math.sin(i * 0.02) * math.cos(j * 0.03) * 0.4, j * 0.01) for i in range(N)))
    for j in range(N):
        f.write(''.join('vt %.5f %.5f\n' % (i / N, j / N) for i in range(N)))
    for j in range(N - 1):
        f.write(''.join('f %d/%d %d/%d %d/%d %d/%d\n' % (a, a, a + N, a + N, a + N + 1, a + N + 1, a + 1, a + 1) for a in (j * N + i + 1 for i in range(N - 1))))
