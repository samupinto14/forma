"""A UDIM-style test: 30 tiles of 4096 px, each on its own material, like a scanned environment."""
import os, sys
from PIL import Image
out = sys.argv[1] if len(sys.argv) > 1 else '.'
tiles = [(u, v) for v in range(1, 6) for u in range(1, 7)]
S = 4096
for (u, v) in tiles:
    g = Image.linear_gradient('L').resize((S, S))
    img = Image.merge('RGB', (g, g.rotate(90), Image.new('L', (S, S), 40 * u)))
    img.save(os.path.join(out, f'scan_u{u}_v{v}.png'), compress_level=1)
with open(os.path.join(out, 'udim.mtl'), 'w') as f:
    for (u, v) in tiles:
        f.write(f'newmtl scan_u{u}_v{v}\nKd 1 1 1\nmap_Kd scan_u{u}_v{v}.png\n')
lines = ['mtllib udim.mtl']
vi = 1
for (u, v) in tiles:
    x, z = (u - 1) * 1.0, (v - 1) * 1.0
    lines += [f'v {x} 0 {z}', f'v {x+1} 0 {z}', f'v {x+1} 0 {z+1}', f'v {x} 0 {z+1}',
              'vt 0 0', 'vt 1 0', 'vt 1 1', 'vt 0 1', f'usemtl scan_u{u}_v{v}',
              f'f {vi}/{vi} {vi+3}/{vi+3} {vi+2}/{vi+2} {vi+1}/{vi+1}']
    vi += 4
open(os.path.join(out, 'udim.obj'), 'w').write('\n'.join(lines) + '\n')
