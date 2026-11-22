// sokoban-solver-01 — a push-based Sokoban solver, written from scratch for
// this bounty. Strategy: the generated levels are a hallway with one to four
// small rooms hanging off it, and a box never has to leave its room (a box
// pushed out through a door can never be pushed back, and every goal sits
// inside a room). So the level is split into its rooms, each room is solved
// on its own with an A* search over pushes, and the player walks between
// rooms along the hallway. A room holds at most 4 boxes in a 6x6 slot, so a
// push search with a strong admissible heuristic solves each room at (or
// extremely close to) its push optimum, which is what the score pays for:
// 1000 points at P* pushes, 500 x P* / pushes otherwise.
//
// The search itself: A* over push states. A state is the sorted box squares
// plus the player's reachable region (canonicalised to its lowest square).
// The heuristic is the Hungarian assignment of boxes to goals over
// precomputed shortest-path distances — admissible, because every push moves
// exactly one box by one square along some path. Dead states are pruned:
// dead squares (reverse push-reachability from the goals, ignoring other
// boxes), corner traps, 2x2 blocks of walls/boxes, and boxes frozen flat
// against a wall. A transposition table keeps the cheapest push count per
// state. If A* runs out of its work budget, a greedy best-first pass with
// the same heuristic tries to find any valid (if not optimal) solution,
// which still scores instead of zero.
//
// Determinism: every cut-off is a count of expanded states, never the wall
// clock. The program never reads the clock, the environment, or any file
// outside /input and /bot. Two runs on the same input are byte-identical.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets, counted in expanded states. Shared across the whole run so
// the 60 s wall budget for the set is spent by work, not by the clock.
const ROOM_ASTAR_CAP = 250_000;
const ROOM_GREEDY_CAP = 250_000;
const WHOLE_ASTAR_CAP = 400_000;
const WHOLE_GREEDY_CAP = 400_000;
const GLOBAL_CAP = 1_500_000;
const budget = { used: 0 };

interface Dir {
  dx: number;
  dy: number;
  ch: string;
}
const DIRS: Dir[] = [
  { dx: 0, dy: -1, ch: 'u' },
  { dx: 0, dy: 1, ch: 'd' },
  { dx: -1, dy: 0, ch: 'l' },
  { dx: 1, dy: 0, ch: 'r' },
];

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: Set<number>;
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = new Set<number>();
  const boxes: number[] = [];
  let player = 0;
  for (let y = 0; y < h; y++) {
    const row = rows[y]!;
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch !== '#') wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.add(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// ---------------------------------------------------------------------------
// Search domain: a set of floor squares (a room plus its hallway door cell,
// or the whole level) with precomputed distances and dead squares.
// ---------------------------------------------------------------------------
interface Domain {
  w: number;
  cells: number[]; // global cell indices, ascending
  at: Map<number, number>; // global -> local
  nbrs: number[][]; // local adjacency: [neighbour local, dir index]
  dist: Int32Array; // n*n shortest-path distances, ignoring boxes (-1 = unreachable)
  dead: Uint8Array; // 1 = a box here can never reach any goal
  goals: number[]; // local goal indices
}

function buildDomain(level: Level, cells: number[]): Domain {
  const n = cells.length;
  const at = new Map<number, number>();
  cells.forEach((c, i) => at.set(c, i));
  const goals: number[] = [];
  cells.forEach((c, i) => {
    if (level.goals.has(c)) goals.push(i);
  });
  const { w, h } = level;
  const nbrs: number[][] = cells.map((c) => {
    const out: number[] = [];
    const x = c % w;
    const y = Math.floor(c / w);
    for (let di = 0; di < DIRS.length; di++) {
      const d = DIRS[di]!;
      const nx = x + d.dx;
      const ny = y + d.dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const j = at.get(ny * w + nx);
      if (j !== undefined) out.push(j, di);
    }
    return out;
  });
  // All-pairs shortest paths over the domain, boxes ignored.
  const dist = new Int32Array(n * n).fill(-1);
  const q: number[] = [];
  for (let s = 0; s < n; s++) {
    dist[s * n + s] = 0;
    q.length = 0;
    q.push(s);
    let head = 0;
    while (head < q.length) {
      const u = q[head++]!;
      const du = dist[s * n + u]!;
      const nb = nbrs[u]!;
      for (let k = 0; k < nb.length; k += 2) {
        const v = nb[k]!;
        if (dist[s * n + v]! < 0) {
          dist[s * n + v] = du + 1;
          q.push(v);
        }
      }
    }
  }
  // Dead squares: reverse push-reachability from the goals, other boxes
  // ignored, so every square left unreached is truly unreachable for a box.
  // A box at p can be pushed into g along d when the player can stand at
  // p - d, so p is live whenever g is live and both p and p - d are floor.
  const live = new Uint8Array(n);
  const stack: number[] = [...goals];
  for (const g of goals) live[g] = 1;
  while (stack.length > 0) {
    const g = stack.pop()!;
    const gx = cells[g]! % w;
    const gy = Math.floor(cells[g]! / w);
    for (const d of DIRS) {
      const px = gx - d.dx;
      const py = gy - d.dy;
      const bx = px - d.dx;
      const by = py - d.dy;
      if (px < 0 || py < 0 || px >= w || py >= h) continue;
      if (bx < 0 || by < 0 || bx >= w || by >= h) continue;
      const p = at.get(py * w + px);
      const b = at.get(by * w + bx);
      if (p === undefined || b === undefined) continue;
      if (!live[p]) {
        live[p] = 1;
        stack.push(p);
      }
    }
  }
  const dead = new Uint8Array(n);
  for (let i = 0; i < n; i++) if (!live[i]) dead[i] = 1;
  return { w, cells, at, nbrs, dist, dead, goals };
}

// Admissible heuristic: cheapest assignment of boxes to distinct goals over
// the precomputed distances. Each push moves one box one square, so the
// optimal assignment total can never exceed the pushes still needed. For
// more than 6 boxes the factorial assignment is replaced by the sum of each
// box's nearest-goal distance, which is still a lower bound.
function heuristic(domain: Domain, boxes: number[]): number {
  const n = boxes.length;
  if (n === 0) return 0;
  const m = domain.goals.length;
  const N = domain.cells.length;
  if (n > 6) {
    let sum = 0;
    for (const b of boxes) {
      let best = Infinity;
      for (const g of domain.goals) {
        const d = domain.dist[b * N + g]!;
        if (d >= 0 && d < best) best = d;
      }
      if (best === Infinity) return Infinity;
      sum += best;
    }
    return sum;
  }
  if (m < n) return Infinity;
  let best = Infinity;
  const used = new Array<boolean>(m).fill(false);
  const rec = (i: number, sum: number): void => {
    if (sum >= best) return;
    if (i === n) {
      best = sum;
      return;
    }
    for (let j = 0; j < m; j++) {
      if (used[j]) continue;
      const d = domain.dist[boxes[i]! * N + domain.goals[j]!]!;
      if (d < 0) continue;
      used[j] = true;
      rec(i + 1, sum + d);
      used[j] = false;
    }
  };
  rec(0, 0);
  return best;
}

// Deadlock test for a box arrangement. `boxSet` holds local box indices.
function isDead(domain: Domain, boxes: number[], boxSet: Set<number>): boolean {
  const { w } = domain;
  const goalSet = new Set(domain.goals);
  // A square is wall-blocked when it is outside the domain entirely (a real
  // wall or another part of the level); a square is blocked when it is
  // wall-blocked or currently holds a box. Corner and frozen tests must use
  // wall-blocked sides only: a neighbouring box may move away later, so it
  // can never trap this box for good. The 2x2 test may use either, since a
  // 2x2 block of walls and boxes can never be entered from outside.
  const wallBlocked = (gx: number, gy: number): boolean => {
    if (gx < 0 || gy < 0 || gx >= w) return true;
    return domain.at.get(gy * w + gx) === undefined;
  };
  const blocked = (gx: number, gy: number): boolean => {
    if (gx < 0 || gy < 0 || gx >= w) return true;
    const g = gy * w + gx;
    const local = domain.at.get(g);
    if (local === undefined) return true; // wall or outside the domain
    return boxSet.has(local);
  };
  for (const b of boxes) {
    if (goalSet.has(b)) continue;
    if (domain.dead[b]) return true;
    const c = domain.cells[b]!;
    const x = c % w;
    const y = Math.floor(c / w);
    // Corner: walls on two perpendicular sides. The box can never be pushed
    // along either wall, and the player can never stand behind it.
    const nW = wallBlocked(x, y - 1);
    const sW = wallBlocked(x, y + 1);
    const wW = wallBlocked(x - 1, y);
    const eW = wallBlocked(x + 1, y);
    if ((nW && wW) || (nW && eW) || (sW && wW) || (sW && eW)) return true;
    // 2x2 block of walls/boxes containing this box.
    for (const [ox, oy] of [
      [0, 0],
      [-1, 0],
      [0, -1],
      [-1, -1],
    ] as const) {
      if (
        blocked(x + ox, y + oy) &&
        blocked(x + ox + 1, y + oy) &&
        blocked(x + ox, y + oy + 1) &&
        blocked(x + ox + 1, y + oy + 1)
      ) {
        return true;
      }
    }
    // Frozen flat against a wall: a wall on one side and walls on both
    // sides along it, so the box cannot slide and the player cannot get
    // behind it on any side.
    if (nW && wW && eW) return true;
    if (sW && wW && eW) return true;
    if (wW && nW && sW) return true;
    if (eW && nW && sW) return true;
  }
  return false;
}

// Player reachability inside a domain with the given boxes as obstacles.
// Returns the reachable mask plus BFS parents for path reconstruction.
function reachInfo(
  domain: Domain,
  boxSet: Set<number>,
  from: number,
): { mask: Uint8Array; prev: Int32Array; prevDir: Int8Array } {
  const n = domain.cells.length;
  const mask = new Uint8Array(n);
  const prev = new Int32Array(n).fill(-1);
  const prevDir = new Int8Array(n).fill(-1);
  const q: number[] = [from];
  mask[from] = 1;
  let head = 0;
  while (head < q.length) {
    const u = q[head++]!;
    const nb = domain.nbrs[u]!;
    for (let k = 0; k < nb.length; k += 2) {
      const v = nb[k]!;
      if (mask[v] || boxSet.has(v)) continue;
      mask[v] = 1;
      prev[v] = u;
      prevDir[v] = nb[k + 1]!;
      q.push(v);
    }
  }
  return { mask, prev, prevDir };
}

// Walk string from `from` to `to` using a reachability BFS result.
function walkString(
  reach: { prev: Int32Array; prevDir: Int8Array },
  from: number,
  to: number,
): string {
  if (from === to) return '';
  let out = '';
  let cur = to;
  while (cur !== from && cur >= 0) {
    out = DIRS[reach.prevDir[cur]!]!.ch + out;
    cur = reach.prev[cur]!;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Push-space A* (or greedy best-first when `greedy`) over one domain.
// Returns the player moves from the start position, or null when the work
// budget runs out without a solution.
// ---------------------------------------------------------------------------
function pushSearch(
  domain: Domain,
  startBoxesGlobal: number[],
  startPlayerGlobal: number,
  cap: number,
  greedy: boolean,
): { moves: string; pushes: number } | null {
  const n = domain.cells.length;
  const goalSet = new Set(domain.goals);
  const startBoxes = startBoxesGlobal
    .map((g) => domain.at.get(g)!)
    .sort((a, b) => a - b);
  const startPlayer = domain.at.get(startPlayerGlobal)!;
  if (startBoxes.every((b) => goalSet.has(b))) return { moves: '', pushes: 0 };
  if (isDead(domain, startBoxes, new Set(startBoxes))) return null;

  const keyOf = (boxes: number[], normPlayer: number): string =>
    String.fromCharCode(...boxes, 256 + normPlayer);
  const normPlayer = (boxes: number[], player: number): number => {
    const info = reachInfo(domain, new Set(boxes), player);
    let m = player;
    for (let i = 0; i < n; i++) if (info.mask[i] && i < m) m = i;
    return m;
  };

  // State store: boxes, actual player square, pushes so far, parent, moves.
  const sBoxes: number[][] = [startBoxes];
  const sPlayer: number[] = [startPlayer];
  const sG: number[] = [0];
  const sParent: number[] = [-1];
  const sSeg: string[] = [''];
  const best = new Map<string, number>();
  best.set(keyOf(startBoxes, normPlayer(startBoxes, startPlayer)), 0);

  // Binary heap of [priority, h, sequence, state, g]. Lower priority first;
  // ties break toward lower h, then earlier insertion: fully deterministic.
  const heap: [number, number, number, number, number][] = [];
  const less = (
    a: [number, number, number, number, number],
    b: [number, number, number, number, number],
  ): boolean =>
    a[0] < b[0] ||
    (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  let seq = 0;
  const h0 = heuristic(domain, startBoxes);
  if (h0 === Infinity) return null;
  heapPush([greedy ? h0 : h0, h0, seq++, 0, 0]);
  let expanded = 0;
  while (heap.length > 0 && expanded < cap && budget.used < GLOBAL_CAP) {
    const [, , , s, g] = heapPop();
    if (best.get(keyOf(sBoxes[s]!, normPlayer(sBoxes[s]!, sPlayer[s]!))) !== s) continue;
    if (sG[s] !== g) continue;
    const boxes = sBoxes[s]!;
    if (boxes.every((b) => goalSet.has(b))) {
      let moves = '';
      let pushes = 0;
      for (let t = s; t > 0; t = sParent[t]!) {
        moves = sSeg[t]! + moves;
        pushes++;
      }
      return { moves, pushes };
    }
    expanded++;
    budget.used++;
    const player = sPlayer[s]!;
    const boxSet = new Set(boxes);
    const reach = reachInfo(domain, boxSet, player);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi]!;
      const c = domain.cells[b]!;
      const x = c % domain.w;
      const y = Math.floor(c / domain.w);
      for (let di = 0; di < DIRS.length; di++) {
        const d = DIRS[di]!;
        const tx = x + d.dx;
        const ty = y + d.dy;
        const sx = x - d.dx;
        const sy = y - d.dy;
        const target = domain.at.get(ty * domain.w + tx);
        const stance = domain.at.get(sy * domain.w + sx);
        if (target === undefined || stance === undefined) continue;
        if (boxSet.has(target)) continue;
        if (!reach.mask[stance]) continue;
        const nb = boxes.slice();
        nb[bi] = target;
        nb.sort((a, b) => a - b);
        const nbSet = new Set(nb);
        if (isDead(domain, nb, nbSet)) continue;
        const h = heuristic(domain, nb);
        if (h === Infinity) continue;
        const g2 = g + 1;
        const key = keyOf(nb, normPlayer(nb, b));
        const seen = best.get(key);
        if (seen !== undefined && sG[seen]! <= g2) continue;
        const t = sBoxes.length;
        sBoxes.push(nb);
        sPlayer.push(b);
        sG.push(g2);
        sParent.push(s);
        sSeg.push(walkString(reach, player, stance) + d.ch);
        best.set(key, t);
        heapPush([greedy ? h : g2 + h, h, seq++, t, g2]);
      }
    }
  }
  return null;
}

// ---------------------------------------------------------------------------
// Whole-level helpers: walking the player without pushing, and replaying a
// move string against the global level to track the true state.
// ---------------------------------------------------------------------------
function walkPath(
  level: Level,
  from: number,
  boxes: Set<number>,
  to: number,
): string | null {
  if (from === to) return '';
  const { w, h } = level;
  const prev = new Int32Array(w * h).fill(-1);
  const prevCh = new Array<string>(w * h).fill('');
  const q: number[] = [from];
  prev[from] = from;
  let head = 0;
  while (head < q.length) {
    const u = q[head++]!;
    if (u === to) break;
    const x = u % w;
    const y = Math.floor(u / w);
    for (const d of DIRS) {
      const nx = x + d.dx;
      const ny = y + d.dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const v = ny * w + nx;
      if (level.wall[v] || boxes.has(v) || prev[v] >= 0) continue;
      prev[v] = u;
      prevCh[v] = d.ch;
      q.push(v);
    }
  }
  if (prev[to] < 0) return null;
  let out = '';
  let cur = to;
  while (cur !== from) {
    out = prevCh[cur]! + out;
    cur = prev[cur]!;
  }
  return out;
}

function simulate(
  level: Level,
  player: number,
  boxes: Set<number>,
  moves: string,
): { player: number; boxes: Set<number> } | null {
  let p = player;
  const bs = new Set(boxes);
  const { w } = level;
  for (const ch of moves) {
    const d = DIRS.find((dd) => dd.ch === ch);
    if (!d) return null;
    const x = p % w;
    const y = Math.floor(p / w);
    const q = (y + d.dy) * w + (x + d.dx);
    if (level.wall[q]) return null;
    if (bs.has(q)) {
      const r = (y + 2 * d.dy) * w + (x + 2 * d.dx);
      if (level.wall[r] || bs.has(r)) return null;
      bs.delete(q);
      bs.add(r);
    }
    p = q;
  }
  return { player: p, boxes: bs };
}

// ---------------------------------------------------------------------------
// Room decomposition and per-room solving.
// ---------------------------------------------------------------------------
interface Room {
  cells: number[];
  cellSet: Set<number>;
  door: number; // room square the player steps onto from the hallway
  hall: number; // hallway square beside the door
}

function findRooms(level: Level): Room[] | null {
  const { w, h } = level;
  const hallwayRow = Math.floor(level.player / w);
  const seen = new Uint8Array(w * h);
  const rooms: Room[] = [];
  for (let i = 0; i < w * h; i++) {
    if (level.wall[i] || Math.floor(i / w) === hallwayRow || seen[i]) continue;
    const cells: number[] = [];
    const stack = [i];
    seen[i] = 1;
    while (stack.length > 0) {
      const c = stack.pop()!;
      cells.push(c);
      const x = c % w;
      const y = Math.floor(c / w);
      for (const d of DIRS) {
        const nx = x + d.dx;
        const ny = y + d.dy;
        if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
        const j = ny * w + nx;
        if (level.wall[j] || Math.floor(j / w) === hallwayRow || seen[j]) continue;
        seen[j] = 1;
        stack.push(j);
      }
    }
    cells.sort((a, b) => a - b);
    let door = -1;
    let hall = -1;
    for (const c of cells) {
      const x = c % w;
      const y = Math.floor(c / w);
      for (const d of DIRS) {
        const nx = x + d.dx;
        const ny = y + d.dy;
        if (ny !== hallwayRow || nx < 0 || nx >= w) continue;
        const j = ny * w + nx;
        if (!level.wall[j]) {
          door = c;
          hall = j;
          break;
        }
      }
      if (door >= 0) break;
    }
    rooms.push({ cells, cellSet: new Set(cells), door, hall });
  }
  if (rooms.length === 0) return null;
  const roomOf = new Map<number, Room>();
  for (const r of rooms) for (const c of r.cells) roomOf.set(c, r);
  for (const b of level.boxes) if (!roomOf.has(b)) return null;
  for (const g of level.goals) if (!roomOf.has(g)) return null;
  for (const r of rooms) {
    if (r.door < 0) return null;
    const nb = r.cells.filter((c) => level.boxes.includes(c)).length;
    const ng = r.cells.filter((c) => level.goals.has(c)).length;
    if (nb === 0 || nb !== ng) return null;
  }
  return rooms;
}

function solveByRooms(level: Level, rooms: Room[]): string | null {
  let player = level.player;
  let boxes = new Set(level.boxes);
  let moves = '';
  const ordered = [...rooms].sort((a, b) => a.door - b.door);
  for (const room of ordered) {
    const walk1 = walkPath(level, player, boxes, room.hall);
    if (walk1 === null) return null;
    moves += walk1;
    player = room.hall;
    const dx = (room.door % level.w) - (room.hall % level.w);
    const dy = Math.floor(room.door / level.w) - Math.floor(room.hall / level.w);
    const step = DIRS.find((d) => d.dx === dx && d.dy === dy);
    if (!step) return null;
    moves += step.ch;
    player = room.door;
    // The room domain also holds the hallway square beside the door, so a
    // box sitting on the door can still be pushed back inside (the player
    // then stands in the hallway to make that push).
    const domain = buildDomain(level, [...room.cells, room.hall]);
    const roomBoxes = [...boxes].filter((c) => room.cellSet.has(c));
    const res =
      pushSearch(domain, roomBoxes, room.door, ROOM_ASTAR_CAP, false) ??
      pushSearch(domain, roomBoxes, room.door, ROOM_GREEDY_CAP, true);
    if (!res) return null;
    moves += res.moves;
    const sim = simulate(level, player, boxes, res.moves);
    if (!sim) return null;
    player = sim.player;
    boxes = sim.boxes;
    const walk2 = walkPath(level, player, boxes, room.hall);
    if (walk2 === null) return null;
    moves += walk2;
    player = room.hall;
  }
  return moves;
}

function solveLevel(level: Level): string {
  const rooms = findRooms(level);
  if (rooms) {
    const byRooms = solveByRooms(level, rooms);
    if (byRooms !== null) return byRooms;
  }
  // Fallback: one push search over the whole level.
  const allCells: number[] = [];
  for (let i = 0; i < level.w * level.h; i++) if (!level.wall[i]) allCells.push(i);
  const domain = buildDomain(level, allCells);
  const res =
    pushSearch(domain, level.boxes, level.player, WHOLE_ASTAR_CAP, false) ??
    pushSearch(domain, level.boxes, level.player, WHOLE_GREEDY_CAP, true);
  return res ? res.moves : '';
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const moves = solveLevel(parseLevel(readFileSync(join(INPUT_DIR, id, 'level.txt'), 'utf8')));
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
