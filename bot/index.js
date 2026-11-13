'use strict'
/*
 * SOMA environment adapter  ("the body").
 *
 * Transport / mechanical layer ONLY:
 *   - keeps one live Minecraft session (mineflayer, offline auth),
 *   - turns the world into a compact numeric frame (no pixels, no text),
 *   - proposes a short list of ADMISSIBLE candidate actions,
 *   - executes exactly the candidate the agent picked (re-validated here),
 *   - streams events + state deltas back to the agent.
 *
 * No policy, no reward shaping, no strategy. All of that lives in agent/.
 */

const net = require('net')
const mineflayer = require('mineflayer')
const { Vec3 } = require('vec3')

// ------------------------------------------------------------------ config --
const CFG = {
  host: process.env.SOMA_MC_HOST || '127.0.0.1',
  port: parseInt(process.env.SOMA_MC_PORT || '25565', 10),
  username: process.env.SOMA_MC_USER || 'SOMA',
  version: process.env.SOMA_MC_VERSION || '1.21.1',

  ipcHost: process.env.SOMA_IPC_HOST || '127.0.0.1',
  ipcPort: parseInt(process.env.SOMA_IPC_PORT || '8787', 10),

  obsMs: parseInt(process.env.SOMA_OBS_MS || '300', 10),
  maxCandidates: parseInt(process.env.SOMA_MAX_CAND || '48', 10),
  maxEnt: parseInt(process.env.SOMA_MAX_ENT || '12', 10),

  reach: 4.5,
  actionTimeoutTicks: 40
}

const DIRS8 = []
for (let i = 0; i < 8; i++) DIRS8.push({ yaw: (i * Math.PI) / 4, dx: Math.sin((i * Math.PI) / 4), dz: Math.cos((i * Math.PI) / 4) })

const FOOD = new Set([
  'apple', 'bread', 'beef', 'cooked_beef', 'porkchop', 'cooked_porkchop', 'mutton', 'cooked_mutton',
  'chicken', 'cooked_chicken', 'rabbit', 'cooked_rabbit', 'cod', 'cooked_cod', 'salmon', 'cooked_salmon',
  'carrot', 'golden_carrot', 'potato', 'baked_potato', 'melon_slice', 'cookie', 'pumpkin_pie',
  'beetroot', 'beetroot_soup', 'mushroom_stew', 'rabbit_stew', 'dried_kelp', 'sweet_berries',
  'glow_berries', 'honey_bottle', 'chorus_fruit'
])

const HOSTILE = new Set([
  'zombie', 'husk', 'drowned', 'zombie_villager', 'skeleton', 'stray', 'wither_skeleton', 'creeper',
  'spider', 'cave_spider', 'enderman', 'witch', 'slime', 'magma_cube', 'blaze', 'ghast', 'silverfish',
  'endermite', 'guardian', 'elder_guardian', 'shulker', 'phantom', 'pillager', 'vindicator', 'ravager',
  'evoker', 'illusioner', 'piglin_brute', 'hoglin', 'zoglin', 'vex', 'warden', 'bogged', 'breeze'
])

const ANIMAL = new Set(['cow', 'pig', 'sheep', 'chicken', 'rabbit', 'horse', 'donkey', 'mule', 'llama', 'goat',
  'mooshroom', 'axolotl', 'fox', 'wolf', 'cat', 'ocelot', 'parrot', 'turtle', 'polar_bear', 'panda',
  'frog', 'bee', 'strider', 'camel', 'sniffer', 'armadillo'])

const UNBREAKABLE = new Set(['bedrock', 'barrier', 'end_portal', 'end_portal_frame', 'end_gateway',
  'nether_portal', 'light', 'structure_void', 'command_block', 'chain_command_block',
  'repeating_command_block', 'spawner', 'reinforced_deepslate'])

const LIQUID = new Set(['water', 'lava', 'flowing_water', 'flowing_lava'])

const INTERACTIVE = /crafting_table|furnace|chest|barrel|door|lever|button|bed|anvil|loom|smithing|brewing|composter|beacon|shulker_box|hopper|dispenser|dropper|blast_furnace|smoker|campfire|bell|note_block|jukebox|enchanting_table|bookshelf|cauldron|cartography|grindstone|stonecutter|fletching/

// ------------------------------------------------------------------- state --
let bot = null
let agentSock = null
let agentBuf = ''
let lastCands = null
const knownBlocks = new Map()
const knownItems = new Map()
const newBlocks = []
const newItems = []

let activeAction = null
let actionResult = null
let pendingEvents = []

const stats = { frames: 0, sent: 0, actions: 0, errors: 0, reconnects: 0, frameMs: 0, actionMs: 0 }

// ------------------------------------------------------------------- utils --
const r2 = (v) => Math.round(v * 100000) / 100000
const isAir = (b) => !b || b.name === 'air' || b.name === 'cave_air' || b.name === 'void_air'
const solid = (b) => !!b && b.boundingBox === 'block'
const liquid = (b) => !!b && (LIQUID.has(b.name) || b.name === 'water' || b.name === 'lava')
const d3 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)
const blockAt = (x, y, z) => bot.blockAt(new Vec3(x, y, z))

// ----------------------------------------------------------------- connect --
function connect() {
  console.log(`[soma] connect ${CFG.username}@${CFG.host}:${CFG.port} v${CFG.version}`)
  bot = mineflayer.createBot({
    host: CFG.host,
    port: CFG.port,
    username: CFG.username,
    version: CFG.version,
    auth: 'offline',
    viewDistance: 'normal',
    checkTimeoutInterval: 60000
  })

  bot.on('error', (e) => { stats.errors++; console.log('[soma] error:', e.message) })
  bot.on('kicked', (r) => console.log('[soma] kicked:', JSON.stringify(r)))
  bot.on('end', (r) => {
    console.log('[soma] disconnected:', r)
    stats.reconnects++
    setTimeout(connect, 5000)
  })
  bot.on('death', () => {
    pendingEvents.push({ e: 'death', m: 1 })
    console.log('[soma] died')
    setTimeout(() => { try { bot.respawn() } catch (e) {} }, 1500)
  })
  bot.on('health', () => {
    if (bot._hp !== undefined && bot.health < bot._hp) pendingEvents.push({ e: 'hurt', m: r2(bot._hp - bot.health) })
    bot._hp = bot.health
  })

  bot.on('spawn', () => {
    const p = bot.entity.position
    console.log(`[soma] spawned ${p.x.toFixed(1)} ${p.y.toFixed(1)} ${p.z.toFixed(1)} hp=${bot.health} food=${bot.food}`)
    bot._hp = bot.health
    activeAction = null
    if (bot._physicsHooked) return
    bot._physicsHooked = true
    bot.on('physicsTick', () => { tickCount++; driveActiveAction() })
    setInterval(sendObservation, CFG.obsMs)
  })
}

let tickCount = 0

// --------------------------------------------------------- action executor --
function driveActiveAction() {
  const a = activeAction
  if (!a) return
  const c = a.cand
  const p = bot.entity.position

  if (c.a === 'goto') {
    const dx = c.x + 0.5 - p.x
    const dz = c.z + 0.5 - p.z
    if (Math.hypot(dx, dz) < 0.9) { finishAction(true, 'arrived'); return }

    const wantYaw = Math.atan2(-dx, dz)
    let dyaw = wantYaw - bot.entity.yaw
    while (dyaw > Math.PI) dyaw -= 2 * Math.PI
    while (dyaw < -Math.PI) dyaw += 2 * Math.PI
    bot.look(bot.entity.yaw + dyaw * 0.35, 0, true)

    bot.setControlState('forward', true)
    bot.setControlState('sprint', true)

    const fx = Math.floor(p.x - Math.sin(bot.entity.yaw) * 0.9)
    const fz = Math.floor(p.z - Math.cos(bot.entity.yaw) * 0.9)
    const bFeet = blockAt(fx, Math.floor(p.y), fz)
    const bHead = blockAt(fx, Math.floor(p.y) + 1, fz)
    const blocked = (bFeet && solid(bFeet)) || (bHead && solid(bHead)) || (bFeet && liquid(bFeet))
    const stepUp = c.y > p.y + 1.2
    if ((blocked && bot.entity.onGround) || (stepUp && bot.entity.onGround)) bot.setControlState('jump', true)
    else bot.setControlState('jump', false)
  }

  const moved = !a.lastPos || d3(p, a.lastPos) > 0.2
  if (moved) { a.lastPos = { x: p.x, y: p.y, z: p.z }; a.ticks = 0 } else a.ticks++
  if (a.ticks > CFG.actionTimeoutTicks) finishAction(false, 'no_progress')
}

function finishAction(ok, reason) {
  const a = activeAction
  if (!a) return
  if (a.cand.a === 'goto') {
    bot.setControlState('forward', false)
    bot.setControlState('jump', false)
    bot.setControlState('sprint', false)
  }
  actionResult = { ok: !!ok, reason: reason || 'done', a: a.cand.a }
  activeAction = null
}

function execAction(cand) {
  if (!cand) return
  if (activeAction) finishAction(false, 'replaced')
  activeAction = { cand, lastPos: null, ticks: 0 }
  stats.actions++

  const fail = (why) => { pendingEvents.push({ e: 'act_fail', m: 1, a: cand.a, w: why }); finishAction(false, why) }

  try {
    switch (cand.a) {
      case 'goto':
        break

      case 'mine': {
        const b = blockAt(cand.x, cand.y, cand.z)
        if (!b || isAir(b)) return fail('gone')
        if (d3(b.position, bot.entity.position) > 5.5) return fail('far')
        bot.lookAt(b.position, true)
        bot.dig(b, true)
          .then(() => {
            pendingEvents.push({ e: 'dig', m: 1, x: cand.x, y: cand.y, z: cand.z, n: cand.n })
            finishAction(true)
          })
          .catch(() => fail('dig_fail'))
        return
      }

      case 'attack': {
        const e = bot.entities[cand.eid]
        if (!e || !e.position) return fail('gone')
        if (d3(e.position, bot.entity.position) > 5.5) return fail('far')
        bot.lookAt(e.position, true)
        bot.attack(e)
        pendingEvents.push({ e: 'attack', m: 1, n: cand.n })
        finishAction(true)
        return
      }

      case 'use': {
        const b = blockAt(cand.x, cand.y, cand.z)
        if (!b) return fail('gone')
        if (d3(b.position, bot.entity.position) > 5.5) return fail('far')
        bot.lookAt(b.position, true)
        bot.activateBlock(b)
          .then(() => { pendingEvents.push({ e: 'use', m: 1, n: cand.n }); finishAction(true) })
          .catch(() => fail('use_fail'))
        return
      }

      case 'place': {
        const ref = blockAt(cand.x, cand.y - 1, cand.z)
        if (!ref) return fail('no_ref')
        bot.lookAt(ref.position, true)
        bot.placeBlock(ref, new Vec3(0, 1, 0))
          .then(() => { pendingEvents.push({ e: 'place', m: 1, x: cand.x, y: cand.y, z: cand.z, n: cand.n }); finishAction(true) })
          .catch(() => fail('place_fail'))
        return
      }

      case 'equip': {
        const it = bot.inventory.slots[cand.slot]
        if (!it) return fail('no_item')
        bot.equip(it, 'hand').then(() => finishAction(true)).catch(() => fail('equip_fail'))
        return
      }

      case 'eat': {
        const it = findFood()
        if (!it) return fail('no_food')
        bot.equip(it, 'hand')
          .then(() => bot.activateItem())
          .then(() => { pendingEvents.push({ e: 'eat', m: 1, n: it.name }); finishAction(true) })
          .catch(() => fail('eat_fail'))
        return
      }

      case 'drop': {
        const it = bot.heldItem
        if (!it) return fail('no_item')
        bot.toss(it.type, null, it.count).then(() => finishAction(true)).catch(() => fail('toss_fail'))
        return
      }

      case 'jump':
        bot.setControlState('jump', true)
        setTimeout(() => { bot.setControlState('jump', false); finishAction(true) }, 200)
        return

      case 'wait':
        bot.clearControlStates()
        setTimeout(() => finishAction(true), 600)
        return

      case 'stop':
        bot.clearControlStates()
        finishAction(true, 'stop')
        return

      default:
        fail('unknown')
    }
  } catch (e) {
    fail('ex:' + e.message)
  }
}

// ------------------------------------------------------------- observation --
function sendObservation() {
  if (!bot || !bot.entity || !agentSock || agentSock.destroyed) return
  const t0 = process.hrtime.bigint()
  const p = bot.entity.position
  const yaw = bot.entity.yaw || 0

  const feetX = Math.floor(p.x)
  const feetY = Math.floor(p.y)
  const feetZ = Math.floor(p.z)

  const self = {
    x: r2(p.x), y: r2(p.y), z: r2(p.z),
    vx: r2(bot.entity.velocity.x), vy: r2(bot.entity.velocity.y), vz: r2(bot.entity.velocity.z),
    yaw: r2(yaw), pitch: r2(bot.entity.pitch || 0),
    hp: bot.health, food: bot.food,
    air: bot.air === undefined ? 20 : bot.air,
    sat: r2(bot.foodSaturation || 0),
    light: r2(bot.entity.light || 0),
    dim: (bot.game && bot.game.dimension) || 'minecraft:overworld',
    on_ground: !!bot.entity.onGround,
    in_water: !!bot.entity.isInWater,
    in_lava: !!bot.entity.isInLava,
    sprinting: !!bot.entity.isSprinting,
    sneaking: !!bot.entity.isSneaking,
    flying: !!bot.entity.isFlying,
    fall: r2(bot.entity.fallDistance || 0),
    collided: !!bot.entity.isCollidedHorizontally,
    time: bot.time ? bot.time.timeOfDay : 0,
    raining: !!bot.isRaining,
    thunder: !!bot.thunder
  }

  // voxel window 8x8x8, step 2
  const ids = new Array(512)
  let k = 0
  for (let ix = 0; ix < 8; ix++) {
    const bx = feetX - 7 + ix * 2
    for (let iy = 0; iy < 8; iy++) {
      const by = feetY - 7 + iy * 2
      for (let iz = 0; iz < 8; iz++) {
        const bz = feetZ - 7 + iz * 2
        let b = null
        try { b = bot.blockAt(new Vec3(bx, by, bz)) } catch (e) {}
        ids[k++] = isAir(b) ? 0 : (b.type || 0)
        if (b && !isAir(b) && b.type) noteBlock(b.type, b.name)
      }
    }
  }

  // entities
  const ents = []
  for (const id in bot.entities) {
    const e = bot.entities[id]
    if (!e || e === bot.entity || !e.position) continue
    const d = d3(e.position, p)
    if (d > 32) continue
    const n = e.name || e.entityType || 'unknown'
    ents.push([e.id, r2(e.position.x - p.x), r2(e.position.y - p.y), r2(e.position.z - p.z), r2(d),
      e.health === undefined ? -1 : e.health,
      HOSTILE.has(n) ? 1 : (ANIMAL.has(n) ? 0 : 2)])
  }
  ents.sort((a, b) => a[4] - b[4])
  const entOut = ents.slice(0, CFG.maxEnt)

  // inventory
  const inv = []
  for (let i = 0; i < bot.inventory.slots.length; i++) {
    const it = bot.inventory.slots[i]
    if (!it) continue
    noteItem(it.type, it.name)
    inv.push([i, it.type, it.count, FOOD.has(it.name) ? 1 : 0])
  }

  const cand = buildCandidates(p, feetY)
  lastCands = cand

  const frame = {
    t: 'obs', seq: ++seqNo, tick: tickCount, ms: Date.now(),
    self, grid: { ox: feetX - 7, oy: feetY - 7, oz: feetZ - 7, step: 2, n: 8, ids },
    ent: entOut, inv, cand,
    bmap: (seqNo % 100 === 1) ? [...knownBlocks] : newBlocks.splice(0, newBlocks.length),
    imap: (seqNo % 100 === 1) ? [...knownItems] : newItems.splice(0, newItems.length),
    ev: pendingEvents.splice(0, pendingEvents.length),
    prev: activeAction ? { a: activeAction.cand.a, ticks: activeAction.ticks, ok: null }
      : { a: 'none', ticks: 0, ok: actionResult ? actionResult.ok : null },
    res: actionResult
  }
  actionResult = null

  try {
    agentSock.write(JSON.stringify(frame) + '\n')
    stats.sent++
  } catch (e) { stats.errors++ }
  stats.frames++
  stats.frameMs = Number(process.hrtime.bigint() - t0) / 1e6
}
let seqNo = 0

// -------------------------------------------------------------- candidates --
function buildCandidates(p, feetY) {
  const out = []
  const feetX = Math.floor(p.x)
  const feetZ = Math.floor(p.z)

  // --- mine targets ---
  const mineables = []
  const seen = new Set()
  for (let dx = -4; dx <= 4; dx++) {
    for (let dy = -3; dy <= 4; dy++) {
      for (let dz = -4; dz <= 4; dz++) {
        const x = feetX + dx, y = feetY + dy, z = feetZ + dz
        let b
        try { b = bot.blockAt(new Vec3(x, y, z)) } catch (e) { continue }
        if (!b || isAir(b) || UNBREAKABLE.has(b.name) || liquid(b)) continue
        if (dy === 0 && dx === 0 && dz === 0) continue
        const key = x + ',' + y + ',' + z
        if (seen.has(key)) continue
        seen.add(key)
        const d = Math.sqrt(dx * dx + dy * dy + dz * dz)
        if (d > CFG.reach) continue
        const drops = b.drops && b.drops.length ? b.drops.length : 0
        mineables.push({ x, y, z, n: b.name, d, s: drops * 10 - d })
      }
    }
  }
  mineables.sort((a, b) => b.s - a.s)
  for (let i = 0; i < Math.min(8, mineables.length); i++) {
    const m = mineables[i]
    out.push({ a: 'mine', x: m.x, y: m.y, z: m.z, n: m.n, d: r2(m.d) })
  }

  // --- attack targets ---
  const targets = []
  for (const id in bot.entities) {
    const e = bot.entities[id]
    if (!e || !e.position || e === bot.entity) continue
    const d = d3(e.position, p)
    if (d > 4.5) continue
    const n = e.name || '?'
    targets.push({ eid: e.id, n, d, h: HOSTILE.has(n) ? 1 : 0 })
  }
  targets.sort((a, b) => (b.h - a.h) || (a.d - b.d))
  for (let i = 0; i < Math.min(3, targets.length); i++) {
    const t = targets[i]
    out.push({ a: 'attack', eid: t.eid, n: t.n, h: t.h, d: r2(t.d) })
  }

  // --- interact ---
  const uses = mineables.filter(m => INTERACTIVE.test(m.n))
  for (let i = 0; i < Math.min(3, uses.length); i++) {
    out.push({ a: 'use', x: uses[i].x, y: uses[i].y, z: uses[i].z, n: uses[i].n, d: r2(uses[i].d) })
  }

  // --- place ---
  const held = bot.heldItem
  if (held) {
    for (const d of DIRS8) {
      const px = Math.floor(p.x + d.dx * 2)
      const pz = Math.floor(p.z + d.dz * 2)
      const tgt = blockAt(px, feetY, pz)
      if (tgt && isAir(tgt)) {
        const ref = blockAt(px, feetY - 1, pz)
        if (ref && solid(ref)) {
          out.push({ a: 'place', x: px, y: feetY, z: pz, n: held.name, d: 2 })
          break
        }
      }
    }
  }

  // --- goto ---
  for (const d of DIRS8) {
    for (const dist of [2, 4, 7]) {
      const gx = Math.floor(p.x + d.dx * dist)
      const gz = Math.floor(p.z + d.dz * dist)
      const gy = findStandY(gx, feetY, gz)
      if (gy === null) continue
      out.push({ a: 'goto', x: gx, y: gy, z: gz, d: dist })
    }
  }

  // --- discrete ---
  if (bot.food !== undefined && bot.food < 18 && findFood()) out.push({ a: 'eat', d: 0 })
  out.push({ a: 'jump', d: 0 })
  out.push({ a: 'wait', d: 0 })
  out.push({ a: 'stop', d: 0 })

  out.sort((a, b) => (a.a === b.a ? a.d - b.d : (a.a < b.a ? -1 : 1)))
  return out.slice(0, CFG.maxCandidates)
}

function findStandY(x, nearY, z) {
  for (let dy = 3; dy >= -4; dy--) {
    const y = nearY + dy
    const below = blockAt(x, y - 1, z)
    if (!below || !solid(below) || liquid(below)) continue
    const feet = blockAt(x, y, z)
    if (feet && solid(feet)) continue
    const head = blockAt(x, y + 1, z)
    if (head && solid(head)) continue
    return y
  }
  return null
}

function findFood() {
  const items = bot.inventory.items()
  for (const it of items) if (FOOD.has(it.name)) return it
  return null
}

// ------------------------------------------------------------ id -> name ----
function noteBlock(id, name) {
  if (!name) return
  if (knownBlocks.has(id)) return
  knownBlocks.set(id, name)
  if (newBlocks.length < 400) newBlocks.push([id, name])
}

function noteItem(id, name) {
  if (!name) return
  if (knownItems.has(id)) return
  knownItems.set(id, name)
  if (newItems.length < 400) newItems.push([id, name])
}

// -------------------------------------------------------------- agent IPC ---
function startIpc() {
  const srv = net.createServer((sock) => {
    agentSock = sock
    agentBuf = ''
    console.log('[soma] agent connected')
    sock.on('data', (d) => {
      agentBuf += d.toString()
      let nl
      while ((nl = agentBuf.indexOf('\n')) >= 0) {
        const line = agentBuf.slice(0, nl)
        agentBuf = agentBuf.slice(nl + 1)
        if (!line.trim()) continue
        let msg
        try { msg = JSON.parse(line) } catch (e) { continue }

        if (msg.t === 'act' && typeof msg.i === 'number') {
          if (!lastCands || !lastCands[msg.i]) { pendingEvents.push({ e: 'bad_action', m: 1, i: msg.i }); continue }
          if (activeAction) continue // already busy: drop (agent will see `prev` on next frame)
          execAction(lastCands[msg.i])
        } else if (msg.t === 'ping') {
          sock.write(JSON.stringify({ t: 'pong' }) + '\n')
        } else if (msg.t === 'stop') {
          bot.clearControlStates()
          finishAction(true, 'stopped')
        } else if (msg.t === 'chat' && typeof msg.m === 'string') {
          try { bot.chat(msg.m.slice(0, 200)) } catch (e) {}
        }
      }
    })
    sock.on('error', () => {})
    sock.on('close', () => { agentSock = null; console.log('[soma] agent disconnected') })
  })
  srv.listen(CFG.ipcHost, CFG.ipcPort, () => console.log(`[soma] ipc ${CFG.ipcHost}:${CFG.ipcPort}`))
}

// -------------------------------------------------------------------- main --
startIpc()
connect()

process.on('SIGTERM', () => { try { bot.quit('shutdown') } catch (e) {} process.exit(0) })
process.on('SIGINT', () => { try { bot.quit('shutdown') } catch (e) {} process.exit(0) })
process.on('uncaughtException', (e) => { stats.errors++; console.log('[soma] uncaught:', e.message) })
process.on('unhandledRejection', (e) => { console.log('[soma] unhandled:', e && e.message) })

setInterval(() => {
  console.log(`[soma] frames=${stats.frames} sent=${stats.sent} act=${stats.actions} err=${stats.errors} frame=${stats.frameMs.toFixed(1)}ms`)
}, 60000).unref()