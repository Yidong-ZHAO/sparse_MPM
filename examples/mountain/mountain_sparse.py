import warp as wp
import numpy as np
from math import pi

import meshio

wp.init() 

import sys
sys.path.append('../..')
from utilities.timer import reentrant_timer

# ============ Timer ============
timer = reentrant_timer()


# ============ MPM setup ============
dt = 0.0035 # s

# ============ Particles ============
n_particles = 425365

@wp.struct
class Particles:
	x: wp.array(dtype=wp.vec3d)
	v: wp.array(dtype=wp.vec3d)
	volume_initial: wp.float64
	volume: wp.array(dtype=wp.float64)
	Kirchhoff_stress: wp.array(dtype=wp.mat33d)
	Affine_C: wp.array(dtype=wp.mat33d)
	deformation_gradient: wp.array(dtype=wp.mat33d)

particles = Particles()
particles.x = wp.zeros(shape=n_particles, dtype=wp.vec3d)
points_read = meshio.read("./data/mountain_particles_f0.ply") 
particles.x = wp.from_numpy(points_read.points, dtype=wp.vec3d)
particles.v = wp.zeros(shape=n_particles, dtype=wp.vec3d)
particles.volume_initial = 0.0100422193241 # m^3
particles.Kirchhoff_stress = wp.zeros(shape=n_particles, dtype=wp.mat33d)
particles.Affine_C = wp.zeros(shape=n_particles, dtype=wp.mat33d)
particles.deformation_gradient = wp.zeros(shape=n_particles, dtype=wp.mat33d)
particles.deformation_gradient.fill_([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

# ============ Hash ============
B = 4 
nodes_per_block = B * B * B
BITS = 21
MASK = (1 << BITS) - 1
BIAS = 1 << (BITS - 1)
EMPTY_KEY = wp.int64(-1)

max_blocks = 2 # this will be modified later
max_nodes = max_blocks * nodes_per_block
hash_capacity = max_blocks * 2
hash_mask = hash_capacity - 1
hash_keys = wp.zeros(shape=hash_capacity, dtype=wp.int64)
hash_vals = wp.zeros(shape=hash_capacity, dtype=wp.int32)
compacted_block_keys = wp.zeros(shape=max_blocks, dtype=wp.int64)
counter = wp.zeros(shape=1, dtype=wp.int32)
overflow = wp.zeros(shape=1, dtype=wp.int32)

n_blocks = counter.numpy()[0]
n_nodes_used = n_blocks * nodes_per_block

# ============ Grid ============
h = 0.431492478463 # m
overh = 1./h

@wp.struct
class Grid:
	mass: wp.array(dtype=wp.float64)
	old_velocity: wp.array(dtype=wp.vec3d)
	new_velocity: wp.array(dtype=wp.vec3d)
	force: wp.array(dtype=wp.vec3d)
	terrain_phi: wp.array(dtype=wp.float64)

grid = Grid()
grid.mass = wp.zeros(shape=max_nodes, dtype=wp.float64)
grid.old_velocity = wp.zeros(shape=max_nodes, dtype=wp.vec3d)
grid.new_velocity = wp.zeros(shape=max_nodes, dtype=wp.vec3d)
grid.force = wp.zeros(shape=max_nodes, dtype=wp.vec3d)
grid.terrain_phi = wp.zeros(shape=max_nodes, dtype=wp.float64)

# ============ Material ============
@wp.struct
class Material:
	# Elasticity
	lame_lambda: wp.float64
	lame_mu: wp.float64
	K: wp.float64
	# Plasticity
	friction_angle: wp.float64
	# Particles
	density: wp.float64
	p_mass: wp.float64
	basal_friction_mu: wp.float64
	FLIP_ratio: wp.float64

material = Material()
youngs_modulus = 5e2 # kPa
poisson_ratio = 0.3
lame_lambda = youngs_modulus*poisson_ratio / ((1.0+poisson_ratio) * (1.0-2.0*poisson_ratio))
lame_mu = youngs_modulus / (2.0*(1.0+poisson_ratio))
material.lame_lambda = lame_lambda
material.lame_mu = lame_mu
material.K = material.lame_lambda + 2./3. * material.lame_mu

material.friction_angle = 30. # degrees

material.density = 2.0 # t/m^3
material.p_mass = material.density * particles.volume_initial
material.basal_friction_mu = 0.35
material.FLIP_ratio = 0.95


# ============ terrain ============
file = open("./data/mountain_terrain.nvdb", "rb")

terrain = wp.Volume.load_from_nvdb(file)



# ============ Warp kernels ============
# Loop on particles, mark active blocks, and try to insert the key into the hash table
@wp.kernel
def mark_active_blocks(particles: Particles,
					   overh: wp.float64,
					   B: wp.int32,
					   BITS: wp.int32,
					   MASK: wp.int64,
					   BIAS: wp.int32,
					   EMPTY_KEY: wp.int64,
					   hash_mask: wp.int32,
					   hash_keys: wp.array(dtype=wp.int64),
					   hash_vals: wp.array(dtype=wp.int32),
					   counter: wp.array(dtype=wp.int32),
					   compacted_block_keys: wp.array(dtype=wp.int64),
					   max_blocks: wp.int32,
					   overflow: wp.array(dtype=wp.int32)
					   ):
	p = wp.tid()
	xp = particles.x[p]

	# Quadratc B-splines base index
	base_x = xp[0] * overh - wp.float64(0.5)
	base_y = xp[1] * overh - wp.float64(0.5)
	base_z = xp[2] * overh - wp.float64(0.5)

	base_int = wp.vector(wp.int32(wp.floor(base_x)), wp.int32(wp.floor(base_y)), wp.int32(wp.floor(base_z)))

	# Loop on grid nodes
	for flattened_loop in range(0, 27):
		i = wp.int(flattened_loop / 9)
		j = wp.mod(wp.int(flattened_loop / 3), 3)
		k = wp.mod(flattened_loop, 3)

		ix = base_int[0] + i
		iy = base_int[1] + j
		iz = base_int[2] + k

		# node coordinate -> block coordinate
		bx = ix // B
		by = iy // B
		bz = iz // B

		key = pack(bx, by, bz, BITS, MASK, BIAS)

		# Insert into hash, assign compact block id
		try_insert_block_key(key, hash_mask, hash_keys, hash_vals, EMPTY_KEY, counter, compacted_block_keys, max_blocks, overflow)

# (i, j, k) -> key (int64)
@wp.func
def pack(bx: wp.int32,
		 by: wp.int32,
		 bz: wp.int32,
		 BITS: wp.int32,
		 MASK: wp.int64,
		 BIAS: wp.int32
		 ) -> wp.int64:
	ux = wp.int64((bx + BIAS)) & MASK
	uy = wp.int64((by + BIAS)) & MASK
	uz = wp.int64((bz + BIAS)) & MASK

	# Pack into 64-bit key
	return (ux << wp.int64(2 * BITS)) | (uy << wp.int64(BITS)) | uz

# Hash function
# See Mix13 in https://zimbry.blogspot.com/2011/09/better-bit-mixing-improving-on.html
@wp.func
def hash64(key: wp.int64) -> wp.int32:
	# 64-bit mix -> int32
	z = wp.uint64(key)
	z = (z ^ (z >> wp.uint64(30))) * wp.uint64(0xbf58476d1ce4e5b9)
	z = (z ^ (z >> wp.uint64(27))) * wp.uint64(0x94d049bb133111eb)
	z = z ^ (z >> wp.uint64(31))
	return wp.int32(z)

# Try to insert the key into the hash table.
# If the insertion fails, overflow will be set to 1
@wp.func
def try_insert_block_key(key: wp.int64,
						 hash_mask: wp.int32,
						 hash_keys: wp.array(dtype=wp.int64),
						 hash_vals: wp.array(dtype=wp.int32),
						 EMPTY_KEY: wp.int64,
						 counter: wp.array(dtype=wp.int32),
						 compacted_block_keys: wp.array(dtype=wp.int64),
						 max_blocks: wp.int32,
						 overflow: wp.array(dtype=wp.int32)
						 ):
	slot = hash64(key) & hash_mask

	flag = wp.int32(0)
	for _ in range(64): # linear probe limit
		old = wp.atomic_cas(hash_keys, slot, EMPTY_KEY, key)
		if old == EMPTY_KEY:
			# Insert a new key
			bid = wp.atomic_add(counter, 0, 1)
			if bid < max_blocks:
				hash_vals[slot] = bid
				compacted_block_keys[bid] = key
				flag = 1
			else:
				hash_vals[slot] = wp.int32(-1)
				wp.atomic_max(overflow, 0, 1) # error
			break
		elif old == key: # The same key has been checked before
			flag = 1
			break
		else: # A different key is found. Linear probing
			slot = (slot + 1) & hash_mask

		# early out
		if overflow[0]==1:
			break

	if flag == 0:
		wp.atomic_max(overflow, 0, 1)


# Rebuild the hash table without modifying the table size
def rebuild_first():
	global hash_keys, hash_vals, counter, overflow, particles, hash_mask, compacted_block_keys, max_blocks

	# Clear hash and reset counter
	hash_keys.fill_(-1)
	hash_vals.fill_(-1)
	counter.fill_(0)
	overflow.fill_(0)

	# mark active blocks
	wp.launch(kernel=mark_active_blocks,
			  dim=n_particles,
			  inputs=[particles, overh, B, BITS, MASK, BIAS, EMPTY_KEY, hash_mask, hash_keys, hash_vals, counter, compacted_block_keys, max_blocks, overflow])

# Keep increasing the table size until no overflow
def increase_table_size():
	global max_blocks, max_nodes, n_blocks, hash_capacity, hash_mask, hash_keys, hash_vals, compacted_block_keys, counter, overflow, overflowed
	loop_counter = 0
	while (max_blocks < n_blocks) or (overflowed):
		max_blocks *= 2
		max_nodes = max_blocks * nodes_per_block
		hash_capacity = max_blocks * 2
		hash_mask = hash_capacity - 1

		# Delete and regenerate
		del hash_keys, hash_vals, compacted_block_keys
		hash_keys = wp.zeros(shape=hash_capacity, dtype=wp.int64)
		hash_vals = wp.zeros(shape=hash_capacity, dtype=wp.int32)
		compacted_block_keys = wp.zeros(shape=max_blocks, dtype=wp.int64)

		# Clear hash and reset counter
		hash_keys.fill_(-1)
		hash_vals.fill_(-1)
		counter.fill_(0)
		overflow.fill_(0)

		# mark acive blocks
		wp.launch(kernel=mark_active_blocks,
				  dim=n_particles,
				  inputs=[particles, overh, B, BITS, MASK, BIAS, EMPTY_KEY, hash_mask, hash_keys, hash_vals, counter, compacted_block_keys, max_blocks, overflow])

		# Get new n_blocks
		n_blocks = counter.numpy()[0]
		overflowed = overflow.numpy()[0]

		loop_counter += 1

# (i, j, k) -> grid id
@wp.func
def node_to_gid(ix: wp.int32, iy: wp.int32, iz: wp.int32,
				B: wp.int32,
				BITS: wp.int32,
				MASK: wp.int64,
				BIAS: wp.int32,
				hash_mask: wp.int32,
				hash_keys: wp.array(dtype=wp.int64),
				hash_vals: wp.array(dtype=wp.int32),
				EMPTY_KEY: wp.int64) -> wp.int32:
	return_gid = wp.int32(-1)

	# Block i, j, k
	bx = ix // B
	by = iy // B
	bz = iz // B

	# Local index
	lx = ix - bx * B
	ly = iy - by * B
	lz = iz - bz * B

	# Flattened local id
	lid = lx + B * (ly + B * lz)

	# Block (i, j, k) -> key
	key = pack(bx, by, bz, BITS, MASK, BIAS)

	# Key -> block id
	bid = find_block_id(key, hash_mask, hash_keys, hash_vals, EMPTY_KEY)
	if bid < 0:
		return_gid = wp.int32(-1)
	else:
		return_gid = bid * (B*B*B) + lid

	return return_gid

# Find the block id from key
@wp.func
def find_block_id(key: wp.int64,
				  hash_mask: wp.int32,
				  hash_keys: wp.array(dtype=wp.int64),
				  hash_vals: wp.array(dtype=wp.int32),
				  EMPTY_KEY: wp.int64) -> wp.int32:
	slot = hash64(key) & hash_mask

	return_block_id = wp.int32(-1)
	for _ in range(64):
		k = hash_keys[slot]
		if k == key:
			return_block_id = hash_vals[slot]
			break
		if k == EMPTY_KEY:
			break

		slot = (slot + 1) & hash_mask


	return return_block_id

# Grid id -> node (i, j, k)
@wp.func
def gid_to_ijk(gid: wp.int32,
			   nodes_per_block: wp.int32,
			   B: wp.int32,
			   compacted_block_keys: wp.array(dtype=wp.int64),
			   BITS: wp.int32,
			   MASK: wp.int64,
			   BIAS: wp.int32
			   ) -> wp.vec3i:
	bid = gid // nodes_per_block
	lid = gid - bid * nodes_per_block

	# local coordinates
	lxyz = lid_to_local(lid, B)
	lx = lxyz[0]
	ly = lxyz[1]
	lz = lxyz[2]

	# block coordinates
	key = compacted_block_keys[bid]
	bxyz = unpack(key, BITS, MASK, BIAS)
	bx = bxyz[0]
	by = bxyz[1]
	bz = bxyz[2]

	# global node coordinates
	ix = bx * B + lx
	iy = by * B + ly
	iz = bz * B + lz

	ijk = wp.vec3i(ix, iy, iz)

	return ijk

# Flattened local id to vector format
@wp.func
def lid_to_local(lid: wp.int32,
				 B: wp.int32) -> wp.vec3i:
	lx = lid % B
	ly = (lid // B) % B
	lz = lid // (B * B)

	lxyz = wp.vec3i(lx, ly, lz)

	return lxyz

# key -> block (i, j, k)
@wp.func
def unpack(key: wp.int64,
			BITS: wp.int32,
			MASK: wp.int64,
			BIAS: wp.int32) -> wp.vec3i:
	uz = key & MASK
	uy = (key >> wp.int64(BITS)) & MASK
	ux = (key >> wp.int64(2 * BITS)) & MASK

	bx = wp.int32(ux) - BIAS
	by = wp.int32(uy) - BIAS
	bz = wp.int32(uz) - BIAS

	bxyz = wp.vec3i(bx, by, bz)

	return bxyz

# Compute the distance for each node to the terrain
@wp.kernel
def distance_to_terrain(grid: Grid,
						nodes_per_block: wp.int32,
						B: wp.int32,
						compacted_block_keys: wp.array(dtype=wp.int64),
						BITS: wp.int32,
						MASK: wp.int64,
						BIAS: wp.int32,
						h: wp.float64,
						terrain: wp.uint64
						):
	gid = wp.tid()

	ijk = gid_to_ijk(gid, nodes_per_block, B, compacted_block_keys, BITS, MASK, BIAS)
	i = ijk[0]
	j = ijk[1]
	k = ijk[2]

	node_x = wp.float(i) * wp.float(h)
	node_y = wp.float(j) * wp.float(h)
	node_z = wp.float(k) * wp.float(h)

	node = wp.vec3(node_x, node_y, node_z)
	node_local = wp.volume_world_to_index(terrain, node)

	signed_distance = wp.volume_sample(terrain, node_local, wp.Volume.LINEAR, dtype=wp.float32)
	signed_distance_64 = wp.float64(signed_distance)

	grid.terrain_phi[gid] = signed_distance_64

# Particle-to-grid transfer
@wp.kernel
def P2G(particles: Particles,
		grid: Grid,
		material: Material,
		h: wp.float64,
		overh: wp.float64,
		B: wp.int32,
		BITS: wp.int32,
		MASK: wp.int64,
		BIAS: wp.int32,
		hash_mask: wp.int32,
		hash_keys: wp.array(dtype=wp.int64),
		hash_vals: wp.array(dtype=wp.int32),
		EMPTY_KEY: wp.int64
		):
	p = wp.tid()

	xp = particles.x[p]

	# Quadratc B-splines base index
	base_x = xp[0] * overh - wp.float64(0.5)
	base_y = xp[1] * overh - wp.float64(0.5)
	base_z = xp[2] * overh - wp.float64(0.5)

	base_int = wp.vector(wp.int32(wp.floor(base_x)), wp.int32(wp.floor(base_y)), wp.int32(wp.floor(base_z)))
	base = wp.vector(wp.float64(base_int[0]), wp.float64(base_int[1]), wp.float64(base_int[2]))

	fx = xp * overh - base

	w = wp.matrix(
		wp.float64(0.5)*(wp.float64(1.5)-fx[0])*(wp.float64(1.5)-fx[0]), wp.float64(0.5)*(wp.float64(1.5)-fx[1])*(wp.float64(1.5)-fx[1]), wp.float64(0.5)*(wp.float64(1.5)-fx[2])*(wp.float64(1.5)-fx[2]),
		wp.float64(0.75)-(fx[0]-wp.float64(1.0))*(fx[0]-wp.float64(1.0)), wp.float64(0.75)-(fx[1]-wp.float64(1.0))*(fx[1]-wp.float64(1.0)), wp.float64(0.75)-(fx[2]-wp.float64(1.0))*(fx[2]-wp.float64(1.0)),
		wp.float64(0.5)*(fx[0]-wp.float64(0.5))*(fx[0]-wp.float64(0.5)), wp.float64(0.5)*(fx[1]-wp.float64(0.5))*(fx[1]-wp.float64(0.5)), wp.float64(0.5)*(fx[2]-wp.float64(0.5))*(fx[2]-wp.float64(0.5)), 
		shape=(3, 3)
	)

	grad_w = wp.matrix(
		(fx[0]-wp.float64(1.5))*overh, (fx[1]-wp.float64(1.5))*overh, (fx[2]-wp.float64(1.5))*overh,
		(wp.float64(2.0)-wp.float64(2.0)*fx[0])*overh, (wp.float64(2.0)-wp.float64(2.0)*fx[1])*overh, (wp.float64(2.0)-wp.float64(2.0)*fx[2])*overh,
		(fx[0]-wp.float64(0.5))*overh, (fx[1]-wp.float64(0.5))*overh, (fx[2]-wp.float64(0.5))*overh,
		shape=(3, 3)
		)

	# Stress
	Kirchhoff_stress = particles.Kirchhoff_stress[p]


	# Loop on grid nodes
	for flattened_loop in range(0, 27):
		i = wp.int(flattened_loop / 9)
		j = wp.mod(wp.int(flattened_loop / 3), 3)
		k = wp.mod(flattened_loop, 3)

		ix = base_int[0] + i
		iy = base_int[1] + j
		iz = base_int[2] + k

		node = wp.vec3d(wp.float64(ix)*h, wp.float64(iy)*h, wp.float64(iz)*h)

		weight = w[i][0] * w[j][1] * w[k][2]
		weight_grad = wp.vec3d(grad_w[i][0]*w[j][1]*w[k][2], w[i][0]*grad_w[j][1]*w[k][2], w[i][0]*w[j][1]*grad_w[k][2])

		gid = node_to_gid(ix, iy, iz, B, BITS, MASK, BIAS, hash_mask, hash_keys, hash_vals, EMPTY_KEY)

		if gid < 0: # Error
			continue

		# P2G
		P2G_momentum = weight * material.p_mass * (particles.v[p] + particles.Affine_C[p]@(node - xp))
		P2G_mass = weight * material.p_mass
		P2G_force = -particles.volume_initial * Kirchhoff_stress * weight_grad
		wp.atomic_add(grid.old_velocity, gid, P2G_momentum)
		wp.atomic_add(grid.mass, gid, P2G_mass)
		wp.atomic_add(grid.force, gid, P2G_force)

# Get nodal new velocity
@wp.kernel
def solve_P2G(grid: Grid,
			  dt: wp.float64):
	gid = wp.tid()

	theta = wp.float64(30.)*wp.float64(wp.pi)/wp.float64(180.)
	standard_gravity = wp.vec3d(wp.float64(9.81)*wp.sin(theta), wp.float64(-9.81)*wp.cos(theta), wp.float64(0.))

	if grid.mass[gid] > wp.float64(1e-8):
		old_v = grid.old_velocity[gid] / grid.mass[gid]

		new_v = old_v + dt*(standard_gravity + grid.force[gid]/grid.mass[gid])

		grid.old_velocity[gid] = old_v
		grid.new_velocity[gid] = new_v 

# Impose boundary constraints from the terrain
@wp.kernel
def impose_boundaries_from_terrain(grid: Grid,
								   nodes_per_block: wp.int32,
								   B: wp.int32,
								   compacted_block_keys: wp.array(dtype=wp.int64),
								   BITS: wp.int32,
								   MASK: wp.int64,
								   BIAS: wp.int32,
								   h: wp.float64,
								   terrain: wp.uint64,
								   material: Material
								   ):
	gid = wp.tid()

	ijk = gid_to_ijk(gid, nodes_per_block, B, compacted_block_keys, BITS, MASK, BIAS)
	i = ijk[0]
	j = ijk[1]
	k = ijk[2]

	if grid.mass[gid]>wp.float64(1e-7) and grid.terrain_phi[gid]<wp.float64(0.):
		# Get level set quantities
		node_x = wp.float(i) * wp.float(h)
		node_y = wp.float(j) * wp.float(h)
		node_z = wp.float(k) * wp.float(h)

		node = wp.vec3(node_x, node_y, node_z)
		# new nodal position
		node = node + dt * wp.vec3(grid.new_velocity[gid])
		node_local = wp.volume_world_to_index(terrain, node)

		level_set_grad_f32 = wp.vec3()
		signed_distance_f32 = wp.volume_sample_grad(terrain, node_local, wp.Volume.LINEAR, level_set_grad_f32, dtype=wp.float32)
		signed_distance = wp.float64(signed_distance_f32)
		level_set_grad_f32 = wp.normalize(level_set_grad_f32) # NOTE: remember to normalize
		level_set_grad = wp.vec3d(level_set_grad_f32)

		# new v
		this_node_new_v = grid.new_velocity[gid]
		this_node_new_v_n = wp.dot(this_node_new_v, level_set_grad) * level_set_grad
		this_node_new_v_t = this_node_new_v - this_node_new_v_n
		this_node_friction_mu = material.basal_friction_mu
		if wp.dot(this_node_new_v, level_set_grad) < wp.float64(0.):
			if wp.sqrt(wp.dot(this_node_new_v_t, this_node_new_v_t))<=this_node_friction_mu * wp.abs(wp.dot(this_node_new_v, level_set_grad)):
				grid.new_velocity[gid] = wp.vec3d()
			else:
				grid.new_velocity[gid] = this_node_new_v_t - this_node_friction_mu*wp.abs(wp.dot(this_node_new_v, level_set_grad))*wp.normalize(this_node_new_v_t)

# Grid-to-particle transfer
@wp.kernel
def G2P(particles: Particles,
		grid: Grid,
		material: Material,
		overh: wp.float64,
		h: wp.float64,
		B: wp.int32,
		BITS: wp.int32,
		MASK: wp.int64,
		BIAS: wp.int32,
		hash_mask: wp.int32,
		hash_keys: wp.array(dtype=wp.int64),
		hash_vals: wp.array(dtype=wp.int32),
		EMPTY_KEY: wp.int64,
		dt: wp.float64
		):
	p = wp.tid()

	xp = particles.x[p]

	overh2 = overh * overh

	# Quadratc B-splines base index
	base_x = xp[0] * overh - wp.float64(0.5)
	base_y = xp[1] * overh - wp.float64(0.5)
	base_z = xp[2] * overh - wp.float64(0.5)

	base_int = wp.vector(wp.int32(wp.floor(base_x)), wp.int32(wp.floor(base_y)), wp.int32(wp.floor(base_z)))
	base = wp.vector(wp.float64(base_int[0]), wp.float64(base_int[1]), wp.float64(base_int[2]))

	fx = xp * overh - base

	w = wp.matrix(
		wp.float64(0.5)*(wp.float64(1.5)-fx[0])*(wp.float64(1.5)-fx[0]), wp.float64(0.5)*(wp.float64(1.5)-fx[1])*(wp.float64(1.5)-fx[1]), wp.float64(0.5)*(wp.float64(1.5)-fx[2])*(wp.float64(1.5)-fx[2]),
		wp.float64(0.75)-(fx[0]-wp.float64(1.0))*(fx[0]-wp.float64(1.0)), wp.float64(0.75)-(fx[1]-wp.float64(1.0))*(fx[1]-wp.float64(1.0)), wp.float64(0.75)-(fx[2]-wp.float64(1.0))*(fx[2]-wp.float64(1.0)),
		wp.float64(0.5)*(fx[0]-wp.float64(0.5))*(fx[0]-wp.float64(0.5)), wp.float64(0.5)*(fx[1]-wp.float64(0.5))*(fx[1]-wp.float64(0.5)), wp.float64(0.5)*(fx[2]-wp.float64(0.5))*(fx[2]-wp.float64(0.5)), 
		shape=(3, 3)
	)

	grad_w = wp.matrix(
		(fx[0]-wp.float64(1.5))*overh, (fx[1]-wp.float64(1.5))*overh, (fx[2]-wp.float64(1.5))*overh,
		(wp.float64(2.0)-wp.float64(2.0)*fx[0])*overh, (wp.float64(2.0)-wp.float64(2.0)*fx[1])*overh, (wp.float64(2.0)-wp.float64(2.0)*fx[2])*overh,
		(fx[0]-wp.float64(0.5))*overh, (fx[1]-wp.float64(0.5))*overh, (fx[2]-wp.float64(0.5))*overh,
		shape=(3, 3)
		)

	# G2P
	new_FLIP_v = particles.v[p]
	new_PIC_v = wp.vec3d()
	new_grad_v = wp.mat33d()
	new_C = wp.mat33d()
	for flattened_loop in range(0, 27):
		i = wp.int(flattened_loop / 9)
		j = wp.mod(wp.int(flattened_loop / 3), 3)
		k = wp.mod(flattened_loop, 3)

		ix = base_int[0] + i
		iy = base_int[1] + j
		iz = base_int[2] + k

		node = wp.vec3d(wp.float64(ix)*h, wp.float64(iy)*h, wp.float64(iz)*h)
		dpos = node - xp

		weight = w[i][0] * w[j][1] * w[k][2]
		weight_grad = wp.vec3d(grad_w[i][0]*w[j][1]*w[k][2], w[i][0]*grad_w[j][1]*w[k][2], w[i][0]*w[j][1]*grad_w[k][2])

		gid = node_to_gid(ix, iy, iz, B, BITS, MASK, BIAS, hash_mask, hash_keys, hash_vals, EMPTY_KEY)

		if gid < 0: # Error
			continue

		# G2P
		g_v = grid.new_velocity[gid]
		g_old_v = grid.old_velocity[gid]
		new_FLIP_v += weight * (g_v - g_old_v)
		new_PIC_v += weight * g_v
		new_grad_v += wp.outer(g_v, weight_grad)
		new_C += wp.float64(4.) * weight * wp.outer(g_v, dpos) * overh2

	# Update constitutive model
	I33 = wp.mat33d(wp.float64(1.), wp.float64(0.), wp.float64(0.),
					wp.float64(0.), wp.float64(1.), wp.float64(0.),
					wp.float64(0.), wp.float64(0.), wp.float64(1.)
					)
	delta_F = I33 + dt * new_grad_v
	new_F_trial = delta_F * particles.deformation_gradient[p]

	# Return mapping
	U = wp.mat33d()
	V = wp.mat33d()
	sig = wp.vec3d()
	wp.svd3(new_F_trial, U, sig, V)

	e_trial = wp.vec3d(wp.log(sig[0]), wp.log(sig[1]), wp.log(sig[2]))
	e_real = return_mapping_DruckerPrager(e_trial, material, particles, p)

	e_real_exp_matrix = wp.mat33d(wp.exp(e_real[0]), wp.float64(0.), wp.float64(0.),
								  wp.float64(0.), wp.exp(e_real[1]), wp.float64(0.),
								  wp.float64(0.), wp.float64(0.), wp.exp(e_real[2]))

	new_F = U * e_real_exp_matrix * wp.transpose(V)
	particles.deformation_gradient[p] = new_F

	# Get new stress
	e_trace = e_real[0] + e_real[1] + e_real[2]
	Kirchhoff_principal = material.lame_lambda * e_trace * wp.vec3d(wp.float64(1.), wp.float64(1.), wp.float64(1.)) + wp.float64(2.) * material.lame_mu * e_real
	Kirchhoff_principal_matrix = wp.mat33d(
								 Kirchhoff_principal[0], wp.float64(0.), wp.float64(0.),
								 wp.float64(0.), Kirchhoff_principal[1], wp.float64(0.),
								 wp.float64(0.), wp.float64(0.), Kirchhoff_principal[2]
								 )
	Kirchhoff_stress = U * Kirchhoff_principal_matrix * wp.transpose(U)
	particles.Kirchhoff_stress[p] = Kirchhoff_stress


	particles.v[p] = material.FLIP_ratio * new_FLIP_v + (wp.float64(1.)-material.FLIP_ratio) * new_PIC_v
	particles.Affine_C[p] = new_C 
	particles.x[p] = particles.x[p] + dt * new_PIC_v


@wp.func
def return_mapping_DruckerPrager(principal_trial_strain: wp.vec3d,
								 material: Material,
								 particles: Particles,
								 p: wp.int32) -> wp.vec3d:
	float64_pi = wp.float64(3.141592653)
	tol = wp.float64(1e-8)

	# Calculate trial stress
	K = material.K
	lame_mu = material.lame_mu
	friction_angle = material.friction_angle

	principal_real_strain = principal_trial_strain

	principal_stress = grad_Psi(principal_trial_strain, K, lame_mu)
	principal_stress_matrix = wp.mat33d(principal_stress[0], wp.float64(0.), wp.float64(0.),
										wp.float64(0.), principal_stress[1], wp.float64(0.),
										wp.float64(0.), wp.float64(0.), principal_stress[2])

	P = wp.float64(1.)/wp.float64(3.) * (principal_stress[0] + principal_stress[1] + principal_stress[2])
	S_trial = principal_stress_matrix - P * wp.identity(n=3, dtype=wp.float64)
	S_trial_norm = wp.sqrt(S_trial[0,0]*S_trial[0,0] + S_trial[1,1]*S_trial[1,1] + S_trial[2,2]*S_trial[2,2])
	Q_trial = wp.sqrt(wp.float64(3.)/wp.float64(2.)) * S_trial_norm

	# Return mapping
	friction_coefficient = wp.float64(2.)*wp.sqrt(wp.float64(6.))*wp.sin(friction_angle*float64_pi/wp.float64(180.)) / (wp.float64(3.)-wp.sin(friction_angle*float64_pi/wp.float64(180.)))
	yield_function = wp.sqrt(wp.float64(2.)/wp.float64(3.)) * Q_trial + friction_coefficient * P

	if yield_function <= wp.float64(0.): # Elasticity
		principal_real_strain = principal_trial_strain
	elif yield_function > wp.float64(0.) and (principal_trial_strain[0]+principal_trial_strain[1]+principal_trial_strain[2]) > wp.float64(0.):
		delta_lambda = wp.sqrt(principal_trial_strain[0]*principal_trial_strain[0] + principal_trial_strain[1]*principal_trial_strain[1] + principal_trial_strain[2]*principal_trial_strain[2])
		principal_real_strain = wp.vec3d()
	elif yield_function > wp.float64(0.): # Plasticity
		delta_lambda = yield_function / (wp.float64(2.)*lame_mu)

		n = wp.vec3d()
		if S_trial_norm > wp.float64(0.):
			n[0] = S_trial[0,0]/S_trial_norm
			n[1] = S_trial[1,1]/S_trial_norm
			n[2] = S_trial[2,2]/S_trial_norm
		principal_real_strain = principal_trial_strain - delta_lambda * n

	return principal_real_strain

# Gradient potential to get stress
@wp.func
def grad_Psi(principal_strain: wp.vec3d,
			 K: wp.float64,
			 lame_mu: wp.float64
			 ) -> wp.vec3d:
	eps_v = principal_strain[0] + principal_strain[1] + principal_strain[2]
	eps_s = wp.sqrt(wp.float64(2.0)/wp.float64(9.0) * ((principal_strain[0]-principal_strain[1])*(principal_strain[0]-principal_strain[1]) + (principal_strain[1]-principal_strain[2])*(principal_strain[1]-principal_strain[2]) + (principal_strain[0]-principal_strain[2])*(principal_strain[0]-principal_strain[2])))

	P = K * eps_v
	Q = wp.float64(3.) * lame_mu * eps_s

	principal_stress = wp.vec3d()
	principal_stress[0] = P + wp.float64(2.0)/wp.float64(3.0) * lame_mu * (wp.float64(2.0)*principal_strain[0] - principal_strain[1] - principal_strain[2])
	principal_stress[1] = P + wp.float64(2.0)/wp.float64(3.0) * lame_mu * (wp.float64(2.0)*principal_strain[1] - principal_strain[0] - principal_strain[2])
	principal_stress[2] = P + wp.float64(2.0)/wp.float64(3.0) * lame_mu * (wp.float64(2.0)*principal_strain[2] - principal_strain[0] - principal_strain[1])

	return principal_stress


# ============ Main ============
output_frame = 0

for step in range(6250):

	with timer("rebuild_grid"):
		# Try to rebuild the hash table without changing its size
		rebuild_first()

		n_blocks = counter.numpy()[0]
		overflowed = overflow.numpy()[0]
		if (overflowed) or (n_blocks > max_blocks):
			# increase the block size
			increase_table_size()

			# Reinitialize background grid
			del grid
			grid = Grid()
			grid.mass = wp.zeros(shape=max_nodes, dtype=wp.float64)
			grid.old_velocity = wp.zeros(shape=max_nodes, dtype=wp.vec3d)
			grid.new_velocity = wp.zeros(shape=max_nodes, dtype=wp.vec3d)
			grid.force = wp.zeros(shape=max_nodes, dtype=wp.vec3d)
			grid.terrain_phi = wp.zeros(shape=max_nodes, dtype=wp.float64)
		else:
			# Reset grid
			grid.mass.zero_()
			grid.old_velocity.zero_()
			grid.new_velocity.zero_()
			grid.force.zero_()
			grid.terrain_phi.zero_()

		n_nodes_used = n_blocks * nodes_per_block
		n_nodes_used_int = int(n_nodes_used)



	with timer("distance_to_terrain"):
		# Compute distance to the terrain
		wp.launch(kernel=distance_to_terrain,
				  dim=n_nodes_used_int,
				  inputs=[grid, nodes_per_block, B, compacted_block_keys, BITS, MASK, BIAS, h, terrain.id])


	with timer("p2g"):
		# P2G
		wp.launch(kernel=P2G,
				  dim=n_particles,
				  inputs=[particles, grid, material, h, overh, B, BITS, MASK, BIAS, hash_mask, hash_keys, hash_vals, EMPTY_KEY])

	with timer("solve_P2G"):
		# Solve P2G
		wp.launch(kernel=solve_P2G,
				  dim=n_nodes_used_int,
				  inputs=[grid, dt])

	with timer("impose_boundaries_from_terrain"):
		# Impose boundaries
		wp.launch(kernel=impose_boundaries_from_terrain,
				  dim=n_nodes_used_int,
				  inputs=[grid, nodes_per_block, B, compacted_block_keys, BITS, MASK, BIAS, h, terrain.id, material])

	with timer("g2p"):
		# G2P
		wp.launch(kernel=G2P,
				  dim=n_particles,
				  inputs=[particles, grid, material, overh, h, B, BITS, MASK, BIAS, hash_mask, hash_keys, hash_vals, EMPTY_KEY, dt])

	with timer("post_processing"):

		if step % 55 == 0:
			print("Frame:", output_frame, "n_nodes_used:", n_nodes_used_int, "max_nodes:", max_nodes)
			x = np.array(particles.x.numpy())
			output_particles = meshio.Mesh(points=x, cells=[], point_data={"v_mag": np.linalg.norm(particles.v.numpy(), axis=1)})
			output_particles.write("output/mountain_%04d.ply" % (output_frame))

			output_frame += 1




print(timer.report())


