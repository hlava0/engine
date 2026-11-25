//! Interactive 3D Archery Range game with walking, jumping, bow charging,
//! projectile arrow physics, target scoring, audio chimes, and an in-game
//! GPU-rendered HUD & Graphics Settings Menu.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Instant;

use engine_core::{
    CameraState, Engine, FramePacer, InstanceDesc, LightDesc, MaterialDesc, SceneWorld,
    create_cube_mesh, create_plane_mesh, create_sphere_mesh,
};
use engine_gameplay::{
    AudioEngine, Collider, GameWorld, MeshRenderer, Name, PhysicsConfig, RigidBody, Transform,
    step_physics,
};
use engine_gpu_driven::types::mat4_translation_scale;
use engine_rhi::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;
const FLOOR_Y: f32 = -1.0;
const EYE_HEIGHT: f32 = 1.15;
const ARROW_POOL_SIZE: usize = 16;

struct PlayerController {
    position: [f32; 3],
    vel_y: f32,
    grounded: bool,
    yaw: f32,
    pitch: f32,
    walk_speed: f32,
    sprint_mult: f32,
    mouse_sens: f32,
}

impl PlayerController {
    fn new() -> Self {
        Self {
            position: [0.0, FLOOR_Y + EYE_HEIGHT, 6.5],
            vel_y: 0.0,
            grounded: true,
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: -0.04,
            walk_speed: 4.8,
            sprint_mult: 1.65,
            mouse_sens: 1.0,
        }
    }

    fn forward_3d(&self) -> [f32; 3] {
        let cp = self.pitch.cos();
        [self.yaw.cos() * cp, self.pitch.sin(), self.yaw.sin() * cp]
    }

    fn forward_xz(&self) -> [f32; 3] {
        [self.yaw.cos(), 0.0, self.yaw.sin()]
    }

    fn right_xz(&self) -> [f32; 3] {
        [-self.yaw.sin(), 0.0, self.yaw.cos()]
    }

    fn update(&mut self, keys: &HashSet<KeyCode>, dt: f32) {
        let look_speed = 1.9 * self.mouse_sens;
        if keys.contains(&KeyCode::ArrowLeft) {
            self.yaw -= look_speed * dt;
        }
        if keys.contains(&KeyCode::ArrowRight) {
            self.yaw += look_speed * dt;
        }
        if keys.contains(&KeyCode::ArrowUp) {
            self.pitch = (self.pitch + look_speed * dt).clamp(-1.42, 1.42);
        }
        if keys.contains(&KeyCode::ArrowDown) {
            self.pitch = (self.pitch - look_speed * dt).clamp(-1.42, 1.42);
        }

        let fwd = self.forward_xz();
        let right = self.right_xz();
        let sprint = if keys.contains(&KeyCode::ShiftLeft) || keys.contains(&KeyCode::ShiftRight) {
            self.sprint_mult
        } else {
            1.0
        };
        let speed = self.walk_speed * sprint * dt;

        let mut move_dir = [0.0f32; 3];
        if keys.contains(&KeyCode::KeyW) {
            move_dir[0] += fwd[0];
            move_dir[2] += fwd[2];
        }
        if keys.contains(&KeyCode::KeyS) {
            move_dir[0] -= fwd[0];
            move_dir[2] -= fwd[2];
        }
        if keys.contains(&KeyCode::KeyD) {
            move_dir[0] += right[0];
            move_dir[2] += right[2];
        }
        if keys.contains(&KeyCode::KeyA) {
            move_dir[0] -= right[0];
            move_dir[2] -= right[2];
        }

        let len = (move_dir[0] * move_dir[0] + move_dir[2] * move_dir[2]).sqrt();
        if len > 1e-4 {
            self.position[0] += (move_dir[0] / len) * speed;
            self.position[2] += (move_dir[2] / len) * speed;
        }

        // Jump & gravity
        if keys.contains(&KeyCode::Space) && self.grounded {
            self.vel_y = 6.2;
            self.grounded = false;
        }
        self.vel_y -= 18.0 * dt;
        self.position[1] += self.vel_y * dt;
        let min_y = FLOOR_Y + EYE_HEIGHT;
        if self.position[1] <= min_y {
            self.position[1] = min_y;
            self.vel_y = 0.0;
            self.grounded = true;
        }

        // Clamp inside courtyard arena
        self.position[0] = self.position[0].clamp(-12.5, 12.5);
        self.position[2] = self.position[2].clamp(-1.0, 12.0);
    }

    fn apply_to_scene(&self, scene: &mut SceneWorld) {
        let fwd = self.forward_3d();
        scene.camera.position = self.position;
        scene.camera.target = [
            self.position[0] + fwd[0],
            self.position[1] + fwd[1],
            self.position[2] + fwd[2],
        ];
    }
}

#[derive(Clone, Copy)]
struct ArrowState {
    instance_index: usize,
    active: bool,
    stuck: bool,
    pos: [f32; 3],
    vel: [f32; 3],
    age: f32,
}

#[derive(Clone, Copy)]
struct TargetState {
    instance_index: usize,
    base_pos: [f32; 3],
    pos: [f32; 3],
    radius: f32,
    amp_x: f32,
    amp_y: f32,
    freq: f32,
    phase: f32,
    points: u32,
    hit_cooldown: f32,
}

fn orient_arrow_matrix(pos: [f32; 3], dir: [f32; 3]) -> [[f32; 4]; 4] {
    let dlen = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt().max(1e-5);
    let f = [dir[0] / dlen, dir[1] / dlen, dir[2] / dlen];
    let world_up = if f[1].abs() > 0.96 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let rx = world_up[1] * f[2] - world_up[2] * f[1];
    let ry = world_up[2] * f[0] - world_up[0] * f[2];
    let rz = world_up[0] * f[1] - world_up[1] * f[0];
    let rlen = (rx * rx + ry * ry + rz * rz).sqrt().max(1e-5);
    let r = [rx / rlen, ry / rlen, rz / rlen];
    let u = [
        f[1] * r[2] - f[2] * r[1],
        f[2] * r[0] - f[0] * r[2],
        f[0] * r[1] - f[1] * r[0],
    ];

    let sx = 0.055;
    let sy = 0.055;
    let sz = 0.85;
    [
        [r[0] * sx, r[1] * sx, r[2] * sx, 0.0],
        [u[0] * sy, u[1] * sy, u[2] * sy, 0.0],
        [f[0] * sz, f[1] * sz, f[2] * sz, 0.0],
        [pos[0], pos[1], pos[2], 1.0],
    ]
}

fn build_archery_scene(
    width: u32,
    height: u32,
) -> (SceneWorld, Vec<TargetState>, Vec<ArrowState>, GameWorld) {
    let cube = create_cube_mesh();
    let plane = create_plane_mesh();
    let sphere = create_sphere_mesh(16, 24);
    let meshes = vec![cube, plane, sphere];

    // Materials:
    // 0: Courtyard stone floor
    // 1: Arena walls
    // 2: Gold bullseye target
    // 3: Ruby target
    // 4: Emerald target
    // 5: Sapphire target
    // 6: Chrome physics sphere
    // 7: Arrow shaft (bright amber/wood emissive)
    let materials = vec![
        MaterialDesc {
            albedo: [0.32, 0.38, 0.30],
            roughness: 0.88,
            metallic: 0.0,
            emission: [0.0; 3],
        },
        MaterialDesc {
            albedo: [0.46, 0.44, 0.40],
            roughness: 0.82,
            metallic: 0.05,
            emission: [0.0; 3],
        },
        MaterialDesc {
            albedo: [0.95, 0.78, 0.22],
            roughness: 0.18,
            metallic: 0.92,
            emission: [0.35, 0.25, 0.04],
        },
        MaterialDesc {
            albedo: [0.90, 0.20, 0.22],
            roughness: 0.35,
            metallic: 0.15,
            emission: [0.25, 0.04, 0.04],
        },
        MaterialDesc {
            albedo: [0.18, 0.85, 0.38],
            roughness: 0.35,
            metallic: 0.15,
            emission: [0.03, 0.22, 0.08],
        },
        MaterialDesc {
            albedo: [0.22, 0.52, 0.95],
            roughness: 0.28,
            metallic: 0.35,
            emission: [0.04, 0.12, 0.28],
        },
        MaterialDesc {
            albedo: [0.85, 0.88, 0.92],
            roughness: 0.12,
            metallic: 0.95,
            emission: [0.0; 3],
        },
        MaterialDesc {
            albedo: [0.95, 0.65, 0.22],
            roughness: 0.40,
            metallic: 0.10,
            emission: [0.65, 0.42, 0.10],
        },
    ];

    let mut instances = Vec::new();
    let mut push_inst = |pos: [f32; 3], scale: [f32; 3], mesh_id: u32, material_id: u32| -> usize {
        let idx = instances.len();
        let t = mat4_translation_scale(pos, scale);
        let bs_r = meshes[mesh_id as usize].bounding_sphere[3];
        let max_s = scale[0].abs().max(scale[1].abs()).max(scale[2].abs());
        instances.push(InstanceDesc {
            transform: t,
            prev_transform: t,
            bounding_sphere: [0.0, 0.0, 0.0, bs_r * max_s],
            mesh_id,
            material_id,
            animated: false,
            base_pos: pos,
            base_scale: scale,
        });
        idx
    };

    // Arena floor & walls
    push_inst([0.0, FLOOR_Y, -3.0], [34.0, 1.0, 36.0], 1, 0);
    push_inst([0.0, 2.5, -20.0], [34.0, 7.0, 1.0], 0, 1); // Backstop wall
    push_inst([-14.5, 2.0, -3.0], [1.0, 6.0, 34.0], 0, 1); // Left wall
    push_inst([14.5, 2.0, -3.0], [1.0, 6.0, 34.0], 0, 1); // Right wall

    // 5 Archery targets (varied distances & motion)
    let target_defs = [
        ([0.0, 0.8, -6.5], 0.95, 0.0, 0.0, 0.0, 0.0, 2u32, 100u32),
        ([-4.2, 1.2, -10.0], 1.05, 2.4, 0.0, 1.4, 0.0, 3u32, 150u32),
        ([4.2, 1.5, -11.5], 1.05, 2.2, 0.4, 1.7, 1.2, 4u32, 200u32),
        ([0.0, 2.2, -15.5], 1.15, 4.5, 0.6, 2.1, 0.7, 5u32, 300u32),
        ([-2.5, 0.6, -18.0], 1.25, 3.0, 0.0, 2.6, 2.4, 2u32, 500u32),
    ];
    let mut targets = Vec::with_capacity(target_defs.len());
    for (pos, diam, amp_x, amp_y, freq, phase, mat_id, points) in target_defs {
        let idx = push_inst(pos, [diam, diam, 0.28], 2, mat_id);
        targets.push(TargetState {
            instance_index: idx,
            base_pos: pos,
            pos,
            radius: diam * 0.55,
            amp_x,
            amp_y,
            freq,
            phase,
            points,
            hit_cooldown: 0.0,
        });
    }

    // Interactive physics spheres
    let mut game = GameWorld::new();
    game.spawn((
        Name::new("GroundPlane"),
        Collider::GroundPlane { y: FLOOR_Y },
    ));
    let sphere_idx_1 = push_inst([-2.2, 0.5, -2.5], [1.2, 1.2, 1.2], 2, 6);
    game.spawn((
        Name::new("ChromeOrb1"),
        Transform::from_position_scale([-2.2, 0.5, -2.5], [1.2, 1.2, 1.2]),
        MeshRenderer {
            mesh_id: 2,
            material_id: 6,
            instance_index: Some(sphere_idx_1),
        },
        RigidBody::dynamic(1.0, 0.85),
        Collider::Sphere { radius: 0.6 },
    ));
    let sphere_idx_2 = push_inst([2.6, 1.2, -3.5], [1.0, 1.0, 1.0], 2, 2);
    game.spawn((
        Name::new("GoldOrb2"),
        Transform::from_position_scale([2.6, 1.2, -3.5], [1.0, 1.0, 1.0]),
        MeshRenderer {
            mesh_id: 2,
            material_id: 2,
            instance_index: Some(sphere_idx_2),
        },
        RigidBody::dynamic(1.0, 0.85),
        Collider::Sphere { radius: 0.5 },
    ));

    // Pre-allocated arrow pool (parked out of sight until fired)
    let mut arrows = Vec::with_capacity(ARROW_POOL_SIZE);
    for _ in 0..ARROW_POOL_SIZE {
        let idx = push_inst([0.0, -100.0, 0.0], [0.055, 0.055, 0.85], 0, 7);
        arrows.push(ArrowState {
            instance_index: idx,
            active: false,
            stuck: false,
            pos: [0.0, -100.0, 0.0],
            vel: [0.0; 3],
            age: 0.0,
        });
    }

    let lights = vec![
        LightDesc {
            pos: [0.0, 0.0, 0.0],
            energy: 5.5,
            dir: [-0.45, -0.82, -0.35],
            is_point: false,
            color: [1.0, 0.96, 0.88],
            range: 1000.0,
        },
        LightDesc {
            pos: [0.0, 5.0, -8.0],
            energy: 14.0,
            dir: [0.0, -1.0, 0.0],
            is_point: true,
            color: [0.95, 0.88, 0.72],
            range: 28.0,
        },
    ];

    let scene = SceneWorld {
        meshes,
        materials,
        instances,
        lights,
        camera: CameraState::new([0.0, FLOOR_Y + EYE_HEIGHT, 6.5], [0.0, 0.5, -6.0], 65.0, width, height),
        time: 0.0,
        dirty_instances: true,
    };

    (scene, targets, arrows, game)
}

struct ArcheryApp {
    enable_validation: bool,
    window: Option<Arc<Window>>,
    engine: Option<Engine>,
    game: GameWorld,
    physics_cfg: PhysicsConfig,
    audio: AudioEngine,
    pacer: FramePacer,
    player: PlayerController,
    targets: Vec<TargetState>,
    arrows: Vec<ArrowState>,
    next_arrow: usize,
    pressed_keys: HashSet<KeyCode>,
    cursor_locked: bool,
    drawing_bow: bool,
    bow_charge: f32,
    hit_flash: f32,
    score: u32,
    hits: u32,
    arrows_fired: u32,
    menu_open: bool,
    selected_row: u32,
    sound_enabled: bool,
    sim_time: f32,
    last_frame: Instant,
    last_title_update: Instant,
    last_gpu_ms: f64,
}

impl ArcheryApp {
    fn new(enable_validation: bool) -> Self {
        let now = Instant::now();
        Self {
            enable_validation,
            window: None,
            engine: None,
            game: GameWorld::new(),
            physics_cfg: PhysicsConfig::default(),
            audio: AudioEngine::new(),
            pacer: FramePacer::default(),
            player: PlayerController::new(),
            targets: Vec::new(),
            arrows: Vec::new(),
            next_arrow: 0,
            pressed_keys: HashSet::new(),
            cursor_locked: false,
            drawing_bow: false,
            bow_charge: 0.0,
            hit_flash: 0.0,
            score: 0,
            hits: 0,
            arrows_fired: 0,
            menu_open: false,
            selected_row: 1,
            sound_enabled: true,
            sim_time: 0.0,
            last_frame: now,
            last_title_update: now,
            last_gpu_ms: 0.0,
        }
    }

    fn set_cursor_lock(&mut self, lock: bool) {
        self.cursor_locked = lock;
        if let Some(win) = self.window.as_ref() {
            if lock {
                let _ = win
                    .set_cursor_grab(CursorGrabMode::Locked)
                    .or_else(|_| win.set_cursor_grab(CursorGrabMode::Confined));
                win.set_cursor_visible(false);
            } else {
                let _ = win.set_cursor_grab(CursorGrabMode::None);
                win.set_cursor_visible(true);
            }
        }
    }

    fn shoot_arrow(&mut self) {
        let fwd = self.player.forward_3d();
        let right = self.player.right_xz();
        let speed = 15.0 + self.bow_charge.clamp(0.0, 1.0) * 26.0;
        let spawn_pos = [
            self.player.position[0] + fwd[0] * 0.7 + right[0] * 0.14,
            self.player.position[1] + fwd[1] * 0.7 - 0.08,
            self.player.position[2] + fwd[2] * 0.7 + right[2] * 0.14,
        ];
        let vel = [fwd[0] * speed, fwd[1] * speed + 1.2, fwd[2] * speed];

        let idx = self.next_arrow % self.arrows.len();
        self.next_arrow = (self.next_arrow + 1) % self.arrows.len();
        self.arrows[idx].active = true;
        self.arrows[idx].stuck = false;
        self.arrows[idx].pos = spawn_pos;
        self.arrows[idx].vel = vel;
        self.arrows[idx].age = 0.0;
        self.arrows_fired += 1;
        if self.sound_enabled {
            self.audio.play_beep(320.0 + self.bow_charge * 180.0, 45);
        }
        self.bow_charge = 0.0;
    }

    fn adjust_menu(&mut self, dir: i32) {
        let Some(engine) = self.engine.as_mut() else {
            return;
        };
        match self.selected_row {
            1 => engine.graphics_settings.enable_rt_gi = !engine.graphics_settings.enable_rt_gi,
            2 => {
                engine.graphics_settings.enable_hiz_culling =
                    !engine.graphics_settings.enable_hiz_culling
            }
            3 => {
                let next_fov = engine.graphics_settings.fov_deg + (dir as f32) * 5.0;
                engine.graphics_settings.fov_deg = next_fov.clamp(50.0, 100.0);
            }
            4 => {
                let next_sens = self.player.mouse_sens + (dir as f32) * 0.25;
                self.player.mouse_sens = next_sens.clamp(0.25, 2.5);
            }
            5 => self.sound_enabled = !self.sound_enabled,
            6 => {
                self.score = 0;
                self.hits = 0;
                self.arrows_fired = 0;
                for a in &mut self.arrows {
                    a.active = false;
                    a.stuck = false;
                    a.pos = [0.0, -100.0, 0.0];
                }
            }
            7 => {
                self.menu_open = false;
            }
            _ => {}
        }
        if self.sound_enabled {
            self.audio.play_beep(620.0, 25);
        }
    }

    fn sync_hud_state(&mut self) {
        let Some(engine) = self.engine.as_mut() else {
            return;
        };
        let stats = self.pacer.stats();
        let active_arrows = self.arrows.iter().filter(|a| a.active).count();
        let gs = engine.graphics_settings;
        let denoiser = if engine.denoiser_backend_name().contains("NRD") {
            "NRD"
        } else {
            "SVGF"
        };

        engine.hud_state.enabled = true;
        engine.hud_state.menu_open = self.menu_open;
        engine.hud_state.selected_row = self.selected_row;
        engine.hud_state.bow_charge = self.bow_charge;
        engine.hud_state.hit_flash = self.hit_flash;
        engine.hud_state.hud_lines = [
            format!(
                "SCORE:{}  HITS:{}  ARROWS:{}/{}",
                self.score, self.hits, active_arrows, ARROW_POOL_SIZE
            ),
            format!(
                "FPS:{:.0}  GPU:{:.2}MS  {}  [M] MENU",
                stats.avg_fps, self.last_gpu_ms, denoiser
            ),
        ];
        engine.hud_state.menu_lines = vec![
            "GRAPHICS & GAME MENU  [UP/DN] [LT/RT]".to_string(),
            format!(
                "1. RT GI + REFLECTIONS : {}",
                if gs.enable_rt_gi { "ON" } else { "OFF" }
            ),
            format!(
                "2. HI-Z OCCLUSION CULL : {}",
                if gs.enable_hiz_culling { "ON" } else { "OFF" }
            ),
            format!("3. FIELD OF VIEW       : {:.0} DEG", gs.fov_deg),
            format!("4. MOUSE SENSITIVITY   : {:.2}X", self.player.mouse_sens),
            format!(
                "5. SOUND EFFECTS       : {}",
                if self.sound_enabled { "ON" } else { "OFF" }
            ),
            "6. RESET SCORE + ARROWS".to_string(),
            "7. RESUME GAME  [M / ESC]".to_string(),
        ];
    }

    fn step_simulation(&mut self, dt: f32) {
        self.sim_time += dt;
        self.hit_flash = (self.hit_flash - dt * 3.5).max(0.0);

        if self.menu_open {
            return;
        }

        self.player.update(&self.pressed_keys, dt);
        if self.drawing_bow {
            self.bow_charge = (self.bow_charge + dt / 0.65).min(1.0);
        }

        // Update moving targets
        for tgt in &mut self.targets {
            tgt.hit_cooldown = (tgt.hit_cooldown - dt).max(0.0);
            if tgt.freq > 0.0 {
                let s = (self.sim_time * tgt.freq + tgt.phase).sin();
                tgt.pos = [
                    tgt.base_pos[0] + s * tgt.amp_x,
                    tgt.base_pos[1] + (self.sim_time * tgt.freq * 1.4 + tgt.phase).cos() * tgt.amp_y,
                    tgt.base_pos[2],
                ];
            }
        }

        // Step ECS rigid-body spheres
        step_physics(&mut self.game.world, &self.physics_cfg, dt);

        // Step arrow projectiles & check collisions
        for arrow in &mut self.arrows {
            if !arrow.active || arrow.stuck {
                continue;
            }
            arrow.age += dt;
            arrow.vel[1] -= 9.81 * dt;
            arrow.pos[0] += arrow.vel[0] * dt;
            arrow.pos[1] += arrow.vel[1] * dt;
            arrow.pos[2] += arrow.vel[2] * dt;

            // Floor & arena wall collision
            if arrow.pos[1] <= FLOOR_Y + 0.06 {
                arrow.pos[1] = FLOOR_Y + 0.06;
                arrow.stuck = true;
                continue;
            }
            if arrow.pos[2] <= -19.4 || arrow.pos[0].abs() >= 13.8 || arrow.age > 8.0 {
                arrow.stuck = true;
                continue;
            }

            // Target hit check
            for tgt in &mut self.targets {
                let dx = arrow.pos[0] - tgt.pos[0];
                let dy = arrow.pos[1] - tgt.pos[1];
                let dz = arrow.pos[2] - tgt.pos[2];
                if dx * dx + dy * dy + dz * dz <= (tgt.radius + 0.15) * (tgt.radius + 0.15) {
                    arrow.stuck = true;
                    if tgt.hit_cooldown <= 0.0 {
                        tgt.hit_cooldown = 0.2;
                        self.score += tgt.points;
                        self.hits += 1;
                        self.hit_flash = 1.0;
                        if self.sound_enabled {
                            self.audio.play_beep(880.0, 70);
                        }
                    }
                    break;
                }
            }

            // Physics spheres hit impulse
            if !arrow.stuck {
                for (_, (tf, rb, col)) in self
                    .game
                    .world
                    .query_mut::<(&Transform, &mut RigidBody, &Collider)>()
                {
                    if let Collider::Sphere { radius } = *col {
                        let dx = arrow.pos[0] - tf.position[0];
                        let dy = arrow.pos[1] - tf.position[1];
                        let dz = arrow.pos[2] - tf.position[2];
                        let r = radius + 0.12;
                        if dx * dx + dy * dy + dz * dz <= r * r {
                            rb.velocity[0] += arrow.vel[0] * 0.28;
                            rb.velocity[1] += arrow.vel[1] * 0.28 + 2.2;
                            rb.velocity[2] += arrow.vel[2] * 0.28;
                            arrow.stuck = true;
                            if self.sound_enabled {
                                self.audio.play_beep(440.0, 50);
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Sync transforms into SceneWorld
        if let Some(engine) = self.engine.as_mut() {
            self.player.apply_to_scene(&mut engine.world);

            for tgt in &self.targets {
                let inst = &mut engine.world.instances[tgt.instance_index];
                inst.prev_transform = inst.transform;
                inst.transform = mat4_translation_scale(tgt.pos, inst.base_scale);
            }
            for arrow in &self.arrows {
                let inst = &mut engine.world.instances[arrow.instance_index];
                inst.prev_transform = inst.transform;
                inst.transform = if arrow.active {
                    orient_arrow_matrix(arrow.pos, arrow.vel)
                } else {
                    mat4_translation_scale([0.0, -100.0, 0.0], inst.base_scale)
                };
            }
            engine.world.sync_from_game_world(&self.game);
            engine.world.dirty_instances = true;
        }
    }
}

impl ApplicationHandler for ArcheryApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Vulkan 1.4 Archery Range — Initializing...")
            .with_inner_size(LogicalSize::new(DEFAULT_WIDTH, DEFAULT_HEIGHT));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );
        let phys = window.inner_size();
        let display_handle = window
            .display_handle()
            .expect("display handle")
            .as_raw();
        let window_handle = window.window_handle().expect("window handle").as_raw();

        let (scene, targets, arrows, game) =
            build_archery_scene(phys.width.max(1), phys.height.max(1));
        self.targets = targets;
        self.arrows = arrows;
        self.game = game;

        let engine = Engine::new_with_window(
            scene,
            phys.width.max(1),
            phys.height.max(1),
            self.enable_validation,
            display_handle,
            window_handle,
        )
        .expect("failed to initialize Vulkan engine");

        self.window = Some(window);
        self.engine = Some(engine);
        self.last_frame = Instant::now();
        self.last_title_update = Instant::now();
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event
            && self.cursor_locked
            && !self.menu_open
        {
            let sens = 0.0022 * self.player.mouse_sens;
            self.player.yaw += (dx as f32) * sens;
            self.player.pitch = (self.player.pitch - (dy as f32) * sens).clamp(-1.42, 1.42);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(engine) = self.engine.as_mut()
                    && let Err(e) = engine.resize_swapchain(size.width, size.height)
                {
                    eprintln!("Swapchain resize error: {e}");
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                if self.menu_open {
                    if state == ElementState::Pressed {
                        self.adjust_menu(1);
                    }
                    return;
                }
                if !self.cursor_locked {
                    if state == ElementState::Pressed {
                        self.set_cursor_lock(true);
                    }
                    return;
                }
                match state {
                    ElementState::Pressed => {
                        self.drawing_bow = true;
                    }
                    ElementState::Released => {
                        if self.drawing_bow {
                            self.drawing_bow = false;
                            self.shoot_arrow();
                        }
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            self.pressed_keys.insert(code);
                            if !event.repeat {
                                match code {
                                    KeyCode::KeyM => {
                                        self.menu_open = !self.menu_open;
                                        self.drawing_bow = false;
                                        self.bow_charge = 0.0;
                                        self.set_cursor_lock(!self.menu_open);
                                    }
                                    KeyCode::Escape => {
                                        if self.menu_open {
                                            self.menu_open = false;
                                            self.set_cursor_lock(true);
                                        } else if self.cursor_locked {
                                            self.set_cursor_lock(false);
                                        } else {
                                            event_loop.exit();
                                        }
                                    }
                                    KeyCode::Tab => {
                                        if !self.menu_open {
                                            self.set_cursor_lock(!self.cursor_locked);
                                        }
                                    }
                                    KeyCode::KeyF if !self.menu_open => {
                                        self.drawing_bow = true;
                                    }
                                    KeyCode::ArrowUp | KeyCode::KeyW if self.menu_open => {
                                        self.selected_row = if self.selected_row <= 1 {
                                            7
                                        } else {
                                            self.selected_row - 1
                                        };
                                    }
                                    KeyCode::ArrowDown | KeyCode::KeyS if self.menu_open => {
                                        self.selected_row = if self.selected_row >= 7 {
                                            1
                                        } else {
                                            self.selected_row + 1
                                        };
                                    }
                                    KeyCode::ArrowLeft | KeyCode::KeyA if self.menu_open => {
                                        self.adjust_menu(-1);
                                    }
                                    KeyCode::ArrowRight
                                    | KeyCode::KeyD
                                    | KeyCode::Enter if self.menu_open => {
                                        self.adjust_menu(1);
                                    }
                                    _ => {}
                                }
                            }
                        }
                        ElementState::Released => {
                            self.pressed_keys.remove(&code);
                            if code == KeyCode::KeyF && self.drawing_bow {
                                self.drawing_bow = false;
                                self.shoot_arrow();
                            }
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let frame_dt = (now - self.last_frame).as_secs_f32().max(1e-5);
                self.last_frame = now;

                let fixed_dt = self.pacer.fixed_dt() as f32;
                let steps = self.pacer.begin_frame(frame_dt as f64);
                for _ in 0..steps {
                    self.step_simulation(fixed_dt);
                }

                self.sync_hud_state();

                if let Some(engine) = self.engine.as_mut() {
                    // Apply subtle bow draw FOV zoom without overwriting user's base FOV
                    let base_fov = engine.graphics_settings.fov_deg;
                    engine.graphics_settings.fov_deg = base_fov - self.bow_charge * 10.0;
                    let res = engine.render_frame_to_swapchain(fixed_dt);
                    engine.graphics_settings.fov_deg = base_fov;

                    match res {
                        Ok((stats, needs_recreate)) => {
                            if !stats.gpu_timings.is_empty() {
                                self.last_gpu_ms =
                                    stats.gpu_timings.iter().map(|t| t.gpu_ms).sum();
                            }
                            if needs_recreate && let Some(win) = self.window.as_ref() {
                                let sz = win.inner_size();
                                let _ = engine.resize_swapchain(sz.width, sz.height);
                            }
                        }
                        Err(e) => {
                            eprintln!("Render error: {e}");
                            event_loop.exit();
                        }
                    }
                }

                if self.last_title_update.elapsed().as_secs_f32() >= 0.25 {
                    self.last_title_update = Instant::now();
                    if let (Some(win), Some(engine)) = (self.window.as_ref(), self.engine.as_ref()) {
                        let st = self.pacer.stats();
                        win.set_title(&format!(
                            "Archery Range | Score: {} | {:.0} FPS | GPU {:.2}ms | {} | [Click] Aim [LMB/F] Shoot [M] Menu",
                            self.score,
                            st.avg_fps,
                            self.last_gpu_ms,
                            engine.denoiser_backend_name(),
                        ));
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(win) = self.window.as_ref() {
            win.request_redraw();
        }
    }
}

fn main() {
    let enable_validation = std::env::args().any(|a| a == "--validation");
    println!("Launching Vulkan 1.4 Archery Range...");
    println!("Controls: WASD walk | Space jump | Shift sprint | Click window to lock mouse | Hold LMB/F to draw bow & shoot | M = Graphics Menu");

    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = ArcheryApp::new(enable_validation);
    event_loop.run_app(&mut app).expect("event loop run");
}
