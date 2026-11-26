//! Interactive windowed Vulkan 1.3/1.4 viewer + ECS/physics/scripting/editor sandbox.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use engine_assets::load_gltf_from_path;
use engine_core::{Engine, FramePacer, SceneWorld, create_demo_scene};
use engine_gameplay::{
    AudioEngine, Collider, GameWorld, MeshRenderer, Name, PhysicsConfig, PhysicsWorld, RigidBody,
    SceneEditor, ScriptComponent, ScriptEngine, Transform,
};
use engine_rhi::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const DEFAULT_WIDTH: u32 = 1280;
const DEFAULT_HEIGHT: u32 = 720;

struct FreeFlyCamera {
    position: [f32; 3],
    yaw: f32,
    pitch: f32,
    move_speed: f32,
    look_speed: f32,
}

impl FreeFlyCamera {
    fn from_scene(scene: &SceneWorld) -> Self {
        let pos = scene.camera.position;
        let tgt = scene.camera.target;
        let dx = tgt[0] - pos[0];
        let dy = tgt[1] - pos[1];
        let dz = tgt[2] - pos[2];
        let horiz = (dx * dx + dz * dz).sqrt().max(1e-4);
        Self {
            position: pos,
            yaw: dz.atan2(dx),
            pitch: dy.atan2(horiz).clamp(-1.45, 1.45),
            move_speed: 4.5,
            look_speed: 1.8,
        }
    }

    fn forward(&self) -> [f32; 3] {
        let cp = self.pitch.cos();
        [self.yaw.cos() * cp, self.pitch.sin(), self.yaw.sin() * cp]
    }

    fn right(&self) -> [f32; 3] {
        let f = self.forward();
        let rx = -f[2];
        let rz = f[0];
        let len = (rx * rx + rz * rz).sqrt().max(1e-4);
        [rx / len, 0.0, rz / len]
    }

    fn update(&mut self, keys: &HashSet<KeyCode>, dt: f32) {
        if keys.contains(&KeyCode::ArrowLeft) {
            self.yaw -= self.look_speed * dt;
        }
        if keys.contains(&KeyCode::ArrowRight) {
            self.yaw += self.look_speed * dt;
        }
        if keys.contains(&KeyCode::ArrowUp) {
            self.pitch = (self.pitch + self.look_speed * dt).clamp(-1.45, 1.45);
        }
        if keys.contains(&KeyCode::ArrowDown) {
            self.pitch = (self.pitch - self.look_speed * dt).clamp(-1.45, 1.45);
        }

        let fwd = self.forward();
        let right = self.right();
        let speed = self.move_speed * dt;

        if keys.contains(&KeyCode::KeyW) {
            for (pos, &f) in self.position.iter_mut().zip(&fwd) {
                *pos += f * speed;
            }
        }
        if keys.contains(&KeyCode::KeyS) {
            for (pos, &f) in self.position.iter_mut().zip(&fwd) {
                *pos -= f * speed;
            }
        }
        if keys.contains(&KeyCode::KeyD) {
            for (pos, &r) in self.position.iter_mut().zip(&right) {
                *pos += r * speed;
            }
        }
        if keys.contains(&KeyCode::KeyA) {
            for (pos, &r) in self.position.iter_mut().zip(&right) {
                *pos -= r * speed;
            }
        }
        if keys.contains(&KeyCode::Space) {
            self.position[1] += speed;
        }
        if keys.contains(&KeyCode::ShiftLeft) || keys.contains(&KeyCode::ShiftRight) {
            self.position[1] -= speed;
        }
    }

    fn apply_to_scene(&self, scene: &mut SceneWorld) {
        let fwd = self.forward();
        scene.camera.position = self.position;
        scene.camera.target = [
            self.position[0] + fwd[0],
            self.position[1] + fwd[1],
            self.position[2] + fwd[2],
        ];
    }
}

struct ViewerApp {
    gltf_path: Option<PathBuf>,
    enable_validation: bool,
    window: Option<Arc<Window>>,
    engine: Option<Engine>,
    game: GameWorld,
    physics: PhysicsWorld,
    physics_cfg: PhysicsConfig,
    scripts: ScriptEngine,
    audio: AudioEngine,
    editor: SceneEditor,
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    pacer: FramePacer,
    camera: FreeFlyCamera,
    pressed_keys: HashSet<KeyCode>,
    mouse_dragging: bool,
    last_cursor: Option<(f64, f64)>,
    last_frame: Instant,
    last_title_update: Instant,
    sim_time: f32,
    last_gpu_ms: f64,
}

impl ViewerApp {
    fn new(gltf_path: Option<PathBuf>, enable_validation: bool) -> Self {
        let now = Instant::now();
        Self {
            gltf_path,
            enable_validation,
            window: None,
            engine: None,
            game: GameWorld::new(),
            physics: PhysicsWorld::new(),
            physics_cfg: PhysicsConfig::default(),
            scripts: ScriptEngine::new(),
            audio: AudioEngine::new(),
            editor: SceneEditor::new(),
            egui_ctx: egui::Context::default(),
            egui_state: None,
            pacer: FramePacer::default(),
            camera: FreeFlyCamera {
                position: [0.0, 1.8, 5.5],
                yaw: -std::f32::consts::FRAC_PI_2,
                pitch: -0.25,
                move_speed: 4.5,
                look_speed: 1.8,
            },
            pressed_keys: HashSet::new(),
            mouse_dragging: false,
            last_cursor: None,
            last_frame: now,
            last_title_update: now,
            sim_time: 0.0,
            last_gpu_ms: 0.0,
        }
    }

    fn populate_demo_gameplay(&mut self, scene: &mut SceneWorld) {
        // Disable built-in SceneWorld sine animation on instances driven by ECS
        for inst in &mut scene.instances {
            inst.animated = false;
        }

        // Ground plane at y = -1.0 (matches floor instance 0)
        self.game
            .spawn((Name::new("GroundPlane"), Collider::GroundPlane { y: -1.0 }));

        // Bouncing gold sphere (instance 4)
        if scene.instances.len() > 4 {
            self.game.spawn((
                Name::new("GoldSphere"),
                Transform::from_position_scale([1.4, 2.2, 0.2], [1.4, 1.4, 1.4]),
                MeshRenderer {
                    mesh_id: 2,
                    material_id: 2,
                    instance_index: Some(4),
                },
                RigidBody::dynamic(1.0, 0.88),
                Collider::Sphere { radius: 0.7 },
            ));
        }

        // Bouncing chrome sphere (instance 5)
        if scene.instances.len() > 5 {
            self.game.spawn((
                Name::new("ChromeSphere"),
                Transform::from_position_scale([0.2, 1.4, 1.2], [1.1, 1.1, 1.1]),
                MeshRenderer {
                    mesh_id: 2,
                    material_id: 3,
                    instance_index: Some(5),
                },
                RigidBody::dynamic(1.0, 0.82),
                Collider::Sphere { radius: 0.55 },
            ));
        }

        // Rhai-scripted orbiting emissive beacon (instance 7)
        if scene.instances.len() > 7 {
            self.game.spawn((
                Name::new("EmissiveBeacon"),
                Transform::from_position_scale([-2.6, 0.4, 1.2], [0.7, 0.7, 0.7]),
                MeshRenderer {
                    mesh_id: 0,
                    material_id: 5,
                    instance_index: Some(7),
                },
                ScriptComponent::new(
                    "x = -2.0 + 1.2 * cos(time * 1.2); y = 0.5 + 0.35 * sin(time * 2.1); z = 1.0 + 1.2 * sin(time * 1.2); ry = time;",
                ),
            ));
        }

        // Selectable terracotta cube (instance 6)
        if scene.instances.len() > 6 {
            self.game.spawn((
                Name::new("TerracottaCube"),
                Transform::from_position_scale([2.2, -0.35, -1.2], [1.3, 1.3, 1.3]),
                MeshRenderer {
                    mesh_id: 0,
                    material_id: 4,
                    instance_index: Some(6),
                },
            ));
        }
    }

    fn populate_gltf_gameplay(&mut self, scene: &SceneWorld) {
        for (idx, inst) in scene.instances.iter().enumerate() {
            let pos = [
                inst.transform[3][0],
                inst.transform[3][1],
                inst.transform[3][2],
            ];
            self.game.spawn((
                Name::new(format!("GltfInstance#{idx}")),
                Transform::from_position_scale(pos, inst.base_scale),
                MeshRenderer {
                    mesh_id: inst.mesh_id,
                    material_id: inst.material_id,
                    instance_index: Some(idx),
                },
            ));
        }
    }
}

impl ApplicationHandler for ViewerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("vulkan-gpu-driven-engine viewer")
            .with_inner_size(LogicalSize::new(DEFAULT_WIDTH, DEFAULT_HEIGHT));

        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("Failed to create viewer window"),
        );
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let mut scene = if let Some(path) = self.gltf_path.as_ref() {
            let loaded = load_gltf_from_path(path).expect("Failed to load glTF/GLB asset");
            let s = SceneWorld::from_loaded_scene(loaded, width, height);
            self.populate_gltf_gameplay(&s);
            s
        } else {
            let mut s = create_demo_scene(width, height);
            self.populate_demo_gameplay(&mut s);
            s
        };

        self.camera = FreeFlyCamera::from_scene(&scene);
        scene.sync_from_game_world(&self.game);

        let display_handle = window.display_handle().expect("display_handle").as_raw();
        let window_handle = window.window_handle().expect("window_handle").as_raw();

        let engine = Engine::new_with_window(
            scene,
            width,
            height,
            self.enable_validation,
            display_handle,
            window_handle,
        )
        .expect("Engine::new_with_window");

        log::info!(
            "Viewer initialized on {} | Denoiser: {} | CoopML: {:?}",
            engine.device_name(),
            engine.denoiser_backend_name(),
            engine.coop_ml_tile()
        );

        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        self.egui_state = Some(egui_state);
        self.window = Some(window);
        self.engine = Some(engine);
        self.last_frame = Instant::now();
        self.last_title_update = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if let (Some(win), Some(egui_state)) = (self.window.as_ref(), self.egui_state.as_mut()) {
            let resp = egui_state.on_window_event(win, &event);
            if resp.consumed {
                return;
            }
        }
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                if let Some(engine) = self.engine.as_mut()
                    && let Err(err) = engine.resize_swapchain(new_size.width, new_size.height)
                {
                    log::error!("Swapchain resize failed: {err}");
                }
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_dragging = state == ElementState::Pressed;
                if !self.mouse_dragging {
                    self.last_cursor = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.mouse_dragging {
                    if let Some((lx, ly)) = self.last_cursor {
                        let dx = (position.x - lx) as f32;
                        let dy = (position.y - ly) as f32;
                        self.camera.yaw += dx * 0.005;
                        self.camera.pitch = (self.camera.pitch - dy * 0.005).clamp(-1.45, 1.45);
                    }
                    self.last_cursor = Some((position.x, position.y));
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            let fresh = self.pressed_keys.insert(code);
                            if fresh {
                                match code {
                                    KeyCode::Escape => event_loop.exit(),
                                    KeyCode::F1 => {
                                        self.editor.show_editor_ui = !self.editor.show_editor_ui;
                                    }
                                    KeyCode::KeyP => {
                                        self.editor.toggle_pause();
                                    }
                                    KeyCode::Tab => {
                                        self.editor.select_next(&self.game.world);
                                        self.audio.play_beep(520.0, 25);
                                    }
                                    KeyCode::KeyI => {
                                        self.editor.nudge_selected(
                                            &mut self.game.world,
                                            [0.0, 0.0, -0.25],
                                        );
                                    }
                                    KeyCode::KeyK => {
                                        self.editor
                                            .nudge_selected(&mut self.game.world, [0.0, 0.0, 0.25]);
                                    }
                                    KeyCode::KeyJ => {
                                        self.editor.nudge_selected(
                                            &mut self.game.world,
                                            [-0.25, 0.0, 0.0],
                                        );
                                    }
                                    KeyCode::KeyL => {
                                        self.editor
                                            .nudge_selected(&mut self.game.world, [0.25, 0.0, 0.0]);
                                    }
                                    _ => {}
                                }
                            }
                        }
                        ElementState::Released => {
                            self.pressed_keys.remove(&code);
                        }
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let raw_dt = (now - self.last_frame).as_secs_f64();
                self.last_frame = now;

                let fixed_steps = self.pacer.begin_frame(raw_dt);
                let dt_f32 = (raw_dt as f32).clamp(1e-4, 0.1);

                self.camera.update(&self.pressed_keys, dt_f32);

                if !self.editor.paused {
                    let fixed_dt = self.pacer.fixed_dt() as f32;
                    for _ in 0..fixed_steps {
                        self.sim_time += fixed_dt;
                        self.physics
                            .step(&mut self.game.world, &self.physics_cfg, fixed_dt);
                        self.scripts
                            .step_scripts(&mut self.game.world, fixed_dt, self.sim_time);
                    }
                }

                let egui_output = if let (Some(win), Some(egui_state)) =
                    (self.window.as_ref(), self.egui_state.as_mut())
                {
                    let raw_input = egui_state.take_egui_input(win);
                    let full_output = self.egui_ctx.run(raw_input, |ctx| {
                        self.editor.draw_ui(ctx, &mut self.game.world);
                    });
                    egui_state.handle_platform_output(win, full_output.platform_output.clone());
                    Some(full_output)
                } else {
                    None
                };

                if let Some(engine) = self.engine.as_mut() {
                    self.camera.apply_to_scene(&mut engine.world);
                    engine.world.sync_from_game_world(&self.game);

                    let ui_arg = egui_output.map(|out| (&self.egui_ctx, out));
                    match engine.render_frame_to_swapchain_with_egui(dt_f32, ui_arg) {
                        Ok((stats, needs_recreate)) => {
                            if !stats.gpu_timings.is_empty() {
                                self.last_gpu_ms = stats.gpu_timings.iter().map(|t| t.gpu_ms).sum();
                            }
                            if needs_recreate && let Some(win) = self.window.as_ref() {
                                let sz = win.inner_size();
                                let _ = engine.resize_swapchain(sz.width, sz.height);
                            }
                        }
                        Err(err) => {
                            log::error!("render_frame_to_swapchain failed: {err}");
                        }
                    }

                    if self.last_title_update.elapsed().as_millis() >= 250
                        && let Some(win) = self.window.as_ref()
                    {
                        self.last_title_update = Instant::now();
                        let pstats = self.pacer.stats();
                        let selected = self.editor.selected_summary(&self.game.world);
                        let pause_tag = if self.editor.paused { " [PAUSED]" } else { "" };
                        win.set_title(&format!(
                            "Vulkan Engine{pause_tag} | {:.1} FPS (1% low {:.1}) | GPU {:.2} ms | {} | CoopML {:?} | Sel: {}",
                            pstats.avg_fps,
                            pstats.one_percent_low_fps,
                            self.last_gpu_ms,
                            engine.denoiser_backend_name(),
                            engine.coop_ml_tile(),
                            selected
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
    let mut args = std::env::args().skip(1);
    let mut gltf_path: Option<PathBuf> = None;
    let mut enable_validation = cfg!(debug_assertions);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--gltf" => {
                if let Some(p) = args.next() {
                    gltf_path = Some(PathBuf::from(p));
                }
            }
            "--validate" => {
                enable_validation = true;
            }
            "--no-validate" => {
                enable_validation = false;
            }
            _ => {}
        }
    }

    let event_loop = EventLoop::new().expect("Failed to create winit EventLoop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = ViewerApp::new(gltf_path, enable_validation);
    event_loop.run_app(&mut app).expect("EventLoop error");
}
