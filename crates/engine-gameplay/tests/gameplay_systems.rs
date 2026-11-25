use engine_gameplay::{
    AudioEngine, AudioSource, Collider, GameWorld, MeshRenderer, Name, PhysicsConfig,
    PointLightComponent, RigidBody, SceneEditor, ScriptComponent, ScriptEngine, Transform,
    step_physics,
};

#[test]
fn ecs_spawning_and_transform_matrix() {
    let mut game = GameWorld::new();
    let e = game.spawn((
        Name::new("HeroCube"),
        Transform::from_position_scale([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]),
        MeshRenderer {
            mesh_id: 0,
            material_id: 1,
            instance_index: Some(0),
        },
        PointLightComponent {
            color: [1.0, 0.9, 0.8],
            energy: 5.0,
            range: 12.0,
        },
        AudioSource {
            frequency_hz: 440.0,
            triggered: false,
        },
    ));
    assert_eq!(game.entity_count(), 1);
    let t = game.world.get::<&Transform>(e).expect("transform");
    let m = t.to_matrix();
    assert!((m[0][0] - 2.0).abs() < 1e-5);
    assert!((m[1][1] - 2.0).abs() < 1e-5);
    assert!((m[2][2] - 2.0).abs() < 1e-5);
    assert!((m[3][0] - 1.0).abs() < 1e-5);
    assert!((m[3][1] - 2.0).abs() < 1e-5);
    assert!((m[3][2] - 3.0).abs() < 1e-5);
}

#[test]
fn physics_gravity_ground_bounce_and_sphere_collision() {
    let mut game = GameWorld::new();
    // Ground plane at y = 0.0
    game.spawn((Name::new("Ground"), Collider::GroundPlane { y: 0.0 }));

    // Falling sphere starting at y = 3.0, radius = 0.5, restitution = 0.8
    let ball = game.spawn((
        Name::new("Ball"),
        Transform::from_position([0.0, 3.0, 0.0]),
        RigidBody::dynamic(1.0, 0.8),
        Collider::Sphere { radius: 0.5 },
    ));

    let cfg = PhysicsConfig::default();
    let dt = 1.0 / 60.0;
    let mut hit_ground = false;
    let mut bounced_up = false;

    for _ in 0..120 {
        step_physics(&mut game.world, &cfg, dt);
        let t = game.world.get::<&Transform>(ball).unwrap();
        let rb = game.world.get::<&RigidBody>(ball).unwrap();
        assert!(
            t.position[1] >= 0.5 - 1e-4,
            "Sphere penetrated ground plane: y={}",
            t.position[1]
        );
        if (t.position[1] - 0.5).abs() < 1e-3 {
            hit_ground = true;
        }
        if hit_ground && rb.velocity[1] > 0.5 {
            bounced_up = true;
            break;
        }
    }
    assert!(hit_ground, "Ball never reached the ground plane");
    assert!(bounced_up, "Ball did not bounce upward after hitting ground");

    // Sphere-vs-sphere head-on elastic collision
    let mut sphere_world = GameWorld::new();
    let mut rb_a = RigidBody::dynamic(1.0, 1.0);
    rb_a.use_gravity = false;
    rb_a.velocity = [2.0, 0.0, 0.0];
    let mut rb_b = RigidBody::dynamic(1.0, 1.0);
    rb_b.use_gravity = false;
    rb_b.velocity = [-2.0, 0.0, 0.0];

    let a = sphere_world.spawn((
        Transform::from_position([-0.6, 0.0, 0.0]),
        rb_a,
        Collider::Sphere { radius: 0.5 },
    ));
    let b = sphere_world.spawn((
        Transform::from_position([0.6, 0.0, 0.0]),
        rb_b,
        Collider::Sphere { radius: 0.5 },
    ));

    let zero_damp = PhysicsConfig {
        gravity: [0.0, 0.0, 0.0],
        linear_damping: 0.0,
    };
    for _ in 0..20 {
        step_physics(&mut sphere_world.world, &zero_damp, 1.0 / 60.0);
    }
    let va = sphere_world.world.get::<&RigidBody>(a).unwrap().velocity[0];
    let vb = sphere_world.world.get::<&RigidBody>(b).unwrap().velocity[0];
    assert!(va < 0.0, "Expected sphere A to rebound left, got vx={va}");
    assert!(vb > 0.0, "Expected sphere B to rebound right, got vx={vb}");
}

#[test]
fn rhai_scripting_animates_transform() {
    let mut game = GameWorld::new();
    let entity = game.spawn((
        Name::new("Orbiter"),
        Transform::from_position([0.0, 1.0, 0.0]),
        ScriptComponent::new("x = time * 2.0; y = 1.0 + dt * 10.0; ry = ry + dt;"),
    ));

    let mut scripts = ScriptEngine::new();
    scripts.step_scripts(&mut game.world, 0.05, 1.5);

    let t = game.world.get::<&Transform>(entity).unwrap();
    assert!((t.position[0] - 3.0).abs() < 1e-4);
    assert!((t.position[1] - 1.5).abs() < 1e-4);
    assert!((t.rotation_euler[1] - 0.05).abs() < 1e-4);
}

#[test]
fn audio_engine_headless_safe() {
    // Must never panic even when no audio device exists in CI / headless environments
    let audio = AudioEngine::new();
    audio.play_beep(440.0, 10);
    audio.play_wav_bytes(&[]);
}

#[test]
fn scene_editor_selection_and_nudge() {
    let mut game = GameWorld::new();
    game.spawn((Name::new("First"), Transform::from_position([0.0, 0.0, 0.0])));
    game.spawn((Name::new("Second"), Transform::from_position([5.0, 0.0, 0.0])));

    let mut editor = SceneEditor::new();
    assert!(!editor.paused);
    editor.toggle_pause();
    assert!(editor.paused);

    let s0 = editor.selected_summary(&game.world);
    assert!(s0.contains("First"), "Expected First, got {s0}");

    assert!(editor.nudge_selected(&mut game.world, [1.5, 0.0, -0.5]));
    let s0_after = editor.selected_summary(&game.world);
    assert!(s0_after.contains("1.50"), "Expected nudged position, got {s0_after}");

    editor.select_next(&game.world);
    let s1 = editor.selected_summary(&game.world);
    assert!(s1.contains("Second"), "Expected Second, got {s1}");
}
