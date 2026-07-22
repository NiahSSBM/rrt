mod game;
mod mesh;
mod object;
mod scene;
mod shader;
mod shader_cache;
mod vgfx;

use crate::game::{GameData, GameEvent, RenderEvent};
use std::sync::{Mutex, mpsc};
use std::time::{Duration, Instant};
use std::{env, thread, usize};
use vgfx::WindowContext;
use winit::event::DeviceEvent;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::window::CursorGrabMode;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::WindowId,
};

static FRAMES_SINCE_LAST_FRAMETIME_UPDATE: Mutex<i32> = Mutex::new(0);
static TIME_SINCE_LAST_FRAMETIME_UPDATE: Mutex<Option<Instant>> = Mutex::new(None);

#[derive(Default)]
struct App {
    window_contexts: Vec<WindowContext>,
    resume_count: u32,
    preferred_device: Option<String>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.resume_count += 1;
        if self.resume_count > 1 {
            println!(
                "Resume requested {} times, not recreating window and not resuming",
                self.resume_count
            );
            return;
        }

        for _ in 0..1 {
            let (to_render, from_game) = mpsc::channel();
            let (to_game, from_render) = mpsc::channel();
            let game_data = GameData::new(to_render, from_render);

            let handle = thread::spawn(|| {
                game::game_main(game_data);
            });

            let window_context = WindowContext::new(
                &event_loop,
                self.preferred_device.clone(),
                handle,
                (to_game, from_game),
            );

            self.window_contexts.push(window_context);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        for window_context in &mut self.window_contexts {
            let window = window_context.window.clone();
            if window_id == window.id() {
                match event {
                    WindowEvent::CloseRequested => {
                        println!("The close button was pressed; stopping");
                        if let Err(e) = window_context
                            .game_thread_sender
                            .send(game::GameEvent::GameClose)
                        {
                            println!(
                                "Failed to send game close event to render thread, closing anyway... {:?}",
                                e
                            );
                        }
                        event_loop.exit();
                    }
                    WindowEvent::RedrawRequested => {
                        // Calculate framerate
                        let fs_lock = FRAMES_SINCE_LAST_FRAMETIME_UPDATE.try_lock();
                        match fs_lock {
                            Ok(mut fs_mg) => {
                                let frametime_update_lock =
                                    TIME_SINCE_LAST_FRAMETIME_UPDATE.try_lock();
                                match frametime_update_lock {
                                    Ok(mut frametime_update_mg) => {
                                        if frametime_update_mg.is_none() {
                                            *frametime_update_mg = Some(Instant::now());
                                        }
                                        if frametime_update_mg.unwrap().elapsed()
                                            > Duration::from_secs(1)
                                        {
                                            println!("{} FPS", *fs_mg);
                                            *fs_mg = 0;
                                            *frametime_update_mg = Some(Instant::now());
                                        }
                                    }
                                    Err(_) => (),
                                }
                                *fs_mg += 1;
                            }
                            Err(_) => (),
                        }

                        let mut should_update_buffers = false;
                        let mut should_update_taskgraph = false;
                        let mut should_update_shaders = false;

                        // Get one event sent from game thread
                        let event = window_context.game_thread_receiver.try_recv();
                        match event {
                            Ok(e) => match e {
                                RenderEvent::AddMesh(mesh) => {
                                    window_context.add_mesh(mesh).unwrap();
                                }
                                RenderEvent::UpdateVertexBuffer => {
                                    should_update_buffers = true;
                                }
                                RenderEvent::UpdateTaskGraph => {
                                    should_update_taskgraph = true;
                                }
                                RenderEvent::UpdateShader => {
                                    should_update_shaders = true;
                                }
                            },
                            Err(_) => (),
                        }

                        if window_context.requested_resize || window_context.recreate_swapchain {
                            window_context.recreate_swapchain = false;
                            window_context.recreate_swapchain();

                            if window_context.requested_resize {
                                window_context.requested_resize = false;
                                window_context.resize_window();
                                let _ = window_context.game_thread_sender.send(
                                    GameEvent::WindowResized((
                                        window_context.viewport.extent[0],
                                        window_context.viewport.extent[1],
                                    )),
                                );
                            }
                        }

                        // Runs if vertices are changed or if shaders or perspective is updated
                        if should_update_buffers {
                            window_context.recreate_buffers();
                        }
                        if should_update_taskgraph {
                            window_context.recreate_taskgraph();
                        }
                        if should_update_shaders {
                            window_context.reload_shaders();
                        }

                        window_context.redraw();
                        window.request_redraw();
                    }
                    WindowEvent::Resized(_size) => {
                        window_context.requested_resize = true;
                    }
                    WindowEvent::Focused(is_focused) => {
                        if is_focused {
                            window.set_cursor_visible(false);
                            let _ = window
                                .set_cursor_grab(CursorGrabMode::Confined)
                                .inspect_err(|e| println!("Could not capture cursor: {e}"));
                        } else {
                            window.set_cursor_visible(true);
                            let _ = window
                                .set_cursor_grab(winit::window::CursorGrabMode::None)
                                .inspect_err(|e| println!("Could not release cursor: {e}"));
                        }
                    }
                    WindowEvent::KeyboardInput {
                        device_id: _device_id,
                        ref event,
                        is_synthetic: _is_synthetic,
                    } => {
                        if event.physical_key == PhysicalKey::Code(KeyCode::Escape) {
                            window.set_cursor_visible(true);
                            let _ = window
                                .set_cursor_grab(winit::window::CursorGrabMode::None)
                                .inspect_err(|e| println!("Could not release cursor: {e}"));
                        }
                        let _ = window_context
                            .game_thread_sender
                            .send(GameEvent::KeyEvent(event.clone()))
                            .inspect_err(|e| {
                                println!(
                                    "Failed to send cursor moved event to game thread: {:?}",
                                    e
                                )
                            });
                    }
                    WindowEvent::CursorMoved {
                        device_id: _device_id,
                        position,
                    } => {
                        let _cursor_delta = [
                            window_context.last_cursor_position.x - position.x,
                            window_context.last_cursor_position.y - position.y,
                        ];

                        window_context.last_cursor_position = position;
                    }
                    _ => (), //println!("Window event received: {:?}", event),
                }
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } => {
                for window_context in &self.window_contexts {
                    let _ = window_context
                        .game_thread_sender
                        .send(GameEvent::CursorMoved(delta))
                        .inspect_err(|e| {
                            println!("Failed to send cursor moved event to game thread: {:?}", e)
                        });
                }
            }
            _ => (), //println!("Device event received: {:?}", event),
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let preferred_device_position: Option<usize> = args
        .iter()
        .position(|f| f.to_ascii_lowercase() == "--device");
    let mut preferred_device: Option<String> = None;
    if preferred_device_position.is_some() {
        preferred_device = args.get(preferred_device_position.unwrap() + 1).cloned();
    }

    // Start initializing the window
    // The window is handled by the main thread, which listens and handles events from the OS like redraw request
    // TODO: Find a better way to change whether we're using wayland or X11. Currently we just force either X11 or Wayland
    let event_loop = EventLoop::builder()
        .with_x11()
        .build()
        .unwrap_or_else(|err| panic!("Couldn't create window event loop: {:?}", err));
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App {
        window_contexts: vec![],
        preferred_device,
        ..Default::default()
    };

    // Execution is blocked here until the event loop is exited when the user closes the window
    event_loop.run_app(&mut app).unwrap_or_else(|err| {
        panic!(
            "Event loop couldn't be created or exited with and error: {:?}",
            err
        )
    });

    for window_context in &mut app.window_contexts {
        if let Some(handle) = window_context.game_thread_handle.take() {
            if let Err(e) = handle.join() {
                println!("Game thread exited with error: {:?}", e);
            }
        }
    }
}
