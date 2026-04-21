mod game;
mod mesh;
mod scene;
pub mod shader;
mod vgfx;

use std::sync::{Mutex, mpsc};
use std::time::{Duration, Instant};
use std::{env, thread, usize};
use vgfx::WindowContext;
use winit::platform::x11::EventLoopBuilderExtX11;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::WindowId,
};

use crate::game::{GameData, RenderEvent};
use crate::vgfx::{Platform};

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

        let window_context = WindowContext::new(&event_loop);
        self.window_contexts.push(window_context);

        for i in 0..self.window_contexts.len() {
            let (to_render, from_game) = mpsc::channel();
            let (to_game, from_render) = mpsc::channel();
            let game_data = GameData {
                to_render,
                from_render,
                render_queue: self.window_contexts[i].queues.clone()[0].clone(),
            };
            self.window_contexts[i].game_thread_receiver = Some(from_game);
            self.window_contexts[i].game_thread_sender = Some(to_game);
            thread::spawn(|| {
                game::game_main(game_data);
            });
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
                        match window_context
                            .game_thread_sender
                            .as_ref()
                            .unwrap()
                            .send(game::GameEvent::GameClose)
                        {
                            Ok(_) => (),
                            Err(err) => println!(
                                "Failed to send game close event to render thread, closing anyway... {:?}",
                                err
                            ),
                        };
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

                        // Get one event sent from game thread
                        let event = window_context
                            .game_thread_receiver
                            .as_ref()
                            .unwrap()
                            .try_recv();
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
                            },
                            Err(_) => (),
                        }
                        // Runs if verticies are changed or if shaders or perspective is updated
                        if should_update_buffers | should_update_taskgraph{
                            window_context.recreate_taskgraph();
                        }

                        // This logic is here so we don't end up regenerating pipelines every frame while resizing
                        // Unless we're on X11, which needs to be resized when requested
                        if window_context.requested_resize
                            && (window_context.last_resized.unwrap().elapsed()
                                > Duration::from_secs_f32(0.5)
                                || window_context.platform == Platform::X11)
                        {
                            window_context.last_resized = Some(Instant::now());
                            window_context.should_resize = true;
                            window_context.requested_resize = false;
                        }

                        if window_context.should_resize || window_context.recreate_swapchain {
                            window_context.recreate_swapchain = false;
                            window_context.recreate_swapchain();

                            if window_context.should_resize {
                                window_context.should_resize = false;
                                window_context.resize_window();
                            }
                        }

                        window_context.redraw();
                        window.request_redraw();
                    }
                    WindowEvent::Resized(_size) => {
                        // This logic is here so we don't end up regenerating pipelines every frame while resizing
                        window_context.last_resized =
                            Some(window_context.last_resized.unwrap_or(Instant::now()));
                        window_context.requested_resize = true;
                    }
                    _ => (), //println!("Event received: {:?}", event),
                }
            }
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
    // TODO: Find a better way to change whether we're using wayland or X11. Currently we're forcing wayland
    let event_loop = EventLoop::builder()
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
}
