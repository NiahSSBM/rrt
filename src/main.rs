mod game;
mod mesh;
pub mod shader;
mod vgfx;

use std::f64::consts::E;
use std::sync::mpsc::Receiver;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use vgfx::WindowContext;
use vgfx::{init_vulkano, recreate_swapchain, redraw, resize_window};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

use crate::game::{GameData, RenderEvent};
use crate::vgfx::update_vertex_buffer;

#[derive(Default)]
struct App {
    window_contexts: Vec<WindowContext>,
    resume_count: u32,
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
        for i in 0..self.window_contexts.len() {
            let window = Arc::new(
                event_loop
                    .create_window(Window::default_attributes())
                    .unwrap_or_else(|err| panic!("Could not create window: {:?}", err)),
            );
            self.window_contexts[i].window = Some(window);
            init_vulkano(&mut self.window_contexts[i]);

            let (to_render, from_game) = mpsc::channel();
            let game_data = GameData {
                to_render,
                render_device: self.window_contexts[i].device.clone().unwrap(),
            };
            self.window_contexts[i].game_thread_receiver = Some(from_game);
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
            let window = window_context.window.clone().unwrap();
            if window_id == window.id() {
                match event {
                    WindowEvent::CloseRequested => {
                        println!("The close button was pressed; stopping");
                        event_loop.exit();
                    }
                    WindowEvent::RedrawRequested => {
                        let mut should_update_buffers = false;

                        // Get one event sent from game thread
                        let event = window_context
                            .game_thread_receiver
                            .as_ref()
                            .unwrap()
                            .try_recv();
                        match event {
                            Ok(e) => match e {
                                // Right now the only thing the game thread sending over is meshes to add
                                RenderEvent::AddMesh(mesh) => {
                                    println!("Adding Mesh");
                                    window_context.add_mesh(mesh).unwrap();
                                    should_update_buffers = true;
                                }
                            },
                            Err(_) => (),
                        }
                        // This only triggers if a mesh was added
                        if should_update_buffers {
                            update_vertex_buffer(window_context);
                        }

                        if window_context.resized || window_context.recreate_swapchain {
                            window_context.recreate_swapchain = false;
                            recreate_swapchain(window_context);

                            if window_context.resized {
                                window_context.resized = false;
                                window_context.last_resized = Some(Instant::now());
                                resize_window(window_context);
                            }
                        }

                        redraw(window_context);
                        window.request_redraw();
                    }
                    WindowEvent::Resized(_size) => {
                        let last_resized = window_context
                            .last_resized
                            .unwrap_or(Instant::now() - Duration::from_secs(5));
                        if last_resized.elapsed() > Duration::from_secs_f32(1.0) {
                            window_context.resized = true;
                        }
                    }
                    _ => (), //println!("Event received: {:?}", event),
                }
            }
        }
    }
}

fn main() {
    // Start initializing the window
    // The window is handled by the main thread, which listens and handles events from the OS like redraw request
    let event_loop = EventLoop::new()
        .unwrap_or_else(|err| panic!("Couldn't create window event loop: {:?}", err));
    event_loop.set_control_flow(ControlFlow::Poll);

    // Additional windows can be added by simply creating another window context and adding it to the app window array below
    let window_context = WindowContext::new(&event_loop);

    let mut app = App {
        window_contexts: vec![window_context],
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
