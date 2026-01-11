use std::sync::mpsc::Receiver;
use std::sync::{Arc, mpsc};
use std::thread;
use std::time::{Duration, Instant};
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};
mod vgfx;
use vgfx::WindowContext;
use vgfx::{init_vulkano, recreate_swapchain, redraw, resize_window};

use crate::game::RenderEvent;
use crate::vgfx::update_vertex_buffer;
mod game;
mod mesh;

#[derive(Default)]
struct App {
    window_contexts: Vec<WindowContext>,
    game_thread_receiver: Option<Receiver<RenderEvent>>,
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
                        for event in self.game_thread_receiver.as_ref().unwrap().try_iter() {
                            match event {
                                RenderEvent::AddMesh(mesh) => {
                                    window_context.add_mesh(mesh).unwrap();
                                    update_vertex_buffer(window_context);
                                },
                            }
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
    let (to_render, from_game) = mpsc::channel();
    thread::spawn(|| {
        game::game_main(to_render);
    });

    // Start initializing the window
    // The window is handled by the main thread, which listens and handles events from the OS like redraw request
    let event_loop = EventLoop::new()
        .unwrap_or_else(|err| panic!("Couldn't create window event loop: {:?}", err));
    event_loop.set_control_flow(ControlFlow::Poll);

    // Additional windows can be added by simply creating another window context and adding it to the app window array below
    let window_context = WindowContext::new(&event_loop);

    let mut app = App {
        window_contexts: vec![window_context],
        game_thread_receiver: Some(from_game),
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
