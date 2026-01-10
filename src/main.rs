use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::window;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};
mod vgfx;
use vgfx::WindowContext;
use vgfx::{init_vulkano, recreate_swapchain, redraw, resize_window};

use crate::mesh::{Mesh, MyVertex};
use crate::vgfx::Shaders;
mod mesh;

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

            let mesh = Mesh::new(
                vec![
                    MyVertex::new([0.5, 0.5]),
                    MyVertex::new([0.0, -0.5]),
                    MyVertex::new([-0.5, 0.5]),
                ],
                Shaders {
                    vs: self.window_contexts[i].default_vs.clone().unwrap(),
                    fs: self.window_contexts[i].default_fs.clone().unwrap(),
                    descriptor_set: None,
                },
            );
            self.window_contexts[i].add_mesh(mesh).unwrap();
        }
        // This locks up the thread
        //self.window.first().unwrap().pre_present_notify();
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
    let event_loop = EventLoop::new()
        .unwrap_or_else(|err| panic!("Couldn't create window event loop: {:?}", err));
    event_loop.set_control_flow(ControlFlow::Poll);

    let window_context = WindowContext::new(&event_loop);

    let mut app = App {
        window_contexts: vec![window_context],
        ..Default::default()
    };
    event_loop.run_app(&mut app).unwrap_or_else(|err| {
        panic!(
            "Event loop couldn't be created or exited with and error: {:?}",
            err
        )
    });
}
