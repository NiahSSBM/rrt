use crate::mesh::{Mesh3D, Triangle};
use crate::shader::{Shader, Vertex3D};
use bitstream_io::{ByteRead, ByteReader, LittleEndian};
use color::palette::css;
use gltf::buffer::View;
use gltf::image::Source;
use gltf::json::accessor::{ComponentType, Type};
use image::buffer::ConvertBuffer;
use image::codecs::png::PngDecoder;
use image::{ColorType, ImageBuffer, ImageDecoder, Luma, LumaA, Rgb, Rgba};
use nalgebra::{Rotation3, Transform3, Vector3, max};
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub struct Object {
    pub path: PathBuf,
    pub is_loaded: bool,
    pub mesh: Option<Arc<Mutex<Mesh3D>>>,
    pub shader: Option<Shader>,
    pub transform: Transform3<f32>,
    pub children: Vec<Object>,
}

impl Object {
    pub fn new() -> Self {
        Self {
            path: PathBuf::new(),
            is_loaded: false,
            mesh: None,
            shader: None,
            transform: Transform3::identity(),
            children: Vec::new(),
        }
    }

    pub fn from_path(path: &Path, shader: Shader) -> Self {
        Self {
            path: path.to_path_buf(),
            is_loaded: false,
            mesh: None,
            shader: Some(shader),
            transform: Transform3::identity(),
            children: Vec::new(),
        }
    }

    pub fn from_mesh(mesh: Mesh3D) -> Self {
        Self {
            path: PathBuf::new(),
            is_loaded: true,
            shader: Some(mesh.shader.clone()),
            mesh: Some(Arc::new(Mutex::new((mesh)))),
            transform: Transform3::identity(),
            children: Vec::new(),
        }
    }

    pub fn translate(&mut self, vector: Vector3<f32>) {
        self.transform = Transform3::from_matrix_unchecked(
            self.transform
                .to_homogeneous()
                .prepend_translation(&vector)
                .into(),
        );
    }

    pub fn rotate(&mut self, rotation: Rotation3<f32>) {
        self.transform *= rotation;
    }

    /// If a path is set, and the file is a supported type (GLB or GLTF), the file is loaded into this objects children.
    /// Does nothing if no path was set, the file does not exist, or no shader was set.
    pub fn load(&mut self) {
        if self.path.to_str().unwrap_or("").is_empty() {
            return;
        }
        if !self.path.exists() {
            println!("{} does not exist!", self.path.display());
            return;
        }
        if self.path.extension() == Some(OsStr::new("gltf"))
            || self.path.extension() == Some(OsStr::new("glb"))
        {
            if self.shader.is_none() {
                println!("Tried to load gltf without a shader!");
                return;
            }
            self.children = load_gltf(self.path.clone(), &mut self.shader.clone().unwrap())
        }
    }
}

/// Loads a GLTF or GLB file from a path and converts it to a Vector of compatible rrt::Objects.
/// Also prints a tree of what is loaded.
fn load_gltf(path: PathBuf, shader: &mut Shader) -> Vec<Object> {
    let mut out: Vec<Object> = Vec::new();
    println!("Loading GLTF {}", path.display());

    let (document, buffers, _images) = match gltf::import(path) {
        Ok(g) => g,
        Err(e) => {
            println!("Error opening GLTF file: {e}");
            return vec![];
        }
    };

    println!("Scenes loaded: {}", document.scenes().count());

    println!("Materials loaded: {}", document.materials().count());
    for material in document.materials() {
        print_material(&material, 1);
    }

    println!("Textures loaded: {}", document.textures().count());
    for texture in document.textures() {
        print_texture(&texture, 1);
        match texture.source().source() {
            Source::View { view, mime_type } => {
                shader.set_texture(
                    gltf_load_image_from_view(&view, buffers.get(0).unwrap()).unwrap(),
                );
            }
            Source::Uri { uri, mime_type } => {}
        }
    }

    for scene in document.scenes() {
        println!(
            "Loaded Scene {}",
            scene.name().unwrap_or("[unnammed scene]")
        );
        println!("Scene has {} nodes", scene.nodes().len());
        for node in scene.nodes() {
            walk_gltf_nodes(&node, 0);
        }
    }

    let models = gltf_load_model(document, buffers);
    for (_, (vertex_components, indices, normal_components, texcoord_components)) in models {
        let normals = normal_components.as_chunks::<3>().0;
        let vertices = vertex_components
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
            .map(|(i, vertex)| {
                Vertex3D::new(
                    *vertex,
                    *normals.get(i).unwrap_or(&[0.0, 0.0, 0.0]),
                    css::WHITE,
                )
            })
            .collect();
        let triangles = indices
            .as_chunks::<3>()
            .0
            .iter()
            .enumerate()
            .map(|(i, indices)| {
                Triangle::new(*indices, *normals.get(i).unwrap_or(&[0.0, 0.0, 0.0]))
            })
            .collect();

        out.push(Object::from_mesh(Mesh3D::new(
            vertices,
            triangles,
            shader.clone(),
        )));
    }
    out
}

fn gltf_load_model(
    document: gltf::Document,
    buffers: Vec<gltf::buffer::Data>,
) -> BTreeMap<usize, (Vec<f32>, Vec<u32>, Vec<f32>, Vec<f32>)> {
    let buffer = buffers.get(0).unwrap(); // Not sure what to do with the other buffers

    let mut vertex_map: BTreeMap<usize, (Vec<f32>, Vec<u32>, Vec<f32>, Vec<f32>)> = BTreeMap::new();
    let default_vertex_map = (vec![], vec![], vec![], vec![]);

    let vertex_accessors = get_gltf_accessors(&document, gltf::Semantic::Positions);
    let normal_accessors = get_gltf_accessors(&document, gltf::Semantic::Normals);
    let tex_accessors = get_gltf_accessors(&document, gltf::Semantic::TexCoords(0));
    let index_accessors = get_gltf_index_accessors(&document);

    for (i, accessor) in vertex_accessors.iter().enumerate() {
        let old_key = vertex_map
            .get(&i)
            .unwrap_or_else(|| &default_vertex_map)
            .clone();

        match accessor.data_type() {
            ComponentType::I8 => {
                let intermediate: Vec<i8> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.1,
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::U8 => {
                let intermediate: Vec<u8> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.1,
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::I16 => {
                let intermediate: Vec<i16> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.1,
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::U16 => {
                let intermediate: Vec<u16> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.1,
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::U32 => {
                let intermediate: Vec<u32> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.1,
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::F32 => {
                let intermediate: Vec<f32> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.1,
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
        }
    }

    for (i, accessor) in normal_accessors.iter().enumerate() {
        let old_key = vertex_map
            .get(&i)
            .unwrap_or_else(|| &default_vertex_map)
            .clone();

        match accessor.data_type() {
            ComponentType::I8 => {
                let intermediate: Vec<i8> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.3,
                    ),
                );
            }
            ComponentType::U8 => {
                let intermediate: Vec<u8> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.3,
                    ),
                );
            }
            ComponentType::I16 => {
                let intermediate: Vec<i16> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.3,
                    ),
                );
            }
            ComponentType::U16 => {
                let intermediate: Vec<u16> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.3,
                    ),
                );
            }
            ComponentType::U32 => {
                let intermediate: Vec<u32> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.3,
                    ),
                );
            }
            ComponentType::F32 => {
                let intermediate: Vec<f32> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        intermediate.iter().map(|f| *f as f32).collect(),
                        old_key.3,
                    ),
                );
            }
        }
    }

    for (i, accessor) in tex_accessors.iter().enumerate() {
        let old_key = vertex_map
            .get(&i)
            .unwrap_or_else(|| &default_vertex_map)
            .clone();

        match accessor.data_type() {
            ComponentType::I8 => {
                let intermediate: Vec<i8> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        old_key.2,
                        intermediate.iter().map(|f| *f as f32).collect(),
                    ),
                );
            }
            ComponentType::U8 => {
                let intermediate: Vec<i8> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        old_key.2,
                        intermediate.iter().map(|f| *f as f32).collect(),
                    ),
                );
            }
            ComponentType::I16 => {
                let intermediate: Vec<i16> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        old_key.2,
                        intermediate.iter().map(|f| *f as f32).collect(),
                    ),
                );
            }
            ComponentType::U16 => {
                let intermediate: Vec<u16> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        old_key.2,
                        intermediate.iter().map(|f| *f as f32).collect(),
                    ),
                );
            }
            ComponentType::U32 => {
                let intermediate: Vec<u32> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        old_key.2,
                        intermediate.iter().map(|f| *f as f32).collect(),
                    ),
                );
            }
            ComponentType::F32 => {
                let intermediate: Vec<f32> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        old_key.1,
                        old_key.2,
                        intermediate.iter().map(|f| *f as f32).collect(),
                    ),
                );
            }
        }
    }

    for (i, accessor) in index_accessors.iter().enumerate() {
        print_accessor(&accessor, 3);
        let old_key = vertex_map
            .get(&i)
            .unwrap_or_else(|| &default_vertex_map)
            .clone();

        match accessor.data_type() {
            ComponentType::I8 => {
                let intermediate: Vec<i8> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        intermediate.iter().map(|f| *f as u32).collect(),
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::U8 => {
                let intermediate: Vec<u8> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        intermediate.iter().map(|f| *f as u32).collect(),
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::I16 => {
                let intermediate: Vec<i16> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        intermediate.iter().map(|f| *f as u32).collect(),
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::U16 => {
                let intermediate: Vec<u16> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        intermediate.iter().map(|f| *f as u32).collect(),
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::U32 => {
                let intermediate: Vec<u32> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        intermediate.iter().map(|f| *f as u32).collect(),
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
            ComponentType::F32 => {
                let intermediate: Vec<f32> = gltf_get_accessor_data(&accessor, &buffer);
                vertex_map.insert(
                    i,
                    (
                        old_key.0,
                        intermediate.iter().map(|f| *f as u32).collect(),
                        old_key.2,
                        old_key.3,
                    ),
                );
            }
        }
    }

    vertex_map
}

fn gltf_get_accessor_data<T: bitstream_io::Primitive>(
    accessor: &gltf::Accessor,
    buffer: &gltf::buffer::Data,
) -> Vec<T> {
    let mut out = Vec::new();
    let data_width = match accessor.data_type() {
        ComponentType::I8 => 1,
        ComponentType::U8 => 1,
        ComponentType::I16 => 2,
        ComponentType::U16 => 2,
        ComponentType::U32 => 4,
        ComponentType::F32 => 4,
    };
    let data_dimensions = match accessor.dimensions() {
        Type::Scalar => 1,
        Type::Vec2 => 2,
        Type::Vec3 => 3,
        Type::Vec4 => 4,
        Type::Mat2 => 4,
        Type::Mat3 => 9,
        Type::Mat4 => 16,
    };
    if let Some(view) = accessor.view() {
        let stride = view.stride().unwrap_or(0);
        let buffer_slice = &buffer[view.offset() + accessor.offset()
            ..view.offset() + accessor.offset() + max(accessor.size(), stride) * accessor.count()];
        let mut reader = ByteReader::endian(buffer_slice, LittleEndian);

        for component in 0..accessor.count() {
            for i in 0..data_dimensions {
                let next = reader.read::<T>().unwrap();
                out.push(next);
                if stride != 0 {
                    reader
                        .skip((stride - data_width * data_dimensions) as u32)
                        .unwrap();
                }
            }
        }
    }
    out
}

fn gltf_load_image_from_view(
    view: &View,
    buffer: &gltf::buffer::Data,
) -> Option<ImageBuffer<Rgba<u8>, Vec<u8>>> {
    let source = std::io::Cursor::new(&buffer[view.offset()..view.offset() + view.length()]);
    let decoder = PngDecoder::new(source).unwrap();
    let color_type = decoder.color_type();
    let dimensions = decoder.dimensions();
    let mut destination: Vec<u8> = vec![0; decoder.total_bytes() as usize];
    if let Ok(_) = decoder.read_image(&mut destination) {
        let out = match color_type {
            ColorType::L8 => {
                if let Some(image) = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
                    dimensions.0,
                    dimensions.1,
                    destination,
                ) {
                    Some::<image::ImageBuffer<Rgba<u8>, Vec<u8>>>(image.convert())
                } else {
                    None
                }
            }
            ColorType::La8 => {
                if let Some(image) = ImageBuffer::<LumaA<u8>, Vec<u8>>::from_raw(
                    dimensions.0,
                    dimensions.1,
                    destination,
                ) {
                    Some::<image::ImageBuffer<Rgba<u8>, Vec<u8>>>(image.convert())
                } else {
                    None
                }
            }
            ColorType::Rgb8 => {
                if let Some(image) = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(
                    dimensions.0,
                    dimensions.1,
                    destination,
                ) {
                    Some::<image::ImageBuffer<Rgba<u8>, Vec<u8>>>(image.convert())
                } else {
                    None
                }
            }
            ColorType::Rgba8 => {
                if let Some(image) = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(
                    dimensions.0,
                    dimensions.1,
                    destination,
                ) {
                    Some::<image::ImageBuffer<Rgba<u8>, Vec<u8>>>(image.convert())
                } else {
                    None
                }
            }
            ColorType::L16 => None, // Non-8 bit colors are not supported currently
            ColorType::La16 => None,
            ColorType::Rgb16 => None,
            ColorType::Rgba16 => None,
            ColorType::Rgb32F => None,
            ColorType::Rgba32F => None,
            _ => None,
        };
        if let Some(_) = out {
            println!("Successfully decoded image");
            out
        } else {
            println!("Failed to decode image");
            None
        }
    } else {
        println!("Failed to read image");
        None
    }
}

fn get_gltf_accessors(
    document: &gltf::Document,
    desired_semantic: gltf::Semantic,
) -> Vec<gltf::Accessor<'_>> {
    let mut accessors = Vec::new();
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            if let Some(accessor) = primitive.get(&desired_semantic) {
                accessors.push(accessor);
            }
        }
    }
    accessors
}

fn get_gltf_index_accessors(document: &gltf::Document) -> Vec<gltf::Accessor> {
    let mut accessors = Vec::new();
    for mesh in document.meshes() {
        for primitive in mesh.primitives() {
            if let Some(accessor) = primitive.indices() {
                accessors.push(accessor);
            }
        }
    }
    accessors
}

/// Recursively prints the contents of a GLTF node.
/// `depth` is the current depth, used for indentations, not a recursion limit.
fn walk_gltf_nodes(node: &gltf::Node, depth: usize) {
    println!(
        "{:width$}Loaded Node {} with {} child nodes",
        "",
        node.name().unwrap_or("[unnammed node]"),
        node.children().len(),
        width = depth
    );
    if let Some(mesh) = node.mesh() {
        println!(
            "{:width$}Loaded Mesh {} with {} primitives",
            "",
            mesh.name().unwrap_or("[unnammed mesh]"),
            mesh.primitives().len(),
            width = depth + 1
        );
        // Primitives are assumed to be triangles
        // Unsure how this behaves if they are any other type
        for primitive in mesh.primitives() {
            println!(
                "{:width$}Loaded {:?} primitive with {} attributes",
                "",
                primitive.mode(),
                primitive.attributes().len(),
                width = depth + 2
            );

            if let Some(index) = primitive.material().index() {
                println!(
                    "{:width$}Primitive uses {} material with index {}",
                    "",
                    primitive.material().name().unwrap_or("[unnammed material]"),
                    index,
                    width = depth + 2
                );
            }

            println!(
                "{:width$}Primitive has {} morph targets",
                "",
                primitive.morph_targets().len(),
                width = depth + 2
            );

            if let Some(accessor) = primitive.indices() {
                println!(
                    "{:width$}Loaded index accessor with {} indices",
                    "",
                    accessor.count(),
                    width = depth + 3
                );
                print_accessor(&accessor, depth + 3)
            }

            for (semantic, accessor) in primitive.attributes() {
                println!(
                    "{:width$}Loaded semantic {:?}",
                    "",
                    semantic,
                    width = depth + 3
                );
                print_accessor(&accessor, depth + 3);
            }
        }
    }
    if let Some(camera) = node.camera() {
        println!(
            "{:width$}Loaded Camera {}",
            "",
            camera.name().unwrap_or("[unnammed camera]"),
            width = depth
        );
    }
    for node in node.children() {
        walk_gltf_nodes(&node, depth + 1);
    }
}

fn print_material(material: &gltf::Material, depth: usize) {
    if material.index().is_none() {
        println!(
            "{:depth$}Material {} is using the default material",
            "",
            material.name().unwrap_or("[unnammed material]"),
        );
        return;
    }

    println!("{:depth$}Material index {}", "", material.index().unwrap());
    let depth = depth + 1;
    if let Some(alpha_cutoff) = material.alpha_cutoff() {
        println!("{:depth$}Alpha cutoff {}", "", alpha_cutoff)
    }
    println!("{:depth$}Alpha mode {:?}", "", material.alpha_mode());
    println!("{:depth$}Double sided {}", "", material.double_sided());
    //println!("{:depth$}Metallic roughness {}", "", ""); // TODO
    //println!("{:depth$}Specular glossiness {}", "" , ""); // TODO
    println!(
        "{:depth$}Normal texture {}",
        "",
        material.normal_texture().is_some()
    );
    println!(
        "{:depth$}Occlusion texture {}",
        "",
        material.occlusion_texture().is_some()
    );
    println!(
        "{:depth$}Emissive texture {}",
        "",
        material.emissive_texture().is_some()
    );
    println!(
        "{:depth$}Emissive factor {:?}",
        "",
        material.emissive_factor()
    );
}

fn print_texture(texture: &gltf::Texture, depth: usize) {
    println!(
        "{:depth$}Loaded texture {} with index {}",
        "",
        texture.name().unwrap_or("[unnammed texture]"),
        texture.index()
    );
    let depth = depth + 1;
    println!(
        "{:depth$}Loaded sampler {} with index {}",
        "",
        texture.sampler().name().unwrap_or("[unnammed sampler]"),
        texture.index()
    );
    println!(
        "{:depth$}Loaded texture from source {}",
        "",
        texture
            .source()
            .name()
            .unwrap_or("[unnammed texture source]")
    );
    match texture.source().source() {
        Source::View { view, mime_type } => {
            println!("{:depth$}Texture is located in a view", "");
            println!("{:depth$}MIME type is {}", "", mime_type);
            print_view(&view, depth + 1);
        }
        Source::Uri { uri, mime_type } => {
            println!("{:depth$}Texture is located in a URI", "");
            if let Some(mime) = mime_type {
                println!("{:depth$}MIME type is {}", "", mime);
            }
            println!("{:depth$}{}", "", uri, depth = depth + 1);
        }
    }
}

fn print_accessor(accessor: &gltf::Accessor, depth: usize) {
    println!(
        "{:depth$}Loaded accessor {}",
        "",
        accessor.name().unwrap_or("[unnammed accessor]"),
    );
    if let Some(sparse) = accessor.sparse() {
        println!("{:depth$}Accessor is sparse", "");
    }
    if let Some(view) = accessor.view() {
        print_view(&view, depth + 1);
    }
    println!(
        "{:depth$}Accessor offset {}",
        "",
        accessor.offset(),
        depth = depth + 1
    );
    println!(
        "{:depth$}Accessor data type {:?}",
        "",
        accessor.data_type(),
        depth = depth + 1
    );
    println!(
        "{:depth$}Accessor component size {}",
        "",
        accessor.size(),
        depth = depth + 1
    );
    println!(
        "{:depth$}Accessor count {}",
        "",
        accessor.count(),
        depth = depth + 1
    );
}

fn print_view(view: &View, depth: usize) {
    println!("{:depth$}View offset {}", "", view.offset(),);
    println!("{:depth$}View size {}", "", view.length(),);
    if let Some(stride) = view.stride() {
        println!("{:depth$}View stride {}", "", stride,);
    }
}
