use bitstream_io::{ByteRead, ByteReader, LittleEndian};
use color::palette::css;
use gltf::json::accessor::{ComponentType, Type};
use nalgebra::{Rotation3, Transform3, Vector3, max};
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use crate::mesh::{Mesh3D, Triangle};
use crate::shader::{Shader, Vertex3D};

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
            self.children = load_gltf(self.path.clone(), self.shader.clone().unwrap())
        }
    }
}

fn load_gltf(path: PathBuf, shader: Shader) -> Vec<Object> {
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
            .iter().enumerate()
            .map(|(i, vertex)| Vertex3D::new(*vertex, *normals.get(i).unwrap_or(&[0.0, 0.0, 0.0]), css::WHITE))
            .collect();
        let triangles = indices
            .as_chunks::<3>()
            .0
            .iter().enumerate()
            .map(|(i, indices)| Triangle::new(*indices, *normals.get(i).unwrap_or(&[0.0, 0.0, 0.0])))
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
        print_accessor(&accessor, 0);
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

fn get_gltf_accessors(
    document: &gltf::Document,
    desired_semantic: gltf::Semantic,
) -> Vec<gltf::Accessor> {
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

fn walk_gltf_nodes(node: &gltf::Node, depth: usize) {
    println!(
        "{:width$}Loaded Node {} with {} child nodes",
        "",
        node.name().unwrap_or("[unnammed node]"),
        node.children().len(),
        width = depth
    );
    println!(
        "{:width$}Node transform {:?}",
        "",
        node.transform(),
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

fn print_accessor(accessor: &gltf::Accessor, depth: usize) {
    println!(
        "{:width$}Loaded accessor {}",
        "",
        accessor.name().unwrap_or("[unnammed accessor]"),
        width = depth + 3
    );
    if let Some(sparse) = accessor.sparse() {
        println!("{:width$}Accessor is sparse", "", width = depth + 3);
    }
    if let Some(view) = accessor.view() {
        println!(
            "{:width$}Accessor view offset {}",
            "",
            view.offset(),
            width = depth + 3
        );
        println!(
            "{:width$}Accessor view size {}",
            "",
            view.length(),
            width = depth + 3
        );
        if let Some(stride) = view.stride() {
            println!(
                "{:width$}Accessor view stride {}",
                "",
                stride,
                width = depth + 3
            );
        }
    }
    println!(
        "{:width$}Accessor offset {}",
        "",
        accessor.offset(),
        width = depth + 3
    );
    println!(
        "{:width$}Accessor data type {:?}",
        "",
        accessor.data_type(),
        width = depth + 3
    );
    println!(
        "{:width$}Accessor component size {}",
        "",
        accessor.size(),
        width = depth + 3
    );
    println!(
        "{:width$}Accessor count {}",
        "",
        accessor.count(),
        width = depth + 3
    );
}
