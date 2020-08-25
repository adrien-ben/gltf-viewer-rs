use super::vertex::ModelVertex;

const VERTEX_PER_FACE: usize = 3;

type Face = [u32; 3];

struct Mesh<'a> {
    faces: Vec<Face>,
    vertices: &'a mut [ModelVertex],
}

impl<'a> Mesh<'a> {
    fn get_vertex(&self, face: usize, vert: usize) -> ModelVertex {
        let face = self.faces[face];
        self.vertices[face[vert] as usize]
    }

    fn get_vertex_mut(&mut self, face: usize, vert: usize) -> &mut ModelVertex {
        let face = self.faces[face];
        &mut self.vertices[face[vert] as usize]
    }
}

impl<'a> mikktspace::Geometry for Mesh<'a> {
    fn num_faces(&self) -> usize {
        self.faces.len()
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        VERTEX_PER_FACE
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.get_vertex(face, vert).position
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.get_vertex(face, vert).normal
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.get_vertex(face, vert).tex_coords_0
    }

    fn set_tangent(
        &mut self,
        tangent: [f32; 3],
        _bi_tangent: [f32; 3],
        _f_mag_s: f32,
        _f_mag_t: f32,
        bi_tangent_preserves_orientation: bool,
        face: usize,
        vert: usize,
    ) {
        let sign = if bi_tangent_preserves_orientation {
            -1.0
        } else {
            1.0
        };
        let vertex = self.get_vertex_mut(face, vert);
        vertex.tangent = [tangent[0], tangent[1], tangent[2], sign];
    }
}

pub fn generate_tangents(indices: Option<&[u32]>, vertices: &mut [ModelVertex]) {
    log::info!("Generating tangents");

    let index_count = indices.map_or(0, |indices| indices.len());
    if !can_generate_inputs(index_count, vertices.len()) {
        log::warn!("Tangents won't be generated");
        return;
    }

    let faces = if let Some(indices) = indices {
        (0..index_count)
            .step_by(VERTEX_PER_FACE)
            .map(|i| [indices[i], indices[i + 1], indices[i + 2]])
            .collect::<Vec<_>>()
    } else {
        let vertex_count = vertices.len() as u32;
        (0..vertex_count)
            .step_by(VERTEX_PER_FACE)
            .map(|i| [i, i + 1, i + 2])
            .collect::<Vec<_>>()
    };

    let mut mesh = Mesh { faces, vertices };

    mikktspace::generate_tangents(&mut mesh);
}

fn can_generate_inputs(index_count: usize, vertex_count: usize) -> bool {
    if vertex_count == 0 {
        log::warn!("A primitive must have at least 1 vertex in order to generate tangents");
        return false;
    }

    if index_count > 0 && index_count % VERTEX_PER_FACE != 0 {
        log::warn!("The number of indices for a given primitive mush be a multiple of 3");
        return false;
    }

    if index_count == 0 && vertex_count % VERTEX_PER_FACE != 0 {
        log::warn!(
            "The number of vertices for a given primitive without indices mush be a multiple of 3"
        );
        return false;
    }

    true
}
