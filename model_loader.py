class ModelLoader:

    def __init__(self, filename, swap_yz=False):
        self.vertices = []
        self.normals = []
        self.texture_coords = []
        self.faces = []

        self.filename = filename
        self.swap_yz = swap_yz

        self.parse_model_obj()

    def parse_model_obj(self):
        """Parse model with .OBJ file format."""

        for line in open(self.filename, "r"):
            if line.startswith('#'):
                continue

            values = line.split()

            if not values:
                continue

            line_symbol = values[0]

            # Geometric vertices, with (x, y, z [,w])
            # Note: w is optional and defaults to 1.0
            if line_symbol == 'v':
                v = list(map(float, values[1:4]))
                if self.swap_yz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)

            # Vertex normals in (x,y,z)
            elif line_symbol == 'vn':
                v = list(map(float, values[1:4]))
                if self.swap_yz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)

            # Texture coordinates, in (u, [v ,w])
            elif line_symbol == 'vt':
                self.texture_coords.append(list(map(float, values[1:3])))

            # Polygonal face element
            elif line_symbol == 'f':
                face = []
                texture_coords = []
                norms = []

                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))

                    if len(w) >= 2 and len(w[1]) > 0:
                        texture_coords.append(int(w[1]))
                    else:
                        texture_coords.append(0)

                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)

                self.faces.append((face, norms, texture_coords))
