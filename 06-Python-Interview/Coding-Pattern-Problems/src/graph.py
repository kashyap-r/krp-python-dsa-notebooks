""" 
Now implement the Graph using the Adjacency List 
The Graph
        A
       / \
      B   C
      |   |
      D---E

Below graph stored:
A → B, C
B → A, D
C → A, E
D → B, E
E → C, D

As a Set: 
{
    "A": {"B", "C"},
    "B": {"A", "D"},
    "C": {"A", "E"},
    "D": {"B", "E"},
    "E": {"C", "D"}
}

"""

class GraphMatrix:
    def __init__(self):
        self.vertices = [] 
        self.matrix = [] 

    def add_vertex(self, vertex):
        if vertex in self.vertices:
            return 
        # now add the vertex to the list of vertices
        self.vertices.append(vertex)

        # add one column to the existing rows 
        for row in self.matrix:
            row.append(0)

        # Add new row for the new vertex
        new_row = [0] * len(self.vertices)
        self.matrix.append(new_row)

    def add_edge(self, vertex1, vertex2):
        if vertex1 not in self.vertices or vertex2 not in self.vertices:
            raise ValueError("Both vertices must be in the graph.")

        i = self.vertices.index(vertex1)
        j = self.vertices.index(vertex2)
        self.matrix[i][j] = 1   
        self.matrix[j][i] = 1

    def get_neighbors(self, vertex):
        if vertex not in self.vertices:
            raise ValueError("Vertex not found in the graph.")

        index = self.vertices.index(vertex)
        neighbors = []
        for j in range(len(self.vertices)):
            if self.matrix[index][j] == 1:
                neighbors.append(self.vertices[j])
        
        return neighbors

    def remove_edge(self, vertex1, vertex2):
        if vertex1 not in self.vertices or vertex2 not in self.vertices:
            raise ValueError("Both vertices must be in the graph.")

        i = self.vertices.index(vertex1)
        j = self.vertices.index(vertex2)
        self.matrix[i][j] = 0  
        self.matrix[j][i] = 0

    def remove_vertex(self, vertex):
        if vertex not in self.vertices:
            raise ValueError("Vertex not found in the graph.")

        index = self.vertices.index(vertex)

        # remove vertex from the list of vertices 
        # and remove the corresponding row and column from the adjacency matrix 
        self.vertices.pop(index)

        # remove the corresponding row and column from the adjacency matrix
        self.matrix.pop(index)  

        # remove the corresponding column from each remaining row
        for row in self.matrix:
            row.pop(index)

    def has_edge(self, vertex1, vertex2):
        if vertex1 not in self.vertices or vertex2 not in self.vertices:
            return False 

        i = self.vertices.index(vertex1)
        j = self.vertices.index(vertex2)

        return self.matrix[i][j] == 1
    
    def display(self):
        print("  ", *self.vertices)
        for i, row in enumerate(self.matrix):
            print(self.vertices[i], row)

class GraphList: 
    def __init__(self):
        self.graph = {}

    def add_vertex(self, vertex):
        if vertex not in self.graph:
            self.graph[vertex] = set()

    def add_edge(self, vertex1, vertex2):
        if vertex1 not in self.graph or vertex2 not in self.graph:
            return 

        self.graph[vertex1].add(vertex2)
        self.graph[vertex2].add(vertex1)

    def remove_edge(self, vertex1, vertex2):
        if vertex1 not in self.graph or vertex2 not in self.graph:
            return 

        self.graph[vertex1].discard(vertex2)
        self.graph[vertex2].discard(vertex2)

    def remove_vertex(self, vertex):
        if vertex not in self.graph:
            return 

        # remove vertex from its beighbors 
        for neighbor in self.graph[vertex]:
            self.graph[neighbor].discard(vertex)

        # remove the vertex itself 
        del self.graph[vertex]

    def get_neighbors(self, vertex):
        if vertex not in self.graph:
            return []
        return list(self.graph[vertex])

    def has_edge(self, vertex1, vertex2):
        if vertex1 not in self.graph:
            return False
        return vertex2 in self.graph[vertex1]

    def display(self):
        for vertex, neighbors in self.graph.items():
            print(vertex, "->", neighbors)

def test_GraphMatrix():
    g = GraphMatrix()
    
    for vertex in ['A', 'B', 'C', 'D', 'E']:
        g.add_vertex(vertex)

    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("B", "D")
    g.add_edge("C", "E")
    g.add_edge("D", "E")

    g.display()

    print("Neighbors of A:", g.get_neighbors("A"))
    print("A-C:", g.has_edge("A", "C"))
    print("A-E:", g.has_edge("A", "E"))

def test_GraphList():
    g = GraphList()

    for vertex in ["A", "B", "C", "D", "E"]:
        g.add_vertex(vertex)

    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("B", "D")
    g.add_edge("C", "E")
    g.add_edge("D", "E")
    g.display()

if __name__ == "__main__":
    # test_GraphMatrix()
    test_GraphList()

"""
                  SAME GRAPH

             A
            / \
           B   C
           |   |
           D---E

         /         \
        /           \
       ↓             ↓
Adjacency Matrix    Adjacency List

  A B C D E          A → B,C
A 0 1 1 0 0          B → A,D
B 1 0 0 1 0          C → A,E
C 1 0 0 0 1          D → B,E
D 0 1 0 0 1          E → C,D
E 0 0 1 1 0
"""
