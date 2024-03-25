clc, clear all;

%% Rectangular Waveguide
% load in mesh data
node_locations = readmatrix("meshes\rectangle\rectangle_locations.csv");
connectivity = readmatrix("meshes\rectangle\rectangle_connectivity.csv");
edge_nodes = readmatrix("meshes\rectangle\rectangle_edge_nodes.csv");
edge_nodes = reshape(edge_nodes, [size(edge_nodes, 1) * size(edge_nodes, 2), 1]);

% derived constants
num_elements = length(connectivity);
num_nodes = length(node_locations);
num_edge_nodes = length(edge_nodes);

% predefine arrays
bs = zeros(num_elements, 3);
cs = zeros(num_elements, 3);
areas = zeros(num_elements, 1);
A = zeros(num_elements, num_elements);
B = zeros(num_elements, num_elements);

%precalculate values
for i=1:num_elements
    % get coordinates of nodes
    coords = get_coordinates(i, connectivity, node_locations);

    % calculate values
    bs(i, 1) = coords(2,2) - coords(3,2);
    bs(i, 2) = coords(3,2) - coords(1,2);
    bs(i, 3) = coords(1,2) - coords(2,2);
    cs(i, 1) = coords(3,1) - coords(2,1);
    cs(i, 2) = coords(1,1) - coords(3,1);
    cs(i, 3) = coords(2,1) - coords(1,1);
    areas(i) = 0.5 * (bs(i, 1) * cs(i, 2) - bs(i, 2) * cs(i, 1));
end

%TE mode assembly
for element=1:num_elements
    for l_idx=1:3
        for k_idx=1:3
            % find global node numbers
            i_idx = connectivity(element, l_idx + 1);
            j_idx = connectivity(element, k_idx + 1);

            % evaluate delta expression
            if l_idx ~= j_idx
                equal_term = 0;
            else
                equal_term = 1;
            end

            % accumulate values
            A(i_idx, j_idx) = A(i_idx, j_idx) + 1 / (4 * areas(element)) * (bs(element, l_idx)*bs(element, k_idx) + cs(element, l_idx)*cs(element,k_idx));
            B(i_idx, j_idx) = B(i_idx, j_idx) + ((1 + equal_term)*areas(element)) / 12;
        end
    end
end

num_eigs = 5;
[eigenvectors, eigenvalues] = eigs(A, B); % solve general eigenvalue problem

%% helper functions

function coords = get_coordinates(element_idx, connectivity, node_locations)
    cons = connectivity(element_idx, :);
    locs1 = node_locations(cons(2), 2:3);
    locs2 = node_locations(cons(3), 2:3);
    locs3 = node_locations(cons(4), 2:3);
    x1 = locs1(1);
    y1 = locs1(2);
    x2 = locs2(1);
    y2 = locs2(2);
    x3 = locs3(1);
    y3 = locs3(2);
    coords = [x1, y1; x2, y2; x3, y3];
end


