
%% all the commands to run to generate the graphs
name_of_cell_variable = mapped_syllable_order_latest_pre_lesion_songs;
mapped_syllable_order_cell = mat2cell(squeeze(name_of_cell_variable), ones(size(name_of_cell_variable, 1), 1), size(name_of_cell_variable, 3));
mapped_syllable_order_cell = mapped_syllable_order_cell';

%remove trailing spaces after the string (export error from Python -> .mat
%file saving)
for i = 1:length(mapped_syllable_order_cell)
    % Remove trailing spaces from each string in the cell array
    mapped_syllable_order_cell{i} = strtrim(mapped_syllable_order_cell{i});
end

% Verify the result by displaying the first few entries
mapped_syllable_order_cell(1:5)

[F_MAT, ALPHABET, N, PI] = pst_build_trans_mat(mapped_syllable_order_cell,2);
TREE = pst_learn(F_MAT, ALPHABET, N,'l',2);
%% To export the PST to cytoscape:
pst_export_to_cytoscape(TREE,ALPHABET)

%% if going to graphvis
PFA=pst_convert_to_pfa(TREE,ALPHABET);
pst_pfa_export_to_graphviz(PFA); %This exports the PFA as a graph
clear all
%This creates the png file from the pfa_export
%print("In your Mac terminal, run the following: dot -Tpng pfa_graphviz_export.dot -o exported_graphvis_image.png")
% dot -Tpng early-pre_pfa_graphviz_export.dot -o early-pre_pfa_graphviz_export.png