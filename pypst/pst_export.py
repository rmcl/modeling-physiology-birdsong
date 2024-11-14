import os
import numpy as np
import math

def pst_export_to_cytoscape(TREE, ALPHABET, **kwargs):
    """
    pst_export_to_cytoscape takes a PST computed by pst_learn
    and generates files suitable for use with Cytoscape.

    Parameters:
    TREE : list
        Structure array returned by pst_learn
    ALPHABET : list or str
        Mapping of phrase identities to rows/columns in frequency table

    Optional Parameters (kwargs):
    output_dir : str
        Directory to store generated files (default: current directory)
    filename : str
        Root name for generated files (default: 'cytoscape_output_tree')
    thresh : float
        Threshold for filtering transitions (default: 1e-5)
    """
    # Default parameters
    output_dir = kwargs.get('output_dir', os.getcwd())
    filename = kwargs.get('filename', 'cytoscape_output_tree')
    thresh = kwargs.get('thresh', 1e-5)

    # Open files for writing
    sif_path = os.path.join(output_dir, f"{filename}.sif")
    noa_gsigma_path = os.path.join(output_dir, f"{filename}.noa")
    cscript_path = os.path.join(output_dir, f"{filename}_script.txt")

    with open(sif_path, 'w') as sif_file, open(noa_gsigma_path, 'w') as noa_gsigma_file, open(cscript_path, 'w') as cscript_file:
        # Remove empty strings from TREE structure
        TREE = [node for node in TREE if node.get('string')]

        # Iterate over TREE to write .sif and .noa files
        for i in range(len(TREE) - 1):
            for j in range(len(TREE[i]['label'])):
                source = TREE[i]['label'][j]

                if not TREE[i + 1]['parent']:
                    continue

                target_idxs = [idx for idx, parent in enumerate(TREE[i + 1]['parent'][0]) if parent == j]
                target_labels = [TREE[i + 1]['label'][idx] for idx in target_idxs]

                sif_file.write(f"{source} trans")
                for target in target_labels:
                    sif_file.write(f" {target}")
                sif_file.write("\n")

        # Prepare node attributes for .noa file
        noa_gsigma_file.write('ID\t')
        if isinstance(ALPHABET, str):
            ALPHABET = list(ALPHABET)

        noa_gsigma_file.write("\t".join(ALPHABET))
        noa_gsigma_file.write("\tFrequency\tLogFrequency\tDepth")

        if 'internal' in TREE[0]:
            noa_gsigma_file.write("\tInternal\n")
            internal_flag = True
        else:
            noa_gsigma_file.write("\n")
            internal_flag = False

        # Iterate again for .noa file and node chart script
        for i, node in enumerate(TREE):
            for j, label in enumerate(node['label']):
                cscript_file.write(f'nodecharts pie nodelist="{label}"')

                noa_gsigma_file.write(f"{label}\t")
                labellist = ''
                colorlist = ''
                valuelist = ''

                for k, value in enumerate(node['g_sigma_s'][:, j]):
                    noa_gsigma_file.write(f"{value:.2f}\t")

                    if value > thresh:
                        labellist += f"{ALPHABET[k]},"
                        colorlist += f"{'#%02X%02X%02X' % tuple(np.random.randint(0, 256, 3))},"
                        valuelist += f"{value},"

                if len(labellist.strip(',')) == 2:
                    labellist += 'null,'
                    valuelist += '0,'
                    colorlist += '#%02X%02X%02X,' % tuple(np.random.randint(0, 256, 3))

                cscript_file.write(f' labellist="{labellist.strip(",")}"')
                cscript_file.write(f' valuelist="{valuelist.strip(",")}"')
                cscript_file.write(f' colorlist="{colorlist.strip(",")}"\n')

                noa_gsigma_file.write(f"{node['f'][j]:g}\t{math.log(node['f'][j]):g}\t{i - 1}")

                if internal_flag:
                    noa_gsigma_file.write(f"\t{node['internal'][j]}")

                noa_gsigma_file.write("\n")
