#!/bin/bash
wget -O ./data/visual/vg_images.zip https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip
unzip ./data/visual/vg_images.zip ./data/visual/
wget -O ./data/visual/region_graphs.json.zip https://visualgenome.org/static/data/dataset/region_graphs.json.zip
unzip ./data/visual/region_graphs.json.zip -d ./data/visual/
