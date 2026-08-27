"""Generate visual scene-graph collages from YOLO-World detections and REACT++ relations."""

import os
import io
import cv2
import numpy as np
import supervision as sv
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLOWorld
import onnxruntime as ort
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server performance
import matplotlib.pyplot as plt
import networkx as nx

# --- STEP 0: Resolve Absolute Path for Models Directory ---
current_file_dir = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.abspath(os.path.join(current_file_dir, "..", "..", "..", "models"))
os.makedirs(model_folder, exist_ok=True)

os.environ['YOLO_HOME'] = model_folder
os.environ['ULTRALYTICS_CONFIG_DIR'] = model_folder

# --- STEP 1: Download Weights programmatically if missing ---
yolo_world_path = os.path.join(model_folder, 'yolov8s-world.pt')
sgg_onnx_path = os.path.join(model_folder, 'react_pp_yolo12m.onnx')

if not os.path.exists(yolo_world_path):
    print("⬇️ Downloading YOLOv8s-World model weights...")

if not os.path.exists(sgg_onnx_path):
    print("⬇️ Downloading REACT++ Scene Graph ONNX weights...")
    hf_hub_download(
        repo_id="maelic/REACTPlusPlus_PSG",
        filename="yolo12m/react_pp_yolo12m.onnx",
        local_dir=model_folder
    )
    downloaded_file = os.path.join(model_folder, "yolo12m", "react_pp_yolo12m.onnx")
    if os.path.exists(downloaded_file):
        os.rename(downloaded_file, sgg_onnx_path)

# Load Models
yolo_world_model = YOLOWorld('yolov8s-world.pt')
sgg_session = ort.InferenceSession(sgg_onnx_path, providers=['CPUExecutionProvider'])

CLASSES_TO_FIND = [
    "tree", "tree branch", "leaves", "foliage", "bush",
    "view outside window", "nature", "plant",
    "cardboard box", "plastic bag", "electronic device", "furniture", 
    "clothing", "trash", "musical instrument", "container", "bottle",
    "person", "pillow"
]
yolo_world_model.set_classes(CLASSES_TO_FIND)

RELATION_PREDICATES = [
    "over", "in front of", "beside", "on", "in", "attached to", "hanging from", 
    "on back of", "falling off", "going down", "sitting on", "sits on", "standing on", 
    "riding", "carrying", "holding", "wearing", "covering", "laying on", "along", 
    "watching", "eating", "sleeping on", "anchored at", "glowing in", "released from", 
    "using", "looking at", "holding hands with", "kissing", "eating from", "swimming in", 
    "flying in", "playing with", "growing on", "cluttered with", "dancing with", 
    "peering into", "bounding over", "leaning on", "lying on", "under", "behind", "next to"
]


def render_graph_as_image(triplets, canvas_size=(640, 480)):
    """
    Renders a node-edge graph diagram (blue object nodes, orange relation nodes)
    similar to standard SGG visual benchmarks.
    """
    G = nx.DiGraph()

    # Build the scene graph structure (Object -> Relation -> Object)
    for idx, (sub, rel, obj) in enumerate(triplets):
        rel_node_id = f"{rel}_{idx}"  # Unique node for relation
        G.add_node(sub, type='object')
        G.add_node(obj, type='object')
        G.add_node(rel_node_id, label=rel, type='relation')

        G.add_edge(sub, rel_node_id)
        G.add_edge(rel_node_id, obj)

    fig, ax = plt.subplots(figsize=(canvas_size[0] / 100, canvas_size[1] / 100), dpi=100)
    ax.set_facecolor('white')

    if len(G.nodes) > 0:
        pos = nx.spring_layout(G, k=1.2, seed=42)

        # Draw Object Nodes (Light Blue)
        obj_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'object']
        if obj_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=obj_nodes, node_color='#87CEEB', node_size=2200, node_shape='o', ax=ax)
        
        # Draw Relation Nodes (Orange)
        rel_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'relation']
        if rel_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=rel_nodes, node_color='#FF7F50', node_size=1800, node_shape='o', ax=ax)

        # Node Labels
        labels = {n: G.nodes[n].get('label', n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold', ax=ax)

        # Edges
        nx.draw_networkx_edges(G, pos, edge_color='#888888', arrows=True, arrowstyle='->', arrowsize=15, ax=ax)

    plt.axis('off')
    plt.tight_layout()

    # Convert Matplotlib canvas to OpenCV BGR image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    plt.close(fig)

    graph_pil = Image.open(buf).convert('RGB')
    graph_bgr = cv2.cvtColor(np.array(graph_pil), cv2.COLOR_RGB2BGR)
    return cv2.resize(graph_bgr, canvas_size)


def run_scene_graph_generator(image, save_output=False, output_path=None):
    """
    Detects objects using YOLO-World, computes relationships using REACT++, 
    and renders a visual Scene Graph Diagram collage.
    """
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    original_image = image.copy()

    # 1. Detect objects via YOLO-World
    results = yolo_world_model.predict(image, conf=0.1, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results).with_nms(threshold=0.5)

    # 2. Extract scene graph relationships via ONNX
    img_resized = cv2.resize(image, (640, 640))
    img_tensor = np.expand_dims(np.transpose(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0, (2, 0, 1)), axis=0)
    input_name = sgg_session.get_inputs()[0].name
    outputs = sgg_session.run(None, {input_name: img_tensor})

    if len(outputs) == 6:
        _, _, _, rel_pairs, rel_labels, rel_scores = outputs
        rel_pairs = rel_pairs[0] if rel_pairs.ndim == 3 else rel_pairs
        rel_labels = rel_labels[0] if rel_labels.ndim == 2 else rel_labels
        rel_scores = rel_scores[0] if rel_scores.ndim == 2 else rel_scores
    elif len(outputs) == 2:
        rel_pairs, rel_logits = outputs[0], outputs[1]
        rel_pairs = rel_pairs[0] if rel_pairs.ndim == 3 else rel_pairs
        rel_logits = rel_logits[0] if rel_logits.ndim == 3 else rel_logits
        rel_labels = np.argmax(rel_logits, axis=-1)
        rel_scores = np.max(rel_logits, axis=-1)
    else:
        rel_pairs, rel_labels, rel_scores = [], [], []

    # 3. Annotate Image with Bounding Boxes
    annotated = original_image.copy()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=5, text_color=sv.Color.BLACK)
    
    labels = [f"#{idx} {CLASSES_TO_FIND[cid]}" for idx, cid in enumerate(detections.class_id)]
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    # 4. Collect Triplets for Node Graph Rendering
    triplets = []
    if len(detections) > 1 and len(rel_pairs) > 0:
        for pair, rel_lbl, score in zip(rel_pairs, rel_labels, rel_scores):
            if score > 0.1:
                sub_idx, obj_idx = int(pair[0]), int(pair[1])
                if sub_idx < len(detections) and obj_idx < len(detections):
                    sub_cls = f"{CLASSES_TO_FIND[detections.class_id[sub_idx]]}_{sub_idx}"
                    obj_cls = f"{CLASSES_TO_FIND[detections.class_id[obj_idx]]}_{obj_idx}"
                    pred_idx = int(rel_lbl)
                    pred = RELATION_PREDICATES[pred_idx] if pred_idx < len(RELATION_PREDICATES) else "related"
                    triplets.append((sub_cls, pred, obj_cls))

    # 5. Render Node Graph Image
    h, w = original_image.shape[:2]
    graph_img = render_graph_as_image(triplets, canvas_size=(w, h // 2))

    if save_output and output_path:
        cv2.imwrite(output_path, annotated)

    # -------- COLLAGE PART --------
    annot_resized = cv2.resize(annotated, (w, h // 2))
    collage = np.vstack((annot_resized, graph_img))
    
    print("Scene Graph Generation completed.")
    return Image.fromarray(cv2.cvtColor(collage, cv2.COLOR_BGR2RGB))